"""EXP-022: audit a fixed, provenance-preserving E5@100 + BM25@50 union.

The two channels have already completed their own nested OOF configuration
selection.  This stage does not tune a union weight: it preserves the E5 top
100, appends BM25's novel top 50, and measures their complementary coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file, stage_run, write_jsonl
from exp021_sparse_depth_tune import _cascade, _iter_evidence, _rankings


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp022_e5_bm25_union_audit.v1"
E5_K = 100
BM25_K = 50
LEGAL_IDENTIFIER = re.compile(
    r"\b(?:điều|khoản|mục|chương)\s*\d+[\w.-]*|\b\d{1,4}/\d{2,4}/[\w-]+\b", re.IGNORECASE
)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _code_sha256() -> str:
    return sha256_file(Path(__file__))


def _rank_map(documents: Sequence[str]) -> dict[str, int]:
    return {str(doc_id): rank for rank, doc_id in enumerate(documents, 1)}


def fixed_union(e5_documents: Sequence[str], bm25_documents: Sequence[str]) -> list[str]:
    """Keep E5@100, add novel BM25, then backfill from E5@101-150."""
    anchor = [str(doc_id) for doc_id in e5_documents[:E5_K]]
    if len(anchor) != E5_K or len(set(anchor)) != E5_K:
        raise RuntimeError("E5 anchor must contain exactly 100 unique parent document IDs")
    novel = [str(doc_id) for doc_id in bm25_documents if str(doc_id) not in set(anchor)][:BM25_K]
    result = anchor + novel
    result.extend(str(doc_id) for doc_id in e5_documents[E5_K:] if str(doc_id) not in set(result))
    if len(result) < E5_K + BM25_K or len(set(result[:E5_K + BM25_K])) != E5_K + BM25_K:
        raise RuntimeError("E5 ranking cannot backfill the fixed 150-document union budget")
    return result[:E5_K + BM25_K]


def _recall(prediction: Sequence[str], gold: set[str]) -> float | None:
    return None if not gold else len(set(prediction) & gold) / len(gold)


def _mean(values: Iterable[float | None]) -> float:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else 0.0


def _quantiles(values: list[int]) -> dict[str, float | int]:
    return {
        "total": int(sum(values)), "mean": float(np.mean(values)), "min": int(min(values)),
        "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90)), "max": int(max(values)),
    }


def _slice_name(query: str) -> str:
    return "legal_identifier" if LEGAL_IDENTIFIER.search(query) else "no_legal_identifier"


def _markdown_examples(samples: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    labels = {
        "bm25_only_gold": "BM25 cứu, E5@100 bỏ sót",
        "e5_only_gold": "E5 cứu, BM25@50 bỏ sót",
        "bm25_beyond_e5_150": "BM25 novel cứu ngoài cả E5 Top-150",
        "both_gold": "Cả hai cùng có gold",
        "neither_gold": "Cả hai cùng bỏ sót gold",
    }
    lines = ["# EXP-022 — E5@100 + BM25@50 union audit", "", "## Mẫu query/gold để đọc", ""]
    for name, title in labels.items():
        lines.extend([f"### {title}", "", "| QID | Query | Gold document | E5 rank | BM25 rank |", "|---|---|---|---:|---:|"])
        for row in samples.get(name, []):
            query = str(row["query"]).replace("|", "\\|").replace("\n", " ")
            label = str(row["document_label"]).replace("|", "\\|").replace("\n", " ")
            if len(query) > 180:
                query = query[:177] + "..."
            if len(label) > 90:
                label = label[:87] + "..."
            lines.append(f"| {row['qid']} | {query} | {row['gold_doc_id']} — {label} | {row['e5_rank'] or '—'} | {row['bm25_rank'] or '—'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_audit(*, dense_candidates: Path, sparse_evidence_dir: Path, sparse_tuning_report: Path,
              train_path: Path, folds_path: Path, preprocessing_dir: Path, documents_path: Path,
              cache_dir: Path, results_dir: Path) -> dict[str, Any]:
    report = json.loads(sparse_tuning_report.read_text(encoding="utf-8"))
    selected = report["selected_by_candidate_budget"].get(str(BM25_K))
    if not selected:
        raise RuntimeError("Sparse tuning report has no nested selection for BM25@50")
    folds = {name: set(map(str, values)) for name, values in json.loads(folds_path.read_text(encoding="utf-8")).items()}
    fold_for = {qid: name for name, qids in folds.items() for qid in qids}
    train = json.loads(train_path.read_text(encoding="utf-8"))
    answers = {str(qid): set(map(str, row["answer"])) for qid, row in train.items()}
    excluded = {str(row["doc_id"]) for row in json.loads((preprocessing_dir / "exclusions.json").read_text(encoding="utf-8"))}
    docs = {str(row["doc_id"]): row for row in read_jsonl(documents_path)}

    dense = {str(row["qid"]): row for row in read_jsonl(dense_candidates)}
    if set(dense) != set(answers) or set(fold_for) != set(answers):
        raise RuntimeError("Dense candidates, fixed folds, and train labels must cover the same queries")
    if {str(row["fold"]) for row in dense.values()} != set(folds):
        raise RuntimeError("Dense candidates do not contain every fixed fold")

    sparse: dict[str, list[str]] = {}
    sparse_passage_ranks: dict[str, dict[str, list[int]]] = {}
    for row in _iter_evidence(sparse_evidence_dir / "shards"):
        qid = str(row["qid"])
        params = selected[fold_for[qid]]
        first, rrf = _rankings(row["evidence"], int(params["depth"]), int(params["parent_rrf_k"]))
        sparse[qid] = _cascade(first, rrf, int(params["fusion_rrf_k"]), int(params["head_cutoff"]))
        sparse_passage_ranks[qid] = {str(doc_id): list(map(int, ranks)) for doc_id, ranks in row["evidence"]}
    if set(sparse) != set(answers):
        raise RuntimeError("Sparse raw evidence does not cover every train query")

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    union_rows: list[dict[str, Any]] = []
    pair_overlap: list[int] = []; e5_only: list[int] = []; bm25_only: list[int] = []
    raw_union_sizes: list[int] = []; union_sizes: list[int] = []; bm25_novel_appended: list[int] = []
    metrics: dict[str, dict[str, list[float | None]]] = {
        "all_gold": {"e5@100": [], "bm25@50": [], "raw_top50_union<=150": [], "filled_union=150": []},
        "retained_gold": {"e5@100": [], "bm25@50": [], "raw_top50_union<=150": [], "filled_union=150": []},
    }
    outcomes = Counter(); slices: dict[str, Counter[str]] = defaultdict(Counter)
    bm25_beyond_e5_150 = Counter(); bm25_beyond_slices: dict[str, int] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with stage_run(results_dir, "e5-bm25-union-audit", total=len(answers)) as logger:
        for position, qid in enumerate(sorted(answers), 1):
            dense_row = dense[qid]
            e5_all_candidates = list(dense_row["candidates"])
            e5_candidates = e5_all_candidates[:E5_K]
            e5_ids = [str(item["doc_id"]) for item in e5_candidates]
            bm25_ids = sparse[qid][:BM25_K]
            raw_union_ids = e5_ids + [doc_id for doc_id in bm25_ids if doc_id not in set(e5_ids)]
            union_ids = fixed_union([str(item["doc_id"]) for item in e5_all_candidates], sparse[qid])
            novel_bm25_ids = [doc_id for doc_id in sparse[qid] if doc_id not in set(e5_ids)][:BM25_K]
            e5_ranks, e5_full_ranks, bm25_ranks = _rank_map(e5_ids), _rank_map([str(item["doc_id"]) for item in e5_all_candidates]), _rank_map(bm25_ids)
            shared = set(e5_ids) & set(bm25_ids)
            pair_overlap.append(len(shared)); e5_only.append(len(set(e5_ids) - set(bm25_ids)))
            bm25_only.append(len(set(bm25_ids) - set(e5_ids))); raw_union_sizes.append(len(raw_union_ids)); union_sizes.append(len(union_ids))
            bm25_novel_appended.append(len(novel_bm25_ids))
            source = {str(item["doc_id"]): item for item in e5_all_candidates}
            candidates: list[dict[str, Any]] = []
            for rank, doc_id in enumerate(union_ids, 1):
                item = source.get(doc_id, {})
                candidates.append({
                    "doc_id": doc_id, "rank": rank,
                    "sources": {
                        **({"e5": {"rank": e5_full_ranks[doc_id], "aggregate_score": float(item["aggregate_score"]), "evidence": item["evidence"]}} if doc_id in e5_full_ranks else {}),
                        **({"bm25": {"rank": bm25_ranks[doc_id], "passage_ranks": sparse_passage_ranks[qid].get(doc_id, [])}} if doc_id in bm25_ranks else {}),
                    },
                })
            union_rows.append({"qid": qid, "query": str(dense_row["query"]), "fold": fold_for[qid], "policy": "e5@100_then_bm25@50_novel", "candidates": candidates})
            gold_all = answers[qid]; gold_retained = gold_all - excluded
            for denominator, gold in (("all_gold", gold_all), ("retained_gold", gold_retained)):
                metrics[denominator]["e5@100"].append(_recall(e5_ids, gold))
                metrics[denominator]["bm25@50"].append(_recall(bm25_ids, gold))
                metrics[denominator]["raw_top50_union<=150"].append(_recall(raw_union_ids, gold))
                metrics[denominator]["filled_union=150"].append(_recall(union_ids, gold))
            category = _slice_name(str(dense_row["query"]))
            for gold_id in sorted(gold_retained):
                in_e5, in_bm25 = gold_id in e5_ranks, gold_id in bm25_ranks
                outcome = "both_gold" if in_e5 and in_bm25 else "e5_only_gold" if in_e5 else "bm25_only_gold" if in_bm25 else "neither_gold"
                outcomes[outcome] += 1; slices[category][outcome] += 1
                record = {
                    "qid": qid, "query": str(dense_row["query"]), "gold_doc_id": gold_id,
                    "document_label": str(docs.get(gold_id, {}).get("document_label", "missing")),
                    "parse_mode": str(docs.get(gold_id, {}).get("parse_mode", "missing")),
                    "legal_identifier": category == "legal_identifier", "e5_rank": e5_ranks.get(gold_id),
                    "bm25_rank": bm25_ranks.get(gold_id),
                }
                samples[outcome].append(record)
                if gold_id not in e5_full_ranks and gold_id in novel_bm25_ids:
                    bm25_beyond_e5_150["occurrences"] += 1
                    bm25_beyond_slices[category] += 1
                    samples["bm25_beyond_e5_150"].append(record)
            if position % 256 == 0 or position == len(answers):
                logger.status(stage="e5-bm25-union-audit", state="RUNNING", completed=position, total=len(answers))
                logger.log(f"progress={position}/{len(answers)}")

        for rows in samples.values():
            rows.sort(key=lambda row: (999 if row["e5_rank"] is None and row["bm25_rank"] is None else min(value for value in (row["e5_rank"], row["bm25_rank"]) if value is not None), row["qid"], row["gold_doc_id"]))
        samples = {name: rows[:8] for name, rows in sorted(samples.items())}
        union_path = cache_dir / "train_oof_candidates.jsonl"
        write_jsonl(union_path, union_rows)
        report_out = {
            "schema_version": SCHEMA, "status": "PASS", "policy": {"e5_anchor": E5_K, "bm25_raw_overlap_depth": BM25_K, "bm25_novel_append_max": BM25_K, "dense_backfill": "E5 ranks 101-150", "max_union": E5_K + BM25_K, "ranking": "e5 anchor, novel bm25 in rank order, dense tail fallback"},
            "inputs": {"dense_candidates_sha256": sha256_file(dense_candidates), "sparse_tuning_report_sha256": sha256_file(sparse_tuning_report), "sparse_evidence_manifest_sha256": sha256_file(sparse_evidence_dir / "manifest.json"), "folds_sha256": sha256_file(folds_path), "preprocessing_exclusions_sha256": sha256_file(preprocessing_dir / "exclusions.json"), "code_sha256": _code_sha256()},
            "queries": len(answers), "candidate_occurrences": {"both": _quantiles(pair_overlap), "e5_only": _quantiles(e5_only), "bm25_only": _quantiles(bm25_only), "raw_top50_union": _quantiles(raw_union_sizes), "bm25_novel_appended": _quantiles(bm25_novel_appended), "filled_union": _quantiles(union_sizes)},
            "recall": {denominator: {name: _mean(values) for name, values in scores.items()} for denominator, scores in metrics.items()},
            "retained_gold_occurrences": dict(sorted(outcomes.items())),
            "retained_gold_outcomes_by_query_pattern": {name: dict(sorted(values.items())) for name, values in sorted(slices.items())},
            "bm25_incremental_over_e5_150_gold_occurrences": {"total": int(bm25_beyond_e5_150["occurrences"]), "by_query_pattern": dict(sorted(bm25_beyond_slices.items()))},
            "samples_path": "samples.json", "union_candidates_sha256": sha256_file(union_path),
        }
        atomic_json(results_dir / "union_report.json", report_out)
        atomic_json(results_dir / "samples.json", samples)
        (results_dir / "SAMPLES.md").write_text(_markdown_examples(samples), encoding="utf-8")
        manifest = {"schema_version": SCHEMA, "stage": "e5-bm25-union-audit", "content_fingerprint": _hash_json(report_out), "artifact_sha256": {"cache/train_oof_candidates.jsonl": sha256_file(union_path), "results/union_report.json": sha256_file(results_dir / "union_report.json"), "results/samples.json": sha256_file(results_dir / "samples.json"), "results/SAMPLES.md": sha256_file(results_dir / "SAMPLES.md")}}
        atomic_json(cache_dir / "manifest.json", manifest)
        atomic_json(cache_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": "e5-bm25-union-audit", "content_fingerprint": manifest["content_fingerprint"]})
        logger.set_telemetry({"queries": len(answers), "union_mean": report_out["candidate_occurrences"]["filled_union"]["mean"]})
    return report_out


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"):
        __import__("sys").stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-candidates", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates" / "train_oof_candidates.jsonl")
    parser.add_argument("--sparse-evidence-dir", type=Path, default=ROOT / "cache" / "exp021_sparse" / "depth_tune" / "raw4096_evidence")
    parser.add_argument("--sparse-tuning-report", type=Path, default=ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "tuning_report.json")
    parser.add_argument("--train-file", type=Path, default=ROOT / "public_test_dataset" / "train.json")
    parser.add_argument("--folds-file", type=Path, default=ROOT / "cache" / "cv_folds.json")
    parser.add_argument("--preprocessing-dir", type=Path, default=ROOT / "cache" / "final_preprocessed_v2")
    parser.add_argument("--documents-path", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1" / "documents.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "exp022_e5_bm25_union")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "exp022_e5_bm25_union")
    args = parser.parse_args(argv)
    print(json.dumps(run_audit(
        dense_candidates=args.dense_candidates, sparse_evidence_dir=args.sparse_evidence_dir,
        sparse_tuning_report=args.sparse_tuning_report, train_path=args.train_file,
        folds_path=args.folds_file, preprocessing_dir=args.preprocessing_dir,
        documents_path=args.documents_path, cache_dir=args.cache_dir, results_dir=args.results_dir,
    ), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
