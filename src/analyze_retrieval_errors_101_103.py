"""Canonical OOF error audit for EXP-101, EXP-102 and EXP-103.

This is an analysis utility, not a retrieval experiment.  It reconstructs the
existing OOF rankings from frozen artifacts, applies the canonical label policy,
and writes rank-distribution/error-overlap evidence for deciding how small a
retrieval shortlist can safely be.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(r"D:\Study\DSC2026\LegalIR")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import exp030_legal_evidence_routing as exp030
import exp101_procrustes_alignment as exp101
import exp102_mil_nce_retrieval as exp102
import exp103_preamble_citation as exp103

OUT = ROOT / "results" / "retrieval_error_analysis_101_103"
KS = (5, 16, 24, 32, 40, 50, 64, 100, 150)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rank_of(ranking: list[str], doc_id: str) -> int:
    try:
        return ranking.index(doc_id) + 1
    except ValueError:
        return 151


def canonical_metrics(rankings: dict[str, list[str]], labels: dict[str, set[str]]) -> dict[str, Any]:
    qids = sorted(qid for qid, gold in labels.items() if gold)
    recalls = {k: [] for k in KS}
    first_ranks: list[int] = []
    coverage_ranks: list[int] = []
    gold_ranks: list[int] = []
    incomplete = {k: [] for k in KS}
    for qid in qids:
        ranking = rankings[qid]
        ranks = [rank_of(ranking, gold) for gold in sorted(labels[qid])]
        gold_ranks.extend(ranks)
        first_ranks.append(min(ranks))
        coverage_ranks.append(max(ranks))
        for k in KS:
            value = sum(rank <= k for rank in ranks) / len(ranks)
            recalls[k].append(value)
            if value < 1.0:
                incomplete[k].append(qid)
    finite = [rank for rank in gold_ranks if rank <= 150]
    return {
        "recall": {str(k): float(np.mean(recalls[k])) for k in KS},
        "equivalent_missing_query_units": {
            str(k): float(sum(1.0 - value for value in recalls[k])) for k in KS
        },
        "queries_with_incomplete_gold": {str(k): len(incomplete[k]) for k in KS},
        "first_gold_rank": {
            "median": float(np.median(first_ranks)),
            "p90": float(np.percentile(first_ranks, 90)),
            "mean_reciprocal_rank": float(np.mean([1.0 / rank if rank <= 150 else 0.0 for rank in first_ranks])),
        },
        "all_gold_coverage_rank": {
            "median": float(np.median(coverage_ranks)),
            "p90": float(np.percentile(coverage_ranks, 90)),
            "p95": float(np.percentile(coverage_ranks, 95)),
        },
        "gold_occurrences": len(gold_ranks),
        "gold_occurrences_outside_150": sum(rank > 150 for rank in gold_ranks),
        "retrieved_gold_rank_median": float(np.median(finite)) if finite else None,
    }


def build_exp101_rankings() -> dict[str, list[str]]:
    context = exp101.load_corpus_and_queries()
    all_qids = sorted(context["train_data"])
    rankings: dict[str, list[str]] = {}
    for fold_name, heldout_raw in sorted(context["folds"].items()):
        heldout = list(map(str, heldout_raw))
        heldout_set = set(heldout)
        train_qids = [qid for qid in all_qids if qid not in heldout_set]
        x_train, y_train = exp101.build_training_pairs(context, train_qids, target_mode="mean_doc")
        matrix = exp101.solve_orthogonal_procrustes(x_train, y_train)
        indices = [context["qid_to_idx"][qid] for qid in heldout]
        transformed = context["query_embeddings"][indices] @ matrix
        _, fold_rankings = exp101.evaluate_queries(
            context, heldout, transformed, rrf_alpha=0.65, rrf_k=32
        )
        rankings.update(fold_rankings)
        print(f"reconstructed EXP-101 {fold_name}", flush=True)
    del context
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rankings


def build_exp102_103_rankings() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    data = exp102.load_corpus_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
    documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
    graph = json.loads((exp103.CACHE_DIR / "corpus_citation_graph.json").read_text(encoding="utf-8"))["citation_graph"]
    raw: dict[str, list[str]] = {}
    expanded: dict[str, list[str]] = {}
    for fold_name, heldout_raw in sorted(data["folds"].items()):
        heldout = list(map(str, heldout_raw))
        model = exp102.ResidualProjection(exp102.DIMENSION, exp102.RANK).to(device)
        model.load_state_dict(torch.load(exp102.CACHE_DIR / f"{fold_name}.pt", map_location=device))
        model.eval()
        _, fold_rankings = exp102.evaluate_model(
            data, heldout, model, documents, device, rrf_mode="static", static_alpha=0.65, rrf_k=32
        )
        raw.update(fold_rankings)
        for qid, ranking in fold_rankings.items():
            expanded[qid] = exp103.expand_single_ranking(
                ranking, graph, seed_k=3, max_citations_per_seed=3, max_total_citations=4
            )
        print(f"reconstructed EXP-102/103 {fold_name}", flush=True)
    return raw, expanded


def query_family(text: str) -> str:
    lowered = text.lower()
    rules = (
        ("explicit_locator", r"\b(điều|khoản|điểm)\s+\d+|\b\d+/\d+/(?:nđ|tt|qđ|qh)"),
        ("definition", r"thế nào là|là gì|được hiểu|khái niệm|giải thích"),
        ("sanction_liability", r"xử phạt|mức phạt|phạt|truy cứu|trách nhiệm hình sự|bồi thường"),
        ("procedure_deadline", r"thủ tục|hồ sơ|thời hạn|bao lâu|trình tự|cấp phép|đăng ký"),
        ("eligibility_right_duty", r"điều kiện|quyền|nghĩa vụ|được phép|có được|đối tượng nào|ai được"),
    )
    for name, pattern in rules:
        if re.search(pattern, lowered):
            return name
    return "other"


def main() -> None:
    labels, label_stats = exp030.canonical_answers(
        ROOT / "public_test_dataset" / "train.json",
        ROOT / "cache" / "final_preprocessed_v2" / "exclusions.json",
        ROOT / "cache" / "final_preprocessed_v2" / "train_label_impact.jsonl",
    )
    train = json.loads((ROOT / "public_test_dataset" / "train.json").read_text(encoding="utf-8"))
    metadata = {str(row["doc_id"]): row for row in read_jsonl(ROOT / "cache" / "structural_v3_e5_final_v1" / "documents.jsonl")}
    chunk_counts = Counter()
    with (ROOT / "cache" / "e5_final_v1" / "chunk_ids.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk_counts[str(json.loads(line)["doc_id"])] += 1

    sidecars: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"):
        sidecars[str(row["qid"])] = {str(c["doc_id"]): c for c in row["candidates"]}

    exp101_rankings = build_exp101_rankings()
    exp102_rankings, exp103_rankings = build_exp102_103_rankings()
    methods = {"exp101_rrf": exp101_rankings, "exp102_rrf": exp102_rankings, "exp103_citation": exp103_rankings}
    metrics = {name: canonical_metrics(rankings, labels) for name, rankings in methods.items()}

    cases = []
    for qid in sorted(qid for qid, gold in labels.items() if gold):
        for gold in sorted(labels[qid]):
            cand = sidecars[qid].get(gold, {})
            sources = cand.get("sources", {})
            row = {
                "qid": qid,
                "question": train[qid]["question"],
                "query_family": query_family(train[qid]["question"]),
                "gold_doc_id": gold,
                "gold_count": len(labels[qid]),
                "parse_mode": metadata.get(gold, {}).get("parse_mode", "missing"),
                "passage_length": metadata.get(gold, {}).get("passage_length"),
                "chunk_count": chunk_counts.get(gold, 0),
                "exp022_pool_rank": cand.get("rank", 151),
                "e5_rank_exp022": sources.get("e5", {}).get("rank", 151),
                "bm25_rank_exp022": sources.get("bm25", {}).get("rank", 151),
                "exp101_rank": rank_of(exp101_rankings[qid], gold),
                "exp102_rank": rank_of(exp102_rankings[qid], gold),
                "exp103_rank": rank_of(exp103_rankings[qid], gold),
            }
            cases.append(row)

    overlap = {}
    for k in (16, 32, 50, 64):
        missed = {
            name: {row["qid"] for row in cases if row[name.replace("_rrf", "") + "_rank"] > k}
            if name != "exp103_citation" else {row["qid"] for row in cases if row["exp103_rank"] > k}
            for name in methods
        }
        common = set.intersection(*missed.values())
        overlap[str(k)] = {
            "missed_query_counts": {name: len(values) for name, values in missed.items()},
            "missed_by_all_three": len(common),
            "fraction_exp102_misses_also_missed_by_all": len(common) / max(1, len(missed["exp102_rrf"])),
        }

    buckets: dict[str, Any] = {}
    deep = [row for row in cases if row["exp102_rank"] > 32]
    for field in ("query_family", "parse_mode"):
        counts = Counter(row[field] for row in deep)
        buckets[f"exp102_gold_outside_32_by_{field}"] = dict(sorted(counts.items()))
    buckets["exp102_gold_outside_32_by_gold_count"] = dict(sorted(Counter(str(row["gold_count"]) for row in deep).items()))
    buckets["exp102_gold_outside_32_source_visibility"] = dict(sorted(Counter(
        "both" if row["e5_rank_exp022"] <= 100 and row["bm25_rank_exp022"] <= 50 else
        "e5_only" if row["e5_rank_exp022"] <= 100 else
        "bm25_only" if row["bm25_rank_exp022"] <= 50 else "neither"
        for row in deep
    ).items()))

    source_files = [
        SRC / "exp101_procrustes_alignment.py", SRC / "exp102_mil_nce_retrieval.py",
        SRC / "exp103_preamble_citation.py", ROOT / "public_test_dataset" / "train.json",
        ROOT / "cache" / "cv_folds.json",
    ] + [exp102.CACHE_DIR / f"fold_{fold}.pt" for fold in range(5)]
    report = {
        "schema_version": "legalir.retrieval_error_analysis_101_103.v1",
        "purpose": "rank-quality and low-K failure analysis; no model selection claim",
        "canonical_label_stats": label_stats,
        "artifact_provenance": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "provenance_warnings": [
            "EXP-101 original reports used 7000 raw-label queries; this audit re-scores canonical labels.",
            "EXP-102 checkpoints have no adjacent manifest, _SUCCESS marker, optimizer/RNG state, or training entry point in the current source.",
            "EXP-101/102 RRF sorts score-only values originating from a set and has no doc_id tie-break, so byte-stable tie ordering is not guaranteed.",
        ],
        "canonical_metrics": metrics,
        "error_overlap": overlap,
        "error_buckets": buckets,
        "exp103_delta_vs_exp102": {
            str(k): metrics["exp103_citation"]["recall"][str(k)] - metrics["exp102_rrf"]["recall"][str(k)] for k in KS
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "gold_rank_cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(cases, key=lambda value: (value["qid"], value["gold_doc_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"report": str(OUT / "REPORT.json"), "cases": len(cases), "metrics": metrics, "overlap": overlap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
