"""Nested-OOF quota ablation and unresolved-miss handoff for E5 + BM25."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file, stage_run, write_jsonl
from exp021_sparse_depth_tune import _cascade, _iter_evidence, _rankings


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp023_e5_bm25_quota_ablation.v1"
CAP = 150
# All configurations have the same cap.  The first pair is deliberately
# sparse-heavy; the last is dense-heavy, so selection tests complementarity.
QUOTAS = ((50, 100), (70, 80), (100, 50), (120, 30))
# Fixed handoff policy for the *next* pre-reranker experiment.  This is kept
# separate from the fold-specific ablation output below: it is the modal
# nested-CV choice (3/5 folds) and preserves a simple, reproducible cap.
DEPLOYMENT_DEFAULT = (100, 50)
LEGAL_IDENTIFIER = re.compile(r"\b(?:điều|khoản|mục|chương)\s*\d+[\w.-]*|\b\d{1,4}/\d{2,4}/[\w-]+\b", re.IGNORECASE)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _code_sha256() -> str:
    return sha256_file(Path(__file__))


def _rank_map(values: Sequence[str]) -> dict[str, int]:
    return {str(value): rank for rank, value in enumerate(values, 1)}


def build_candidate(e5_ranking: Sequence[str], bm25_ranking: Sequence[str], quota: tuple[int, int]) -> tuple[list[str], int]:
    """Preserve an E5 anchor, add novel BM25, then deterministically backfill."""
    e5_quota, bm25_quota = quota
    anchor = [str(value) for value in e5_ranking[:e5_quota]]
    if len(anchor) != e5_quota or len(set(anchor)) != e5_quota:
        raise RuntimeError("Malformed E5 ranking for quota ablation")
    result = list(anchor)
    novel: list[str] = []
    for value in bm25_ranking:
        doc_id = str(value)
        if doc_id not in set(result):
            result.append(doc_id)
            novel.append(doc_id)
        if len(novel) == bm25_quota:
            break
    if len(result) == CAP:
        return result, len(novel)
    for source in (e5_ranking[e5_quota:], bm25_ranking):
        for doc_id in source:
            doc_id = str(doc_id)
            if doc_id not in set(result):
                result.append(doc_id)
            if len(result) == CAP:
                break
        if len(result) == CAP:
            break
    if len(result) != CAP or len(set(result)) != CAP:
        raise RuntimeError(
            "Cannot fill the fixed 150-document candidate budget "
            f"(e5={len(e5_ranking)}/{len(set(e5_ranking))}, bm25={len(bm25_ranking)}/{len(set(bm25_ranking))}, "
            f"anchor={len(anchor)}, novel={len(novel)}, result={len(result)}/{len(set(result))})"
        )
    return result, len(novel)


def _recall(prediction: Sequence[str], gold: set[str]) -> float:
    return len(set(prediction) & gold) / len(gold) if gold else 0.0


def _metric(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Sequence[str]) -> tuple[float, float]:
    recalls: list[float] = []; mrrs: list[float] = []
    for qid in qids:
        gold = answers[qid]
        if not gold:
            continue
        prediction = predictions[qid]
        recalls.append(_recall(prediction, gold))
        first = next((rank for rank, doc_id in enumerate(prediction, 1) if doc_id in gold), None)
        mrrs.append(0.0 if first is None else 1.0 / first)
    return sum(recalls) / len(recalls), sum(mrrs) / len(mrrs)


def _sample_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# EXP-023 remaining candidate misses", "", "These are retained gold documents absent from the selected E5+BM25 pool of 150. They are handoff cases for future Qwen and lexical/query-memory channels, not evidence that either channel will rescue them.", "", "| QID | Class | Query | Gold document | E5 rank (top150) | BM25 rank |", "|---|---|---|---|---:|---:|"]
    for row in rows[:20]:
        query = str(row["query"]).replace("|", "\\|").replace("\n", " ")
        label = str(row["document_label"]).replace("|", "\\|").replace("\n", " ")
        if len(query) > 180: query = query[:177] + "..."
        if len(label) > 80: label = label[:77] + "..."
        lines.append(f"| {row['qid']} | {row['miss_type']} | {query} | {row['gold_doc_id']} — {label} | {row['e5_rank'] or '—'} | {row['bm25_rank'] or '—'} |")
    return "\n".join(lines) + "\n"


def _report_markdown(report: Mapping[str, Any]) -> str:
    """Render a human handoff without making generated JSON the only record."""
    default_e5, default_bm25 = DEPLOYMENT_DEFAULT
    remaining = report["remaining_retained_gold"]
    by_type = remaining["by_type"]
    selections = report["selected_by_fold"]
    default_folds = sum(
        values["e5_anchor"] == default_e5 and values["bm25_novel_max"] == default_bm25
        for values in selections.values()
    )
    return "\n".join([
        "# EXP-023 — E5 + BM25 quota ablation and handoff",
        "",
        "## Default candidate policy for the next pre-reranker stage",
        "",
        f"Use **E5 anchor @ {default_e5} + up to {default_bm25} novel BM25 documents**, with a fixed cap of {CAP} parent-document candidates/query.",
        "Append BM25 documents only when absent from the E5 anchor, in BM25 rank order. If fewer than 50 are novel, backfill first from the E5 tail and then BM25 until the list contains exactly 150 unique parent IDs. Preserve source ranks/provenance for LambdaMART.",
        "",
        f"This is the modal nested-OOF quota choice ({default_folds}/{len(selections)} folds); it is a fixed downstream policy, not a claim that a per-query or public optimum was selected. The fold-specific candidates in this EXP-023 cache remain an ablation artifact.",
        "",
        "## Nested OOF quota screen",
        "",
        "Selection for each held-out fold maximizes retained-gold Recall@150 on the other four folds, then MRR@150, then fixed grid order.",
        "",
        f"- OOF retained Recall@150: `{report['oof']['retained_recall@150']:.6f}`",
        f"- OOF retained MRR@150: `{report['oof']['retained_mrr@150']:.6f}`",
        f"- Actual novel BM25 additions: mean `{report['bm25_novel_actual']['mean']:.3f}`, min `{report['bm25_novel_actual']['min']}`, max `{report['bm25_novel_actual']['max']}`.",
        "",
        "| Held-out fold | E5 anchor | BM25 novel max | Retained Recall@150 | MRR@150 |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {fold} | {values['quota'][0]} | {values['quota'][1]} | {values['retained_recall@150']:.6f} | {values['retained_mrr@150']:.6f} |"
            for fold, values in sorted(report["fold_metrics"].items())
        ],
        "",
        "## Remaining retained-gold misses",
        "",
        f"There are `{remaining['total']}` retained gold occurrences absent from the fold-selected 150-candidate lists.",
        "",
        f"- **both_model_miss** (`{by_type.get('both_model_miss', 0)}`): the gold parent is absent from E5's stored Top-150 *and* from the selected BM25 ranking list. This is bounded by the rankings inspected here, so it is a handoff set for independent channels (Qwen dense or query-memory/lexical), **not** proof that the document is unretrievable or that either channel will rescue it.",
        f"- **budget_allocation_miss** (`{by_type.get('budget_allocation_miss', 0)}`): the gold parent occurs in at least one inspected E5/BM25 source ranking, but is omitted by this 150-document quota/allocation. It is a candidate-budget problem, not evidence of a semantic or lexical failure.",
        "",
        "The sample rows and per-occurrence ranks are in `REMAINING_MISSES.md` and `remaining_misses.jsonl`. Any claim that the first class is semantic versus lexical requires a fold-isolated rescue audit from the proposed Qwen and query-memory channels.",
        "",
        "## Scope",
        "",
        "These are candidate-stage OOF coverage metrics only. They do not measure LambdaMART or final reranker quality, and they must not be read as a public/submission result.",
        "",
    ])


def run_ablation(*, dense_candidates: Path, dense_manifest: Path, sparse_evidence_dir: Path,
                 sparse_report: Path, sparse_input_audit: Path, train_path: Path, folds_path: Path,
                 preprocessing_dir: Path, documents_path: Path, cache_dir: Path, results_dir: Path) -> dict[str, Any]:
    dense_meta = json.loads(dense_manifest.read_text(encoding="utf-8"))
    sparse_audit = json.loads(sparse_input_audit.read_text(encoding="utf-8"))
    if dense_meta["inputs"]["corpus_fingerprint"] != sparse_audit["structural_fingerprint"]:
        raise RuntimeError("Dense and sparse artifacts belong to different structural corpora")
    sparse_meta = json.loads(sparse_report.read_text(encoding="utf-8"))
    train = json.loads(train_path.read_text(encoding="utf-8"))
    answers = {str(qid): set(map(str, row["answer"])) for qid, row in train.items()}
    excluded = {str(row["doc_id"]) for row in json.loads((preprocessing_dir / "exclusions.json").read_text(encoding="utf-8"))}
    retained = {qid: gold - excluded for qid, gold in answers.items()}
    folds = {name: sorted(map(str, values)) for name, values in json.loads(folds_path.read_text(encoding="utf-8")).items()}
    fold_for = {qid: name for name, values in folds.items() for qid in values}
    docs = {str(row["doc_id"]): row for row in read_jsonl(documents_path)}
    dense_rows = {str(row["qid"]): row for row in read_jsonl(dense_candidates)}
    if set(dense_rows) != set(answers) or set(fold_for) != set(answers):
        raise RuntimeError("Train, folds, and dense OOF candidates must have identical query IDs")
    e5 = {qid: [str(item["doc_id"]) for item in row["candidates"]] for qid, row in dense_rows.items()}
    if not all(len(values) >= CAP and len(set(values[:CAP])) == CAP for values in e5.values()):
        raise RuntimeError("Dense OOF candidates must contain 150 unique parent IDs/query")

    bm25: dict[int, dict[str, list[str]]] = {budget: {} for _, budget in QUOTAS}
    for row in _iter_evidence(sparse_evidence_dir / "shards"):
        qid = str(row["qid"]); fold = fold_for[qid]
        for _, budget in QUOTAS:
            parameters = sparse_meta["selected_by_candidate_budget"][str(budget)][fold]
            first, rrf = _rankings(row["evidence"], int(parameters["depth"]), int(parameters["parent_rrf_k"]))
            bm25[budget][qid] = _cascade(first, rrf, int(parameters["fusion_rrf_k"]), int(parameters["head_cutoff"]))
    if not all(set(values) == set(answers) for values in bm25.values()):
        raise RuntimeError("Sparse OOF rankings do not cover every query")

    candidates: dict[tuple[int, int], dict[str, list[str]]] = {}
    for quota in QUOTAS:
        candidates[quota] = {}
        for qid in answers:
            try:
                candidates[quota][qid] = build_candidate(e5[qid], bm25[quota[1]][qid], quota)[0]
            except RuntimeError as error:
                raise RuntimeError(f"Quota {quota} cannot build qid={qid}: {error}") from error
    selected: dict[str, tuple[int, int]] = {}
    fold_metrics: dict[str, dict[str, float | list[int]]] = {}
    for heldout, heldout_qids in sorted(folds.items()):
        train_qids = [qid for fold, qids in folds.items() if fold != heldout for qid in qids]
        selected[heldout] = max(
            QUOTAS,
            key=lambda quota: (*_metric(candidates[quota], retained, train_qids), -QUOTAS.index(quota)),
        )
        recall, mrr = _metric(candidates[selected[heldout]], retained, heldout_qids)
        fold_metrics[heldout] = {"retained_recall@150": recall, "retained_mrr@150": mrr, "quota": list(selected[heldout])}

    cache_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []; misses: list[dict[str, Any]] = []; novel_counts: list[int] = []
    miss_counter = Counter(); pattern_counter: dict[str, Counter[str]] = defaultdict(Counter)
    with stage_run(results_dir, "e5-bm25-quota-ablation", total=len(answers)) as logger:
        for position, qid in enumerate(sorted(answers), 1):
            quota = selected[fold_for[qid]]; prediction, novel_count = build_candidate(e5[qid], bm25[quota[1]][qid], quota)
            novel_counts.append(novel_count)
            e5_rank, bm25_rank = _rank_map(e5[qid]), _rank_map(bm25[quota[1]][qid])
            output_rows.append({"qid": qid, "query": str(train[qid]["question"]), "fold": fold_for[qid], "selected_quota": {"e5_anchor": quota[0], "bm25_novel_max": quota[1]}, "candidates": [{"doc_id": doc_id, "rank": rank, "e5_rank": e5_rank.get(doc_id), "bm25_rank": bm25_rank.get(doc_id)} for rank, doc_id in enumerate(prediction, 1)]})
            pattern = "legal_identifier" if LEGAL_IDENTIFIER.search(str(train[qid]["question"])) else "no_legal_identifier"
            for gold_id in sorted(retained[qid] - set(prediction)):
                if gold_id not in e5_rank and gold_id not in bm25_rank:
                    miss_type = "both_model_miss"
                else:
                    miss_type = "budget_allocation_miss"
                miss_counter[miss_type] += 1; pattern_counter[pattern][miss_type] += 1
                misses.append({"qid": qid, "query": str(train[qid]["question"]), "gold_doc_id": gold_id, "document_label": str(docs.get(gold_id, {}).get("document_label", "missing")), "parse_mode": str(docs.get(gold_id, {}).get("parse_mode", "missing")), "legal_identifier": pattern == "legal_identifier", "miss_type": miss_type, "e5_rank": e5_rank.get(gold_id), "bm25_rank": bm25_rank.get(gold_id), "selected_quota": {"e5_anchor": quota[0], "bm25_novel_max": quota[1]}})
            if position % 256 == 0 or position == len(answers):
                logger.status(stage="e5-bm25-quota-ablation", state="RUNNING", completed=position, total=len(answers)); logger.log(f"progress={position}/{len(answers)}")
        output_path = cache_dir / "train_oof_candidates.jsonl"; misses_path = results_dir / "remaining_misses.jsonl"
        write_jsonl(output_path, output_rows); write_jsonl(misses_path, misses)
        overall_recall, overall_mrr = _metric({row["qid"]: [item["doc_id"] for item in row["candidates"]] for row in output_rows}, retained, sorted(answers))
        report_path = results_dir / "REPORT.md"
        report = {"schema_version": SCHEMA, "status": "PASS", "cap": CAP, "quota_grid": [list(value) for value in QUOTAS], "selection": "nested five-fold, maximize retained Recall@150; tie-break MRR@150 then fixed grid order", "recommended_default": {"e5_anchor": DEPLOYMENT_DEFAULT[0], "bm25_novel_max": DEPLOYMENT_DEFAULT[1], "policy": "E5 anchor, append novel BM25, then E5/BM25 deterministic backfill to exactly 150 unique parent IDs", "rationale": "modal nested-OOF quota selection; fixed policy for the next pre-reranker stage"}, "remaining_miss_definition": {"both_model_miss": "Gold parent absent from E5 stored Top-150 and the selected BM25 ranking list; source-list bounded handoff, not proof of unretrievability.", "budget_allocation_miss": "Gold parent present in at least one inspected source ranking but absent after the fixed 150-document quota/allocation."}, "selected_by_fold": {fold: {"e5_anchor": quota[0], "bm25_novel_max": quota[1]} for fold, quota in selected.items()}, "fold_metrics": fold_metrics, "oof": {"retained_recall@150": overall_recall, "retained_mrr@150": overall_mrr}, "bm25_novel_actual": {"mean": sum(novel_counts) / len(novel_counts), "min": min(novel_counts), "max": max(novel_counts)}, "remaining_retained_gold": {"total": len(misses), "by_type": dict(sorted(miss_counter.items())), "by_query_pattern": {name: dict(sorted(values.items())) for name, values in sorted(pattern_counter.items())}}, "inputs": {"dense_candidates_sha256": sha256_file(dense_candidates), "dense_manifest_sha256": sha256_file(dense_manifest), "sparse_evidence_manifest_sha256": sha256_file(sparse_evidence_dir / "manifest.json"), "sparse_report_sha256": sha256_file(sparse_report), "folds_sha256": sha256_file(folds_path), "code_sha256": _code_sha256()}, "artifacts": {"oof_candidates": output_path.name, "remaining_misses": misses_path.name, "report": report_path.name}}
        atomic_json(results_dir / "quota_report.json", report)
        misses.sort(key=lambda row: (row["miss_type"], row["qid"], row["gold_doc_id"]))
        (results_dir / "REMAINING_MISSES.md").write_text(_sample_markdown(misses), encoding="utf-8")
        report_path.write_text(_report_markdown(report), encoding="utf-8")
        manifest = {"schema_version": SCHEMA, "stage": "e5-bm25-quota-ablation", "content_fingerprint": _hash_json(report), "artifact_sha256": {"cache/train_oof_candidates.jsonl": sha256_file(output_path), "results/quota_report.json": sha256_file(results_dir / "quota_report.json"), "results/remaining_misses.jsonl": sha256_file(misses_path), "results/REMAINING_MISSES.md": sha256_file(results_dir / "REMAINING_MISSES.md"), "results/REPORT.md": sha256_file(report_path)}}
        atomic_json(cache_dir / "manifest.json", manifest); atomic_json(cache_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": manifest["stage"], "content_fingerprint": manifest["content_fingerprint"]})
        logger.set_telemetry({"queries": len(answers), "retained_recall@150": overall_recall})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"): __import__("sys").stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-candidates", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates" / "train_oof_candidates.jsonl")
    parser.add_argument("--dense-manifest", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates" / "manifest.json")
    parser.add_argument("--sparse-evidence-dir", type=Path, default=ROOT / "cache" / "exp021_sparse" / "depth_tune" / "raw4096_evidence")
    parser.add_argument("--sparse-report", type=Path, default=ROOT / "results" / "exp021_sparse" / "depth_rrf_tuning" / "tuning_report.json")
    parser.add_argument("--sparse-input-audit", type=Path, default=ROOT / "results" / "exp021_sparse" / "input_audit" / "sparse_input_audit.json")
    parser.add_argument("--train-file", type=Path, default=ROOT / "public_test_dataset" / "train.json")
    parser.add_argument("--folds-file", type=Path, default=ROOT / "cache" / "cv_folds.json")
    parser.add_argument("--preprocessing-dir", type=Path, default=ROOT / "cache" / "final_preprocessed_v2")
    parser.add_argument("--documents-path", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1" / "documents.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "exp023_e5_bm25_quota")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "exp023_e5_bm25_quota")
    args = parser.parse_args(argv)
    print(json.dumps(run_ablation(dense_candidates=args.dense_candidates, dense_manifest=args.dense_manifest, sparse_evidence_dir=args.sparse_evidence_dir, sparse_report=args.sparse_report, sparse_input_audit=args.sparse_input_audit, train_path=args.train_file, folds_path=args.folds_file, preprocessing_dir=args.preprocessing_dir, documents_path=args.documents_path, cache_dir=args.cache_dir, results_dir=args.results_dir), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
