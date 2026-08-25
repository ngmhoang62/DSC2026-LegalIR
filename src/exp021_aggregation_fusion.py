"""Cheap, post-retrieval comparison of BM25 document-aggregation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exp012b_core import atomic_json, read_jsonl
from exp012b_retrieval import evaluate_rankings, recall_at_k, weighted_rrf


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row["qid"]): list(row["documents"]) for row in read_jsonl(path)}


def audit_aggregation_fusion(*, rrf_path: Path, first_path: Path, train_path: Path, output_path: Path) -> dict[str, Any]:
    rrf = _load(rrf_path)
    first = _load(first_path)
    train = json.loads(train_path.read_text(encoding="utf-8"))
    answers = {str(qid): set(map(str, row["answer"])) for qid, row in train.items()}
    if set(rrf) != set(first) or set(rrf) != set(answers):
        raise RuntimeError("aggregation rankings/train query set mismatch")
    fused: dict[str, list[str]] = {}
    head_tail: dict[str, list[str]] = {}
    union_metrics: dict[str, float] = {}
    for qid in sorted(answers):
        rows = weighted_rrf({"rrf3": rrf[qid], "first_passage": first[qid]}, limit=100)
        fused[qid] = [str(row["doc_id"]) for row in rows]
        head = fused[qid][:20]
        head_tail[qid] = head + [
            str(row["doc_id"]) for row in rrf[qid] if str(row["doc_id"]) not in set(head)
        ][:80]
    for source_depth in (25, 50):
        union = {
            qid: [str(row["doc_id"]) for row in first[qid][:source_depth]]
            + [str(row["doc_id"]) for row in rrf[qid][:source_depth] if str(row["doc_id"]) not in {str(item["doc_id"]) for item in first[qid][:source_depth]}]
            for qid in answers
        }
        union_metrics[f"union_first{source_depth}_rrf3{source_depth}_max{2 * source_depth}"] = recall_at_k(
            union, answers, 2 * source_depth
        )
    report = {
        "fused_equal_rrf": evaluate_rankings(fused, answers, ks=(1, 5, 10, 20, 50, 100)),
        "fused_head20_then_rrf3_tail": evaluate_rankings(head_tail, answers, ks=(1, 5, 10, 20, 50, 100)),
        "candidate_union_recall": union_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, report)
    return report
