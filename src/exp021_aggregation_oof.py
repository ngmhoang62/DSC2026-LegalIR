"""Nested-fold selection for post-retrieval BM25 aggregation choices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exp012b_core import atomic_json, read_jsonl
from exp012b_retrieval import evaluate_rankings, recall_at_k, weighted_rrf


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row["qid"]): list(row["documents"]) for row in read_jsonl(path)}


def nested_aggregation_oof(*, rrf_path: Path, first_path: Path, train_path: Path, folds_path: Path, output_path: Path) -> dict[str, Any]:
    rrf_rows, first_rows = _load(rrf_path), _load(first_path)
    train = json.loads(train_path.read_text(encoding="utf-8"))
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    answers = {str(qid): set(map(str, row["answer"])) for qid, row in train.items()}
    if set(rrf_rows) != set(first_rows) or set(rrf_rows) != set(answers):
        raise RuntimeError("aggregation ranking/train query set mismatch")
    configs: dict[str, dict[str, list[str]]] = {
        "rrf3": {qid: [str(row["doc_id"]) for row in rows] for qid, rows in rrf_rows.items()},
        "first_passage": {qid: [str(row["doc_id"]) for row in rows] for qid, rows in first_rows.items()},
    }
    fused: dict[str, list[str]] = {}
    for qid in answers:
        fused[qid] = [str(row["doc_id"]) for row in weighted_rrf(
            {"rrf3": rrf_rows[qid], "first_passage": first_rows[qid]}, limit=100
        )]
    configs["equal_fusion"] = fused
    for cutoff in (10, 20, 30):
        configs[f"fusion_head{cutoff}_rrf3_tail"] = {
            qid: fused[qid][:cutoff] + [
                str(row["doc_id"]) for row in rrf_rows[qid]
                if str(row["doc_id"]) not in set(fused[qid][:cutoff])
            ][:100 - cutoff]
            for qid in answers
        }
    fold_qids = {name: set(map(str, values)) for name, values in folds.items()}
    if set().union(*fold_qids.values()) != set(answers) or sum(map(len, fold_qids.values())) != len(answers):
        raise RuntimeError("folds must partition every training query")
    oof: dict[str, list[str]] = {}
    selected: dict[str, Any] = {}
    fold_metrics: dict[str, Any] = {}
    for heldout_name, heldout in sorted(fold_qids.items()):
        tuning = set(answers) - heldout
        ranked = sorted(
            (
                recall_at_k({qid: configs[name][qid] for qid in tuning}, {qid: answers[qid] for qid in tuning}, 5),
                recall_at_k({qid: configs[name][qid] for qid in tuning}, {qid: answers[qid] for qid in tuning}, 10),
                name,
            )
            for name in configs
        )
        best_r5, best_r10, best_name = ranked[-1]
        held_predictions = {qid: configs[best_name][qid] for qid in heldout}
        oof.update(held_predictions)
        selected[heldout_name] = {"config": best_name, "tuning_recall@5": best_r5, "tuning_recall@10": best_r10}
        fold_metrics[heldout_name] = evaluate_rankings(held_predictions, {qid: answers[qid] for qid in heldout}, ks=(1, 5, 10, 20, 50, 100))
    report = {
        "status": "PASS", "selection_metric": "recall@5_then_recall@10",
        "candidate_configs": sorted(configs), "selected_by_fold": selected,
        "fold_metrics": fold_metrics,
        "oof_metrics": evaluate_rankings(oof, answers, ks=(1, 5, 10, 20, 50, 100)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_path, report)
    return report
