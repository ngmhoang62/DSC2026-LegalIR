"""Leakage-safe tuning over cached rankings/logits; never runs neural models."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp012b_core import atomic_json, load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings, weighted_rrf
from exp012b_reranker import aggregate_ce_documents, fuse_stage1_and_ce


def load_folds(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("cv_folds.json must be a non-empty object")
    seen: set[str] = set()
    result: dict[str, list[str]] = {}
    for fold, qids in payload.items():
        values = [str(qid) for qid in qids]
        overlap = seen & set(values)
        if overlap:
            raise ValueError(f"Queries assigned to multiple folds: {sorted(overlap)[:5]}")
        seen.update(values)
        result[str(fold)] = values
    return result


def _rank_predictions(
    source_by_qid: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    qids: Sequence[str],
    weights: Mapping[str, float],
    limit: int,
) -> dict[str, list[str]]:
    return {
        qid: [
            row["doc_id"]
            for row in weighted_rrf(source_by_qid[qid], weights=weights, limit=limit)
        ]
        for qid in qids
    }


def nested_tune_stage1(
    source_by_qid: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
    *,
    weight_grid: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    grid = list(weight_grid or [
        {"bm25": bm25, "bge_block": block, "bge_leaf": leaf}
        for bm25, block, leaf in itertools.product((0.8, 1.0, 1.2, 1.5), (0.5, 1.0), (1.0, 1.5, 2.0))
    ])
    all_qids = set(answers)
    oof: dict[str, list[str]] = {}
    chosen: dict[str, dict[str, float]] = {}
    fold_metrics: dict[str, dict[str, float]] = {}
    for fold, heldout in folds.items():
        heldout_set = set(heldout)
        tuning_qids = sorted(all_qids - heldout_set)
        best_weights: Mapping[str, float] | None = None
        best_key: tuple[float, float, str] | None = None
        for weights in grid:
            predictions = _rank_predictions(source_by_qid, tuning_qids, weights, 50)
            metrics = evaluate_rankings(predictions, {qid: answers[qid] for qid in tuning_qids})
            key = (metrics["recall@50"], metrics["recall@5"], json.dumps(weights, sort_keys=True))
            if best_key is None or key > best_key:
                best_key = key
                best_weights = weights
        assert best_weights is not None
        heldout_predictions = _rank_predictions(source_by_qid, list(heldout), best_weights, 50)
        oof.update(heldout_predictions)
        chosen[fold] = dict(best_weights)
        fold_metrics[fold] = evaluate_rankings(
            heldout_predictions, {qid: answers[qid] for qid in heldout}
        )
    overall = evaluate_rankings(oof, answers)
    return {"chosen_weights": chosen, "fold_metrics": fold_metrics, "metrics": overall, "predictions": oof}


def nested_tune_zero_shot(
    stage1_by_qid: Mapping[str, Sequence[dict[str, Any]]],
    score_rows: Sequence[dict[str, Any]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    gamma_rankings = {
        gamma: aggregate_ce_documents(score_rows, gamma=gamma)
        for gamma in (0.0, 0.1, 0.2, 0.3)
    }
    all_qids = set(answers)
    oof: dict[str, list[str]] = {}
    chosen: dict[str, dict[str, float]] = {}
    fold_metrics: dict[str, dict[str, float]] = {}
    for fold, heldout in folds.items():
        tuning_qids = sorted(all_qids - set(heldout))
        best: tuple[float, float, float, float] | None = None
        best_config: tuple[float, float] | None = None
        for gamma, ce_weight in itertools.product((0.0, 0.1, 0.2, 0.3), (1.0, 1.5, 2.0, 3.0)):
            predictions = {
                qid: [row["doc_id"] for row in fuse_stage1_and_ce(
                    stage1_by_qid[qid], gamma_rankings[gamma][qid], ce_weight=ce_weight, limit=5
                )]
                for qid in tuning_qids
            }
            metrics = evaluate_rankings(predictions, {qid: answers[qid] for qid in tuning_qids}, ks=(5,))
            key = (metrics["recall@5"], metrics["precision@5"], -gamma, -ce_weight)
            if best is None or key > best:
                best = key
                best_config = (gamma, ce_weight)
        assert best_config is not None
        gamma, ce_weight = best_config
        predictions = {
            qid: [row["doc_id"] for row in fuse_stage1_and_ce(
                stage1_by_qid[qid], gamma_rankings[gamma][qid], ce_weight=ce_weight, limit=5
            )]
            for qid in heldout
        }
        oof.update(predictions)
        chosen[fold] = {"gamma": gamma, "ce_weight": ce_weight}
        fold_metrics[fold] = evaluate_rankings(predictions, {qid: answers[qid] for qid in heldout}, ks=(5,))
    return {
        "chosen_config": chosen,
        "fold_metrics": fold_metrics,
        "metrics": evaluate_rankings(oof, answers, ks=(5,)),
        "predictions": oof,
    }


def evaluate_zero_shot_artifacts(
    candidates_path: Path,
    scores_path: Path,
    train_path: Path,
    folds_path: Path,
    output_path: Path,
    stage1_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    stage1 = {
        record["qid"]: [
            {"doc_id": candidate["doc_id"], "rank": candidate["rank"]}
            for candidate in record["candidates"]
        ]
        for record in read_jsonl(candidates_path)
    }
    scores = list(read_jsonl(scores_path))
    answers = load_answers(train_path)
    folds = load_folds(folds_path)
    result = nested_tune_zero_shot(stage1, scores, answers, folds)
    serializable = {key: value for key, value in result.items() if key != "predictions"}
    baseline = dict(stage1_metrics or {})
    recall = float(result["metrics"]["recall@5"])
    precision = float(result["metrics"]["precision@5"])
    stage1_recall = float(baseline.get("recall@5", float("-inf")))
    stage1_precision = float(baseline.get("precision@5", float("-inf")))
    checks = {
        "recall_floor_pass": recall >= 0.8584,
        "precision_floor_pass": precision >= 0.1821,
        "stage1_recall_non_regression_pass": recall >= stage1_recall,
        "stage1_precision_non_regression_pass": precision >= stage1_precision,
    }
    serializable["stage1_metrics"] = baseline
    serializable["stage1_delta"] = {
        "recall@5": recall - stage1_recall,
        "precision@5": precision - stage1_precision,
    }
    serializable["promotion_gate"] = {**checks, "overall_pass": all(checks.values())}
    atomic_json(output_path, serializable)
    return result
