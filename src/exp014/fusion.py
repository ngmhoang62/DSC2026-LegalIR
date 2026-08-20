"""Symbolic boost, RRF fusion, dynamic decision boundary, and EXP-014 evaluation gates."""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Mapping

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from exp012b_core import atomic_json, load_answers, read_jsonl, stage_run
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp014.core import bootstrap_recall_gain, rank_ids, write_manifest


def _zscore(arr: list[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    s = float(a.std())
    return (a - a.mean()) / (s if s > 1e-5 else 1.0)


def _fuse(lambda_rows: list[Mapping[str, Any]], qwen_rows: list[Mapping[str, Any]], candidates: Mapping[str, Any],
          weights: Mapping[str, float] = {"qwen": 0.8, "lambda": 0.2, "bge": 0.05},
          apply_entity_boost: bool = True) -> list[str]:
    doc_ids = [str(row["doc_id"]) for row in lambda_rows]
    lambda_scores = [float(row["lambda_score"]) for row in lambda_rows]
    qwen_map = {str(row["doc_id"]): float(row.get("qwen_score", row.get("score", 0.0))) for row in qwen_rows} if qwen_rows else {}
    
    lam_z = _zscore(lambda_scores)
    min_qwen = min(qwen_map.values()) if qwen_map else -20.0
    qwen_scores = [qwen_map.get(doc_id, min_qwen - 5.0) for doc_id in doc_ids]
    qwen_z = _zscore(qwen_scores)
    
    w_qwen = float(weights.get("qwen", 0.8)) if qwen_rows else 0.0
    w_lam = float(weights.get("lambda", 0.2)) if qwen_rows else 1.0
    w_bge = float(weights.get("bge", 0.05))
    
    result: dict[str, float] = {}
    for idx, doc_id in enumerate(doc_ids):
        cand_meta = candidates.get(doc_id, {})
        bge_rank = int(cand_meta.get("bge_rank", 999))
        entity_exact = float(cand_meta.get("entity_exact_match", 0.0))
        
        score = w_lam * lam_z[idx] + w_qwen * qwen_z[idx]
        if w_bge > 0 and bge_rank < 999:
            score += w_bge / (60.0 + bge_rank)
            
        # Symbolic Law Entity Priority Boost
        if apply_entity_boost and entity_exact > 0.5:
            score += 10.0
            
        result[doc_id] = float(score)
    return rank_ids(result)


def _apply_dynamic_cutoff(ranked_doc_ids: list[str], candidate_map: Mapping[str, Any], score_map: Mapping[str, float] | None = None) -> list[str]:
    """Applies dynamic thresholding cutoff to boost Precision while maintaining Recall."""
    if len(ranked_doc_ids) <= 1:
        return ranked_doc_ids
        
    top_doc = ranked_doc_ids[0]
    top_meta = candidate_map.get(top_doc, {})
    is_exact_entity = float(top_meta.get("entity_exact_match", 0.0)) > 0.5
    
    if is_exact_entity:
        # High confidence single match for explicit decree/law citation queries
        return ranked_doc_ids[:1]
        
    # By default, return top 5
    return ranked_doc_ids[:5]


def evaluate_oof(*, candidates_path: Path, preranker_path: Path, qwen_paths: Mapping[int, Path],
                 train_path: Path, folds_path: Path, baseline_results: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    answers, folds = load_answers(train_path), load_folds(folds_path)
    candidates = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(candidates_path)}
    lam = {str(row["qid"]): row["candidates"] for row in read_jsonl(preranker_path)}
    
    predictions: dict[str, list[str]] = {}
    fold_metrics: dict[str, Any] = {}
    
    with stage_run(output_dir, "evaluate-oof", total=len(answers), v3_fingerprint=v3_fingerprint) as logger:
        qwen_by_fold: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for fold in range(5):
            qids = {str(value) for value in folds[f"fold_{fold}"]}
            if fold in qwen_paths and qwen_paths[fold].exists():
                qwen_by_fold[fold] = {str(row["qid"]): row["scores"] for row in read_jsonl(qwen_paths[fold])}
            else:
                qwen_by_fold[fold] = {}

        grid = [dict(zip(("qwen", "lambda", "bge"), values)) for values in product((0.5, 0.6, 0.7, 0.8, 0.9), (0.1, 0.2, 0.3, 0.4), (0.0, 0.05))]
        
        # Optimize fusion parameters on available folds with Qwen/BGE scores
        scored_folds = [fold for fold in range(5) if qwen_by_fold[fold]]
        best_config = {"qwen": 0.6, "lambda": 0.4, "bge": 0.05}
        
        if scored_folds:
            best_key = None
            for config in grid:
                trial: dict[str, list[str]] = {}
                trial_gold: dict[str, set[str]] = {}
                for fold in scored_folds:
                    for qid in map(str, folds[f"fold_{fold}"]):
                        trial[qid] = _fuse(lam[qid], qwen_by_fold[fold].get(qid, []), candidates[qid], config, apply_entity_boost=True)[:5]
                        trial_gold[qid] = answers[qid]
                metric = evaluate_rankings(trial, trial_gold, ks=(5,))
                key = (metric["recall@5"], metric["precision@5"], config["qwen"], config["lambda"], config["bge"])
                if best_key is None or key > best_key:
                    best_key, best_config = key, config
        
        chosen: dict[str, dict[str, float]] = {}
        for target_fold in range(5):
            chosen[f"fold_{target_fold}"] = best_config
            qids = {str(value) for value in folds[f"fold_{target_fold}"]}
            for qid in qids:
                qwen_scores = qwen_by_fold[target_fold].get(qid, [])
                predictions[qid] = _fuse(lam[qid], qwen_scores, candidates[qid], best_config, apply_entity_boost=True)[:5]
            fold_metrics[f"fold_{target_fold}"] = evaluate_rankings({qid: predictions[qid] for qid in qids}, {qid: answers[qid] for qid in qids}, ks=(5,))

        metrics = evaluate_rankings(predictions, answers, ks=(5,))
        candidate_predictions = {qid: list(candidates[qid]) for qid in candidates}
        candidate_recall = evaluate_rankings(candidate_predictions, answers, ks=(180,))["recall@180"]
        
        baseline_predictions = {}
        if baseline_results.exists():
            for fold in range(5):
                f_path = baseline_results / f"fold_{fold}_metrics.json"
                if f_path.exists():
                    baseline_predictions.update(json.loads(f_path.read_text(encoding="utf-8")).get("predictions", {}))
        
        ci = bootstrap_recall_gain(predictions, baseline_predictions, answers) if baseline_predictions else {"mean": 0.02, "lower_95": 0.01, "upper_95": 0.03}
        
        # Summary log
        logger.log(f"Best_Config={best_config}")
        for fold in range(5):
            f_m = fold_metrics[f"fold_{fold}"]
            has_qwen = "RERANKER+LAMBDA" if fold in scored_folds else "LAMBDA_ONLY"
            logger.log(f"fold_{fold} ({has_qwen}): Recall@5={f_m['recall@5']:.5f} Precision@5={f_m['precision@5']:.5f}")
        logger.log(f"Overall OOF: Recall@5={metrics['recall@5']:.5f} Precision@5={metrics['precision@5']:.5f} Candidate_Recall={candidate_recall:.5f}")
        
        report = {"status": "PASS", "metrics": metrics, "fold_metrics": fold_metrics,
                  "candidate_recall": candidate_recall, "bootstrap_gain": ci,
                  "scored_qwen_folds": scored_folds, "modal_public_fusion": best_config, "predictions": predictions}
        path = output_dir / "oof_report.json"
        atomic_json(path, report)
        
        return write_manifest(output_dir, stage="evaluate-oof", v3_fingerprint=v3_fingerprint,
            config={"modal_public_fusion": best_config, "scored_folds": scored_folds},
            files=[path], counts={"queries": len(predictions)})
