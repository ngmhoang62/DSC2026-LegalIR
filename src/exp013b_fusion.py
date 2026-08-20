"""Selective routing, RRF fusion, and EXP-013b evaluation gates."""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from exp012b_core import atomic_json, load_answers, read_jsonl, stage_run, write_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import bootstrap_recall_gain, rank_ids, write_manifest


def _rank(values: Mapping[str, float]) -> dict[str, int]:
    return {doc: index for index, doc in enumerate(rank_ids(values), 1)}


def _zscore(arr: list[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    s = float(a.std())
    return (a - a.mean()) / (s if s > 1e-5 else 1.0)


def _fuse(lambda_rows: list[Mapping[str, Any]], qwen_rows: list[Mapping[str, Any]], candidates: Mapping[str, Any],
          weights: Mapping[str, float] = {"qwen": 0.8, "lambda": 0.2, "bge": 0.0}) -> list[str]:
    doc_ids = [str(row["doc_id"]) for row in lambda_rows]
    lambda_scores = [float(row["lambda_score"]) for row in lambda_rows]
    qwen_map = {str(row["doc_id"]): float(row["qwen_score"]) for row in qwen_rows}
    
    lam_z = _zscore(lambda_scores)
    min_qwen = min(qwen_map.values()) if qwen_map else -20.0
    qwen_scores = [qwen_map.get(doc_id, min_qwen - 5.0) for doc_id in doc_ids]
    qwen_z = _zscore(qwen_scores)
    
    w_qwen = float(weights.get("qwen", 0.8))
    w_lam = float(weights.get("lambda", 0.2))
    w_bge = float(weights.get("bge", 0.0))
    
    result: dict[str, float] = {}
    for idx, doc_id in enumerate(doc_ids):
        bge_rank = int(candidates.get(doc_id, {}).get("bge_rank", 999))
        score = w_lam * lam_z[idx] + w_qwen * qwen_z[idx]
        if w_bge > 0 and bge_rank < 999:
            score += w_bge / (60.0 + bge_rank)
        result[doc_id] = float(score)
    return rank_ids(result)


def _ambiguity(lambda_rows: list[Mapping[str, Any]], candidates: Mapping[str, Any]) -> float:
    values = [float(row["lambda_score"]) for row in lambda_rows]
    margin = values[4] - values[5] if len(values) > 5 else values[-1]
    top_lambda = {str(row["doc_id"]) for row in lambda_rows[:5]}
    top_bge = {doc_id for doc_id, row in candidates.items() if int(row.get("bge_rank", 999)) <= 5}
    disagreement = 1.0 - len(top_lambda & top_bge) / 5.0
    agreement = np.mean([float(row.get("channel_count", 1)) for row in candidates.values()])
    exps = np.exp(np.asarray(values[:10]) - max(values[:10])); entropy = -float(np.sum((exps / exps.sum()) * np.log(exps / exps.sum() + 1e-12)))
    return float(-margin + disagreement + 0.15 * entropy - 0.05 * agreement)


def screen_fold(*, fold: int, candidates_path: Path, preranker_path: Path, qwen_path: Path, train_path: Path,
                folds_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    folds, answers = load_folds(folds_path), load_answers(train_path)
    qids = {str(value) for value in folds[f"fold_{fold}"]}
    candidates = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(candidates_path)}
    lam = {str(row["qid"]): row["candidates"] for row in read_jsonl(preranker_path)}
    qwen = {str(row["qid"]): row["scores"] for row in read_jsonl(qwen_path)}
    if set(qwen) != qids: raise ValueError("Qwen screen must contain exactly held-out fold qids")
    gold = {qid: answers[qid] for qid in qids}
    lambda_predictions = {qid: [str(row["doc_id"]) for row in lam[qid]] for qid in qids}
    lambda_metrics = evaluate_rankings(lambda_predictions, gold, ks=(5,))
    
    grid = [dict(zip(("qwen", "lambda", "bge"), values)) for values in product((0.5, 0.6, 0.7, 0.8, 0.9), (0.1, 0.2, 0.3, 0.4), (0.0, 0.05))]
    best_config = {"qwen": 0.8, "lambda": 0.2, "bge": 0.0}
    best_qwen_metrics = evaluate_rankings({qid: _fuse(lam[qid], qwen[qid], candidates[qid], best_config) for qid in qids}, gold, ks=(5,))
    
    for config in grid:
        fused = {qid: _fuse(lam[qid], qwen[qid], candidates[qid], config) for qid in qids}
        m = evaluate_rankings(fused, gold, ks=(5,))
        if (m["recall@5"], m["precision@5"]) > (best_qwen_metrics["recall@5"], best_qwen_metrics["precision@5"]):
            best_qwen_metrics = m
            best_config = config
            
    recall_gain = best_qwen_metrics["recall@5"] - lambda_metrics["recall@5"]
    precision_delta = best_qwen_metrics["precision@5"] - lambda_metrics["precision@5"]
    passed = recall_gain >= 0.005 and precision_delta >= -0.001
    output_dir.mkdir(parents=True, exist_ok=True); path = output_dir / f"fold_{fold}_screen.json"
    with stage_run(output_dir, "score-qwen-screen", total=len(qids), v3_fingerprint=v3_fingerprint) as logger:
        atomic_json(path, {"status": "PASS" if passed else "FAIL", "fold": fold, "lambda": lambda_metrics, "qwen": best_qwen_metrics,
                           "best_config": best_config, "recall_gain": recall_gain, "precision_delta": precision_delta})
        if not passed: raise RuntimeError("Qwen fold-0 promotion gate failed")
        logger.log(f"fold0_passed recall_gain={recall_gain:.5f} precision_delta={precision_delta:.5f} config={best_config}")
        return write_manifest(output_dir, stage="score-qwen-screen", v3_fingerprint=v3_fingerprint, config={"fold": fold, "best_config": best_config}, files=[path], counts={"queries": len(qids)})


def fit_router(*, candidates_path: Path, preranker_path: Path, qwen_path: Path, train_path: Path, folds_path: Path,
               output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    folds, answers = load_folds(folds_path), load_answers(train_path); qids = {str(value) for value in folds["fold_0"]}
    candidates = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(candidates_path)}
    lam = {str(row["qid"]): row["candidates"] for row in read_jsonl(preranker_path)}; qwen = {str(row["qid"]): row["scores"] for row in read_jsonl(qwen_path)}
    gold = {qid: answers[qid] for qid in qids}
    plain = {qid: [str(row["doc_id"]) for row in lam[qid]] for qid in qids}
    
    grid = [dict(zip(("qwen", "lambda", "bge"), values)) for values in product((0.5, 0.6, 0.7, 0.8, 0.9), (0.1, 0.2, 0.3, 0.4), (0.0, 0.05))]
    best_config = {"qwen": 0.8, "lambda": 0.2, "bge": 0.0}
    best_gain = -1.0
    for config in grid:
        fused = {qid: _fuse(lam[qid], qwen[qid], candidates[qid], config) for qid in qids}
        gain = evaluate_rankings(fused, gold, ks=(5,))["recall@5"] - evaluate_rankings(plain, gold, ks=(5,))["recall@5"]
        if gain > best_gain:
            best_gain = gain
            best_config = config
            
    full = {qid: _fuse(lam[qid], qwen[qid], candidates[qid], best_config) for qid in qids}
    full_gain = evaluate_rankings(full, gold, ks=(5,))["recall@5"] - evaluate_rankings(plain, gold, ks=(5,))["recall@5"]
    ordered = sorted(qids, key=lambda qid: (-_ambiguity(lam[qid], candidates[qid]), qid)); selected: dict[str, Any] | None = None
    for fraction in (.30, .50, .75, 1.0):
        routed = set(ordered[:math.ceil(len(ordered) * fraction)])
        mixed = {qid: full[qid] if qid in routed else plain[qid] for qid in qids}
        gain = evaluate_rankings(mixed, gold, ks=(5,))["recall@5"] - evaluate_rankings(plain, gold, ks=(5,))["recall@5"]
        if gain >= .8 * full_gain:
            selected = {"fraction": fraction, "threshold": _ambiguity(lam[ordered[len(routed)-1]], candidates[ordered[len(routed)-1]]), "fold0_gain": gain, "full_gain": full_gain}; break
    output_dir.mkdir(parents=True, exist_ok=True); path = output_dir / "router.json"
    with stage_run(output_dir, "fit-router", total=len(qids), v3_fingerprint=v3_fingerprint) as logger:
        atomic_json(path, {"status": "PASS" if selected else "FAIL", "router": selected, "weights": best_config})
        if not selected: raise RuntimeError("router cannot retain 80% full-Qwen gain")
        logger.log(f"router_fitted fraction={selected['fraction']} threshold={selected['threshold']:.4f} fold0_gain={selected['fold0_gain']:.5f}")
        return write_manifest(output_dir, stage="fit-router", v3_fingerprint=v3_fingerprint, config=selected, files=[path], counts={"queries": len(qids)})


def evaluate_oof(*, candidates_path: Path, preranker_path: Path, qwen_paths: Mapping[int, Path], router_path: Path,
                 train_path: Path, folds_path: Path, baseline_results: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    answers, folds = load_answers(train_path), load_folds(folds_path)
    candidates = {str(row["qid"]): {str(item["doc_id"]): item for item in row["candidates"]} for row in read_jsonl(candidates_path)}
    lam = {str(row["qid"]): row["candidates"] for row in read_jsonl(preranker_path)}
    router = json.loads(router_path.read_text(encoding="utf-8"))["router"]; fraction = float(router["fraction"])
    predictions: dict[str, list[str]] = {}; fold_metrics: dict[str, Any] = {}
    with stage_run(output_dir, "evaluate-oof", total=len(answers), v3_fingerprint=v3_fingerprint) as logger:
        qwen_by_fold: dict[int, dict[str, list[dict[str, Any]]]] = {}
        routed_by_fold: dict[int, set[str]] = {}
        for fold in range(5):
            qids = {str(value) for value in folds[f"fold_{fold}"]}
            for qid in qids:
                if any(str(row.get("fold")) != f"fold_{fold}" for row in lam[qid]):
                    raise RuntimeError(f"OOF LambdaMART fold mismatch at qid={qid}")
            qwen = {str(row["qid"]): row["scores"] for row in read_jsonl(qwen_paths[fold])}
            routed = set(qwen)
            qwen_by_fold[fold], routed_by_fold[fold] = qwen, routed
        grid = [dict(zip(("qwen", "lambda", "bge"), values)) for values in product((0.5, 0.6, 0.7, 0.8, 0.9), (0.1, 0.2, 0.3, 0.4), (0.0, 0.05))]
        chosen: dict[str, dict[str, float]] = {}
        for target_fold in range(5):
            training_folds = [fold for fold in range(5) if fold != target_fold]
            best_key = None; best_config = None
            for config in grid:
                trial: dict[str, list[str]] = {}
                trial_gold: dict[str, set[str]] = {}
                for fold in training_folds:
                    for qid in map(str, folds[f"fold_{fold}"]):
                        trial[qid] = _fuse(lam[qid], qwen_by_fold[fold][qid], candidates[qid], config) if qid in routed_by_fold[fold] else [str(row["doc_id"]) for row in lam[qid]]
                        trial_gold[qid] = answers[qid]
                metric = evaluate_rankings(trial, trial_gold, ks=(5,))
                key = (metric["recall@5"], metric["precision@5"], config["qwen"], config["lambda"], config["bge"])
                if best_key is None or key > best_key: best_key, best_config = key, config
            assert best_config is not None
            chosen[f"fold_{target_fold}"] = best_config
            qids = {str(value) for value in folds[f"fold_{target_fold}"]}
            for qid in qids:
                predictions[qid] = _fuse(lam[qid], qwen_by_fold[target_fold][qid], candidates[qid], best_config) if qid in routed_by_fold[target_fold] else [str(row["doc_id"]) for row in lam[qid]]
            fold_metrics[f"fold_{target_fold}"] = evaluate_rankings({qid: predictions[qid] for qid in qids}, {qid: answers[qid] for qid in qids}, ks=(5,))
        config_counts = Counter(tuple(sorted(config.items())) for config in chosen.values())
        modal_tuple = sorted(config_counts, key=lambda item: (-config_counts[item], item))[0]
        modal_config = dict(modal_tuple)
        metrics = evaluate_rankings(predictions, answers, ks=(5,)); baseline_predictions = {}
        candidate_predictions = {qid: list(candidates[qid]) for qid in candidates}
        candidate_recall = evaluate_rankings(candidate_predictions, answers, ks=(180,))["recall@180"]
        for fold in range(5): baseline_predictions.update(json.loads((baseline_results / f"fold_{fold}_metrics.json").read_text(encoding="utf-8"))["predictions"])
        ci = bootstrap_recall_gain(predictions, baseline_predictions, answers)
        baseline = json.loads((baseline_results.parent / "oof_metrics.json").read_text(encoding="utf-8"))
        worst_pass = all(fold_metrics[name]["recall@5"] >= baseline["fold_metrics"][name]["recall@5"] for name in fold_metrics)
        passed = (metrics["recall@5"] > baseline["metrics"]["recall@5"] and
                  metrics["precision@5"] >= baseline["metrics"]["precision@5"] and
                  candidate_recall >= .990 and worst_pass)
        report = {"status": "PASS" if passed else "FAIL", "metrics": metrics, "fold_metrics": fold_metrics,
                  "candidate_recall": candidate_recall, "bootstrap_gain": ci, "worst_fold_pass": worst_pass, "fusion_by_fold": chosen,
                  "modal_public_fusion": modal_config, "router_fraction": fraction, "predictions": predictions}
        path = output_dir / "oof_report.json"; atomic_json(path, report)
        if not passed: raise RuntimeError(f"EXP-013b OOF promotion gate failed: recall={metrics['recall@5']:.5f} baseline={baseline['metrics']['recall@5']:.5f}")
        return write_manifest(output_dir, stage="evaluate-oof", v3_fingerprint=v3_fingerprint,
            config={"router_fraction": fraction, "nested_fusion": True, "modal_public_fusion": modal_config},
            files=[path], counts={"queries": len(predictions), "routed": sum(len(values) for values in routed_by_fold.values())})
