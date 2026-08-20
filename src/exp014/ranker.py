"""Fold-safe LambdaMART shortlist models for EXP-014."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from exp012b_core import atomic_json, load_answers, read_jsonl, stage_run, write_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp014.core import dump_pickle, rank_ids, write_manifest
from exp014.features import FEATURE_COLUMNS, extract_features, feature_matrix


def train_preranker_oof(*, candidates_path: Path, train_path: Path, folds_path: Path, output_dir: Path,
                        v3_fingerprint: str) -> dict[str, Any]:
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:
        raise RuntimeError("lightgbm is required; run pip install lightgbm") from error

    candidates, answers, folds = list(read_jsonl(candidates_path)), load_answers(train_path), load_folds(folds_path)
    by_qid, type_codes = extract_features(candidates)
    if set(by_qid) != set(answers):
        raise ValueError("candidate/train query sets differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[dict[str, Any]]] = {}
    model_paths: list[Path] = []
    with stage_run(output_dir, "train-preranker-oof", total=len(by_qid), v3_fingerprint=v3_fingerprint) as logger:
        for fold_name, heldout_values in sorted(folds.items()):
            heldout = [str(value) for value in heldout_values]
            training = sorted(set(by_qid) - set(heldout))
            if set(training) & set(heldout):
                raise RuntimeError("fold leakage")
            
            train_rows = [row for qid in training for row in by_qid[qid]]
            y_train = np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in train_rows], dtype=np.int32)
            model = LGBMRanker(objective="lambdarank", metric="ndcg", learning_rate=0.03,
                               n_estimators=500, num_leaves=63, min_child_samples=20, random_state=42,
                               n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(feature_matrix(train_rows), y_train, group=[len(by_qid[qid]) for qid in training], eval_at=[5])
            
            model_path = output_dir / f"{fold_name}.pkl"
            dump_pickle(model_path, {"model": model, "feature_columns": FEATURE_COLUMNS, "type_codes": type_codes,
                                     "training_qids": training, "heldout_qids": heldout})
            model_paths.append(model_path)
            
            for qid in heldout:
                rows = by_qid[qid]
                values = model.predict(feature_matrix(rows))
                score_by_doc = {str(row["doc_id"]): float(value) for row, value in zip(rows, values)}
                ordered = rank_ids(score_by_doc)
                predictions[qid] = [{"doc_id": doc_id, "rank": rank, "lambda_score": score_by_doc[doc_id], "fold": fold_name}
                                    for rank, doc_id in enumerate(ordered, 1)]
            completed = len(predictions)
            logger.status(stage="train-preranker-oof", state="RUNNING", completed=completed, total=len(by_qid))
            logger.log(f"fold={fold_name} completed={completed}/{len(by_qid)}")

        prediction_path = output_dir / "oof_predictions.jsonl"
        metric_path = output_dir / "oof_metrics.json"
        write_jsonl(prediction_path, [{"qid": qid, "candidates": values} for qid, values in sorted(predictions.items())])
        ranks = {qid: [row["doc_id"] for row in values] for qid, values in predictions.items()}
        metrics = evaluate_rankings(ranks, answers, ks=(5, 20, 24, 25, 30, 32, 50))
        atomic_json(metric_path, {"metrics": metrics, "feature_columns": list(FEATURE_COLUMNS), "type_codes": type_codes})
        rec25 = metrics.get("recall@25", metrics["recall@24"])
        logger.log(f"Recall@5={metrics['recall@5']:.6f} Recall@25={rec25:.6f} Recall@32={metrics['recall@32']:.6f}")
        return write_manifest(output_dir, stage="train-preranker-oof", v3_fingerprint=v3_fingerprint,
            config={"model": "LightGBM.LGBMRanker", "fold_isolated": True, "feature_columns": list(FEATURE_COLUMNS)},
            files=[prediction_path, metric_path, *model_paths], counts={"queries": len(predictions), "models": len(model_paths)})


def audit_shortlist(*, predictions_path: Path, train_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    records, answers = list(read_jsonl(predictions_path)), load_answers(train_path)
    ranks = {str(row["qid"]): [str(candidate["doc_id"]) for candidate in row["candidates"]] for row in records}
    metrics = evaluate_rankings(ranks, answers, ks=(5, 24, 25, 32))
    
    rec25 = metrics.get("recall@25", metrics["recall@24"])
    rec32 = metrics["recall@32"]
    shortlist = 25 if rec25 >= 0.975 else 32 if rec32 >= 0.970 else (32 if rec32 >= 0.965 else 0)
    
    report = {"status": "PASS" if shortlist else "FAIL", "metrics": metrics, "shortlist_k": shortlist,
              "rule": "prefer_25_else_32"}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shortlist_audit.json"
    with stage_run(output_dir, "audit-shortlist", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        atomic_json(path, report)
        if not shortlist:
            raise RuntimeError("shortlist Recall@25/32 gate failed (requires >= 0.965)")
        logger.log(f"shortlist_k={shortlist} Recall@{shortlist}={metrics.get(f'recall@{shortlist}', metrics['recall@32']):.6f}")
        return write_manifest(output_dir, stage="audit-shortlist", v3_fingerprint=v3_fingerprint, config={"minimum_recall": 0.965}, files=[path], counts={"queries": len(records), "shortlist_k": shortlist})


def train_final_preranker(*, candidates_path: Path, train_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:
        raise RuntimeError("lightgbm is required") from error
    records, answers = list(read_jsonl(candidates_path)), load_answers(train_path)
    by_qid, type_codes = extract_features(records)
    rows = [row for qid in sorted(by_qid) for row in by_qid[qid]]
    model = LGBMRanker(objective="lambdarank", metric="ndcg", learning_rate=0.03, n_estimators=500,
                       num_leaves=63, min_child_samples=20, random_state=42, n_jobs=-1, verbosity=-1,
                       deterministic=True, force_col_wise=True)
    with stage_run(output_dir, "train-final", total=len(by_qid), v3_fingerprint=v3_fingerprint) as logger:
        model.fit(feature_matrix(rows), np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in rows], dtype=np.int32),
                  group=[len(by_qid[qid]) for qid in sorted(by_qid)], eval_at=[5])
        path = output_dir / "final.pkl"
        dump_pickle(path, {"model": model, "feature_columns": FEATURE_COLUMNS, "type_codes": type_codes})
        return write_manifest(output_dir, stage="train-final", v3_fingerprint=v3_fingerprint, config={"all_train": True}, files=[path], counts={"queries": len(by_qid)})


def score_preranker(*, candidates_path: Path, model_path: Path, output_path: Path) -> int:
    from exp014.core import load_pickle
    model_state = load_pickle(model_path)
    records = list(read_jsonl(candidates_path))
    by_qid, _ = extract_features(records, model_state["type_codes"])
    model = model_state["model"]
    output: list[dict[str, Any]] = []
    for qid in sorted(by_qid):
        rows = by_qid[qid]
        scores = model.predict(feature_matrix(rows))
        by_doc = {str(row["doc_id"]): float(score) for row, score in zip(rows, scores)}
        output.append({"qid": qid, "candidates": [{"doc_id": doc_id, "rank": rank, "lambda_score": by_doc[doc_id]}
                                                    for rank, doc_id in enumerate(rank_ids(by_doc), 1)]})
    return write_jsonl(output_path, output)
