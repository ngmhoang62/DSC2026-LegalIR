"""Fold-safe LambdaMART shortlist models for EXP-013b."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from exp012b_core import atomic_json, load_answers, read_jsonl, stage_run, write_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import dump_pickle, rank_ids, write_manifest


FEATURE_COLUMNS = (
    "bge_rank", "bge_score", "bm25_rank", "bm25_score", "memory_rank", "memory_score",
    "memory_neighbors_count", "bge_reciprocal", "bm25_reciprocal", "memory_reciprocal", "rrf_score",
    "channel_count", "best_source_rank", "same_best_chunk",
    "bge_bm25_overlap", "bge_memory_overlap", "bm25_memory_overlap", "all_channel_agreement",
    "candidate_rank", "query_tokens", "document_length", "document_chunks", "article_count",
    "chunk_density", "avg_article_length", "fallback", "hierarchy_available",
    "bge_margin", "bm25_margin", "bge_diff_from_best", "bm25_diff_from_best",
    "bge_normalized", "bm25_normalized", "bge_x_bm25_score",
    "rank_disagreement", "min_rank", "document_type_code",
)


def _source(candidate: Mapping[str, Any], name: str) -> tuple[float, float, str | None, list[str]]:
    value = candidate.get("sources", {}).get(name, {})
    return float(value.get("rank", 999)), float(value.get("score", 0.0)), value.get("best_chunk_id"), list(value.get("neighbor_qids", []))


def _type_codes(records: list[dict[str, Any]]) -> dict[str, int]:
    return {name: number for number, name in enumerate(sorted({str(row.get("document_type", "unknown")) for record in records for row in record["candidates"]}))}


def _features(records: list[dict[str, Any]], type_codes: dict[str, int] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    type_codes = dict(type_codes) if type_codes is not None else _type_codes(records)
    output: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        qid, query_words = str(record["qid"]), len(str(record["query"]).split())
        cands = record["candidates"]
        bge_ranked = sorted([_source(c, "bge")[1] for c in cands if _source(c, "bge")[0] < 999], reverse=True)
        bm25_ranked = sorted([_source(c, "bm25")[1] for c in cands if _source(c, "bm25")[0] < 999], reverse=True)
        best_bge, second_bge = (bge_ranked + [0.0, 0.0])[:2]
        best_bm25, second_bm25 = (bm25_ranked + [0.0, 0.0])[:2]
        rows: list[dict[str, Any]] = []
        for candidate in cands:
            bge_rank, bge_score, bge_chunk, _ = _source(candidate, "bge")
            bm25_rank, bm25_score, bm25_chunk, _ = _source(candidate, "bm25")
            memory_rank, memory_score, _, memory_neighbors = _source(candidate, "memory")
            doc_len = float(candidate.get("document_length", 0))
            chunk_cnt = float(candidate.get("chunk_count", 0))
            art_cnt = float(candidate.get("article_count", 0))
            rr_bge = 0.0 if bge_rank >= 999 else 1.0 / bge_rank
            rr_bm25 = 0.0 if bm25_rank >= 999 else 1.0 / bm25_rank
            rr_mem = 0.0 if memory_rank >= 999 else 1.0 / memory_rank
            rrf_score = (1.0 / (60 + bge_rank)) + (1.0 / (60 + bm25_rank)) + (1.0 / (60 + memory_rank) if memory_rank < 999 else 0.0)
            same_best_chunk = float(bge_chunk is not None and bm25_chunk is not None and bge_chunk == bm25_chunk)
            row = {
                "qid": qid, "doc_id": str(candidate["doc_id"]),
                "bge_rank": bge_rank, "bge_score": bge_score, "bm25_rank": bm25_rank, "bm25_score": bm25_score,
                "memory_rank": memory_rank, "memory_score": memory_score,
                "memory_neighbors_count": float(len(memory_neighbors)),
                "bge_reciprocal": rr_bge, "bm25_reciprocal": rr_bm25, "memory_reciprocal": rr_mem, "rrf_score": rrf_score,
                "channel_count": float(candidate["channel_count"]), "best_source_rank": float(candidate["best_source_rank"]),
                "same_best_chunk": same_best_chunk,
                "bge_bm25_overlap": float(bge_rank < 999 and bm25_rank < 999),
                "bge_memory_overlap": float(bge_rank < 999 and memory_rank < 999),
                "bm25_memory_overlap": float(bm25_rank < 999 and memory_rank < 999),
                "all_channel_agreement": float(bge_rank < 999 and bm25_rank < 999 and memory_rank < 999),
                "candidate_rank": float(candidate["rank"]), "query_tokens": float(query_words),
                "document_length": doc_len, "document_chunks": chunk_cnt, "article_count": art_cnt,
                "chunk_density": chunk_cnt / max(doc_len / 1000.0, 1.0),
                "avg_article_length": doc_len / max(art_cnt, 1.0),
                "fallback": float(str(candidate.get("parse_mode")) == "fallback"),
                "hierarchy_available": float(candidate.get("hierarchy_available", 0)),
                "bge_margin": float(best_bge - second_bge), "bm25_margin": float(best_bm25 - second_bm25),
                "bge_diff_from_best": float(best_bge - bge_score), "bm25_diff_from_best": float(best_bm25 - bm25_score),
                "bge_normalized": float(bge_score / max(best_bge, 1e-5)), "bm25_normalized": float(bm25_score / max(best_bm25, 1e-5)),
                "bge_x_bm25_score": float(bge_score * bm25_score),
                "rank_disagreement": float(abs(bge_rank - bm25_rank)) if bge_rank < 999 and bm25_rank < 999 else 999.0,
                "min_rank": float(min(bge_rank, bm25_rank, memory_rank)),
                "document_type_code": float(type_codes.get(str(candidate.get("document_type", "unknown")), -1)),
            }
            rows.append(row)
        output[qid] = rows
    return output, type_codes


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[row[name] for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)


def train_preranker_oof(*, candidates_path: Path, train_path: Path, folds_path: Path, output_dir: Path,
                        v3_fingerprint: str) -> dict[str, Any]:
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:
        raise RuntimeError("lightgbm is required; run pip install -r requirements.txt") from error
    candidates, answers, folds = list(read_jsonl(candidates_path)), load_answers(train_path), load_folds(folds_path)
    by_qid, type_codes = _features(candidates)
    if set(by_qid) != set(answers): raise ValueError("candidate/train query sets differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[dict[str, Any]]] = {}
    model_paths: list[Path] = []
    with stage_run(output_dir, "train-preranker-oof", total=len(by_qid), v3_fingerprint=v3_fingerprint) as logger:
        for fold_name, heldout_values in sorted(folds.items()):
            heldout = [str(value) for value in heldout_values]
            training = sorted(set(by_qid) - set(heldout))
            if set(training) & set(heldout): raise RuntimeError("fold leakage")
            train_rows = [row for qid in training for row in by_qid[qid]]
            y_train = np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in train_rows], dtype=np.int32)
            model = LGBMRanker(objective="lambdarank", metric="ndcg", learning_rate=0.03,
                               n_estimators=450, num_leaves=63, min_child_samples=20, random_state=42,
                               n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(_matrix(train_rows), y_train, group=[len(by_qid[qid]) for qid in training], eval_at=[5])
            model_path = output_dir / f"{fold_name}.pkl"
            dump_pickle(model_path, {"model": model, "feature_columns": FEATURE_COLUMNS, "type_codes": type_codes,
                                     "training_qids": training, "heldout_qids": heldout})
            model_paths.append(model_path)
            for qid in heldout:
                rows = by_qid[qid]
                values = model.predict(_matrix(rows))
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
        metrics = evaluate_rankings(ranks, answers, ks=(5, 20, 24, 30, 32, 50))
        atomic_json(metric_path, {"metrics": metrics, "feature_columns": list(FEATURE_COLUMNS), "type_codes": type_codes})
        logger.log(f"Recall@5={metrics['recall@5']:.6f} Recall@24={metrics['recall@24']:.6f} Recall@32={metrics['recall@32']:.6f}")
        return write_manifest(output_dir, stage="train-preranker-oof", v3_fingerprint=v3_fingerprint,
            config={"model": "LightGBM.LGBMRanker", "fold_isolated": True, "feature_columns": list(FEATURE_COLUMNS)},
            files=[prediction_path, metric_path, *model_paths], counts={"queries": len(predictions), "models": len(model_paths)})


def audit_shortlist(*, predictions_path: Path, train_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    records, answers = list(read_jsonl(predictions_path)), load_answers(train_path)
    ranks = {str(row["qid"]): [str(candidate["doc_id"]) for candidate in row["candidates"]] for row in records}
    metrics = evaluate_rankings(ranks, answers, ks=(5, 24, 32))
    shortlist = 24 if metrics["recall@24"] >= 0.975 else 32 if metrics["recall@32"] >= 0.975 else 0
    report = {"status": "PASS" if shortlist else "FAIL", "metrics": metrics, "shortlist_k": shortlist,
              "rule": "prefer_24_else_32"}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shortlist_audit.json"
    with stage_run(output_dir, "audit-shortlist", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        atomic_json(path, report)
        if not shortlist: raise RuntimeError("shortlist Recall@24/32 gate failed")
        logger.log(f"shortlist_k={shortlist} Recall@{shortlist}={metrics[f'recall@{shortlist}']:.6f}")
        return write_manifest(output_dir, stage="audit-shortlist", v3_fingerprint=v3_fingerprint, config={"minimum_recall": 0.975}, files=[path], counts={"queries": len(records), "shortlist_k": shortlist})


def train_final_preranker(*, candidates_path: Path, train_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    try:
        from lightgbm import LGBMRanker
    except ImportError as error: raise RuntimeError("lightgbm is required") from error
    records, answers = list(read_jsonl(candidates_path)), load_answers(train_path)
    by_qid, type_codes = _features(records)
    rows = [row for qid in sorted(by_qid) for row in by_qid[qid]]
    model = LGBMRanker(objective="lambdarank", metric="ndcg", learning_rate=0.03, n_estimators=450,
                       num_leaves=63, min_child_samples=20, random_state=42, n_jobs=-1, verbosity=-1,
                       deterministic=True, force_col_wise=True)
    with stage_run(output_dir, "train-final", total=len(by_qid), v3_fingerprint=v3_fingerprint) as logger:
        model.fit(_matrix(rows), np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in rows], dtype=np.int32),
                  group=[len(by_qid[qid]) for qid in sorted(by_qid)], eval_at=[5])
        path = output_dir / "final.pkl"
        dump_pickle(path, {"model": model, "feature_columns": FEATURE_COLUMNS, "type_codes": type_codes})
        return write_manifest(output_dir, stage="train-final", v3_fingerprint=v3_fingerprint, config={"all_train": True}, files=[path], counts={"queries": len(by_qid)})


def score_preranker(*, candidates_path: Path, model_path: Path, output_path: Path) -> int:
    from exp013b_core import load_pickle
    model_state = load_pickle(model_path)
    records = list(read_jsonl(candidates_path))
    by_qid, _ = _features(records, model_state["type_codes"])
    model = model_state["model"]
    output: list[dict[str, Any]] = []
    for qid in sorted(by_qid):
        rows = by_qid[qid]
        scores = model.predict(_matrix(rows))
        by_doc = {str(row["doc_id"]): float(score) for row, score in zip(rows, scores)}
        output.append({"qid": qid, "candidates": [{"doc_id": doc_id, "rank": rank, "lambda_score": by_doc[doc_id]}
                                                    for rank, doc_id in enumerate(rank_ids(by_doc), 1)]})
    return write_jsonl(output_path, output)
