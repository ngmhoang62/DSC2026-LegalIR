"""Exact candidate MaxSim and fold-isolated LambdaMART for EXP-013."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from exp012b_core import load_answers, read_jsonl, stage_run, write_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013_candidates import _read_query_store
from exp013_core import stable_topk, write_stage_manifest
from exp013_late_interaction import l2_normalize


FEATURE_COLUMNS = (
    "candidate_rank", "rrf_score", "best_source_rank", "exact_maxsim",
    "bm25_rank", "bm25_score", "bge_rank", "bge_score", "colbert_rank",
    "colbert_score", "memory_rank", "memory_score", "query_tokens", "document_tokens",
)


def _document_spans(passages_path: Path) -> dict[str, tuple[int, int, int]]:
    """Map doc→contiguous token interval, validating the crucial storage invariant."""
    spans: dict[str, tuple[int, int, int]] = {}
    previous: str | None = None
    for row in read_jsonl(passages_path):
        doc_id = str(row["doc_id"])
        start, end = int(row["token_start"]), int(row["token_end"])
        if previous is not None and doc_id != previous and doc_id in spans:
            raise RuntimeError(f"Document token vectors are not contiguous: {doc_id}")
        if doc_id not in spans:
            spans[doc_id] = (start, end, 1)
        else:
            old_start, old_end, count = spans[doc_id]
            if start != old_end:
                raise RuntimeError(f"Gap in token vectors for document: {doc_id}")
            spans[doc_id] = (old_start, end, count + 1)
        previous = doc_id
    return spans


def score_exact_maxsim(
    *, candidates_path: Path,
    query_dir: Path,
    leaf_dir: Path,
    output_dir: Path,
    v3_fingerprint: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Exact MaxSim over all compressed leaves of only the Top-64 parents."""
    query_ids, query_vectors, query_offsets = _read_query_store(query_dir)
    if not set(query_ids):
        raise ValueError("Empty query token store")
    qrow = {qid: index for index, qid in enumerate(query_ids)}
    spans = _document_spans(leaf_dir / "passages.jsonl")
    vectors = np.load(leaf_dir / "token_vectors.int8.npy", mmap_mode="r")
    scales = np.load(leaf_dir / "token_scales.f16.npy", mmap_mode="r")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exact_scores.jsonl"
    import torch
    records = list(read_jsonl(candidates_path))
    with stage_run(output_dir, "score-exact-maxsim", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        # A 1.25 GiB int8 store fits comfortably after the ColBERT backbone has
        # been released. Keeping it on GPU eliminates millions of disk seeks.
        vector_tensor = torch.tensor(np.asarray(vectors), dtype=torch.int8, device=device)
        scale_tensor = torch.tensor(np.asarray(scales), dtype=torch.float16, device=device)
        output: list[dict[str, Any]] = []
        for number, record in enumerate(records, 1):
            qid = str(record["qid"])
            if qid not in qrow:
                raise ValueError(f"Candidate qid absent from query store: {qid}")
            index = qrow[qid]
            query = torch.tensor(np.asarray(query_vectors[query_offsets[index]:query_offsets[index + 1]]), dtype=torch.float16, device=device)
            candidate_ids = [str(row["doc_id"]) for row in record["candidates"]]
            missing = [doc for doc in candidate_ids if doc not in spans]
            if missing:
                raise ValueError(f"Unknown candidate document(s): {missing[:3]}")
            token_parts = []
            scale_parts = []
            document_index = []
            lengths: list[int] = []
            for local, doc_id in enumerate(candidate_ids):
                start, end, _ = spans[doc_id]
                token_parts.append(vector_tensor[start:end])
                scale_parts.append(scale_tensor[start:end])
                length = end - start
                document_index.append(torch.full((length,), local, dtype=torch.long, device=device))
                lengths.append(length)
            packed = torch.cat(token_parts, dim=0).float() * torch.cat(scale_parts, dim=0).float().unsqueeze(1)
            doc_index = torch.cat(document_index)
            with torch.inference_mode():
                similarities = query.float() @ packed.T
                by_doc = torch.full((len(query), len(candidate_ids)), float("-inf"), dtype=torch.float32, device=device)
                by_doc.scatter_reduce_(1, doc_index.view(1, -1).expand(len(query), -1), similarities, reduce="amax", include_self=True)
                scores = by_doc.mean(0).cpu().numpy()
            order = stable_topk(scores, candidate_ids, len(candidate_ids))
            score_by_doc = {candidate_ids[local]: float(scores[local]) for local in range(len(candidate_ids))}
            rank_by_doc = {candidate_ids[local]: rank for rank, local in enumerate(order, 1)}
            output.append({"qid": qid, "query_tokens": int(len(query)), "scores": [
                {"doc_id": doc_id, "exact_maxsim": score_by_doc[doc_id], "exact_rank": rank_by_doc[doc_id],
                 "document_tokens": lengths[local]}
                for local, doc_id in enumerate(candidate_ids)
            ]})
            if number % 64 == 0 or number == len(records):
                logger.status(stage="score-exact-maxsim", state="RUNNING", completed=number, total=len(records))
                logger.log(f"progress={number}/{len(records)}")
        del vector_tensor, scale_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        write_jsonl(output_path, output)
        result = write_stage_manifest(output_dir, stage="score-exact-maxsim", v3_fingerprint=v3_fingerprint,
            config={"candidate_limit": 64, "backend": "cuda_exact_maxsim_int8_v1"}, files=[output_path],
            counts={"queries": len(output), "documents_scored": sum(len(row["scores"]) for row in output)})
    return result


def build_feature_rows(candidates_path: Path, exact_path: Path, output_path: Path) -> int:
    exact = {str(row["qid"]): row for row in read_jsonl(exact_path)}
    output: list[dict[str, Any]] = []
    for record in read_jsonl(candidates_path):
        qid = str(record["qid"])
        by_doc = {str(row["doc_id"]): row for row in exact[qid]["scores"]}
        for candidate in record["candidates"]:
            doc_id = str(candidate["doc_id"])
            info = by_doc[doc_id]
            sources = candidate.get("sources", {})
            feature = {"qid": qid, "doc_id": doc_id, "candidate_rank": float(candidate["rank"]),
                "rrf_score": float(candidate["rrf_score"]), "best_source_rank": float(candidate["best_source_rank"]),
                "exact_maxsim": float(info["exact_maxsim"]), "query_tokens": float(exact[qid]["query_tokens"]),
                "document_tokens": float(info["document_tokens"]), "exact_rank": float(info["exact_rank"])}
            for source in ("bm25", "bge", "colbert", "memory"):
                values = sources.get(source, {})
                feature[f"{source}_rank"] = float(values.get("rank", 999.0))
                feature[f"{source}_score"] = float(values.get("score", 0.0))
            output.append(feature)
    return write_jsonl(output_path, output)


def train_lambdamart_oof(
    *, features_path: Path,
    train_path: Path,
    folds_path: Path,
    output_dir: Path,
    v3_fingerprint: str,
) -> dict[str, Any]:
    """Five fold-isolated LambdaMART predictions, with no transformer training."""
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:
        raise RuntimeError("EXP-013 LambdaMART requires `lightgbm`; run `python -m pip install -r requirements.txt`.") from error
    answers = load_answers(train_path)
    folds = load_folds(folds_path)
    rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(features_path):
        rows_by_qid[str(row["qid"])].append(row)
    if set(rows_by_qid) != set(answers):
        raise ValueError("Feature query set differs from training labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "oof_predictions.jsonl"
    report_path = output_dir / "oof_metrics.json"
    predictions: dict[str, list[str]] = {}
    with stage_run(output_dir, "train-lambdamart-oof", total=len(rows_by_qid), v3_fingerprint=v3_fingerprint) as logger:
        for fold_name, heldout in sorted(folds.items()):
            heldout_ids = [str(qid) for qid in heldout]
            train_ids = sorted(set(rows_by_qid) - set(heldout_ids))
            train_rows = [row for qid in train_ids for row in rows_by_qid[qid]]
            test_rows = [row for qid in heldout_ids for row in rows_by_qid[qid]]
            x_train = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in train_rows], dtype=np.float32)
            y_train = np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in train_rows], dtype=np.int32)
            groups = [len(rows_by_qid[qid]) for qid in train_ids]
            model = LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=0.04,
                n_estimators=300, num_leaves=31, min_child_samples=30, random_state=42, n_jobs=-1, verbosity=-1)
            model.fit(x_train, y_train, group=groups)
            scores = model.predict(np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in test_rows], dtype=np.float32))
            cursor = 0
            for qid in heldout_ids:
                count = len(rows_by_qid[qid])
                subset = test_rows[cursor:cursor + count]
                local_scores = scores[cursor:cursor + count]
                ids = [str(row["doc_id"]) for row in subset]
                order = stable_topk(local_scores, ids, count)
                predictions[qid] = [ids[index] for index in order]
                cursor += count
            logger.log(f"fold={fold_name} train_queries={len(train_ids)} heldout_queries={len(heldout_ids)}")
            logger.status(stage="train-lambdamart-oof", state="RUNNING", completed=len(predictions), total=len(rows_by_qid))
        write_jsonl(output_path, [{"qid": qid, "doc_ids": values} for qid, values in sorted(predictions.items())])
        metrics = evaluate_rankings(predictions, answers, ks=(5, 10, 20, 30, 50, 64))
        report = {"schema_version": "legalir.exp013_slid.metrics.v1", "v3_fingerprint": v3_fingerprint,
                  "metrics": metrics, "feature_columns": list(FEATURE_COLUMNS), "folds": sorted(folds)}
        from exp012b_core import atomic_json
        atomic_json(report_path, report)
        result = write_stage_manifest(output_dir, stage="train-lambdamart-oof", v3_fingerprint=v3_fingerprint,
            config={"objective": "lambdarank", "fold_isolated": True, "feature_columns": list(FEATURE_COLUMNS)},
            files=[output_path, report_path], counts={"queries": len(predictions), "feature_rows": sum(len(rows) for rows in rows_by_qid.values())})
        logger.log(f"Recall@5={metrics['recall@5']:.6f} Precision@5={metrics['precision@5']:.6f}")
    return result
