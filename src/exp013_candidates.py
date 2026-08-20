"""Fold-safe candidate sources for EXP-013.

The module keeps recall channels separate until a deterministic RRF union.  It
does not use relevance labels while building a document candidate record, apart
from the explicitly fold-isolated query-memory channel.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from exp012b_core import atomic_json, load_answers, load_queries, read_jsonl, stage_run, write_jsonl
from exp012b_tuning import load_folds
from exp013_core import fuse_rankings, stable_topk, validate_candidate_record, write_stage_manifest
from exp013_core import oracle_metrics
from exp013_late_interaction import l2_normalize


def _read_query_store(query_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    manifest = json.loads((query_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = json.loads((query_dir / "qids.json").read_text(encoding="utf-8"))
    vectors = np.load(query_dir / "query_vectors.f16.npy", mmap_mode="r")
    offsets = np.load(query_dir / "query_offsets.i64.npy", mmap_mode="r")
    if len(ids) + 1 != len(offsets):
        raise ValueError("Malformed EXP-013 query vector store")
    return [str(qid) for qid in ids], vectors, offsets


def retrieve_colbert_prototypes(
    query_dir: Path,
    prototype_dir: Path,
    output_dir: Path,
    *,
    v3_fingerprint: str,
    device: str = "cuda",
    top_documents: int = 20,
    batch_size: int = 4,
) -> dict[str, Any]:
    """GPU MaxSim against compact document prototypes, never all v3 leaves."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    query_ids, query_vectors, query_offsets = _read_query_store(query_dir)
    query_manifest = json.loads((query_dir / "manifest.json").read_text(encoding="utf-8"))
    prototype_manifest = json.loads((prototype_dir / "manifest.json").read_text(encoding="utf-8"))
    if query_manifest.get("v3_fingerprint") != v3_fingerprint or prototype_manifest.get("v3_fingerprint") != v3_fingerprint:
        raise RuntimeError("Query/prototype artifact has stale structural-v3 fingerprint")
    document_ids = [str(value) for value in json.loads((prototype_dir / "document_ids.json").read_text(encoding="utf-8"))]
    prototypes = np.load(prototype_dir / "prototypes.f16.npy", mmap_mode="r")
    offsets = np.load(prototype_dir / "prototype_offsets.i64.npy", mmap_mode="r")
    if len(document_ids) + 1 != len(offsets):
        raise ValueError("Malformed prototype offset store")
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / "rankings.jsonl"
    import torch
    with stage_run(output_dir, "retrieve-colbert-prototypes", total=len(query_ids), v3_fingerprint=v3_fingerprint) as logger:
        prototype_tensor = torch.tensor(np.asarray(prototypes), dtype=torch.float16, device=device)
        doc_index = torch.repeat_interleave(
            torch.arange(len(document_ids), device=device, dtype=torch.long),
            torch.tensor(np.diff(offsets), device=device, dtype=torch.long),
        )
        output: list[dict[str, Any]] = []
        for start in range(0, len(query_ids), batch_size):
            stop = min(start + batch_size, len(query_ids))
            lengths = [int(query_offsets[row + 1] - query_offsets[row]) for row in range(start, stop)]
            maximum = max(lengths)
            values = torch.zeros((stop - start, maximum, prototype_tensor.shape[1]), dtype=torch.float16, device=device)
            mask = torch.zeros((stop - start, maximum), dtype=torch.bool, device=device)
            for local, row in enumerate(range(start, stop)):
                leaf = torch.tensor(np.asarray(query_vectors[query_offsets[row]:query_offsets[row + 1]]), dtype=torch.float16, device=device)
                values[local, :len(leaf)] = leaf
                mask[local, :len(leaf)] = True
            with torch.inference_mode():
                similarities = values @ prototype_tensor.T
                per_document = torch.full((stop - start, maximum, len(document_ids)), float("-inf"), dtype=torch.float16, device=device)
                per_document.scatter_reduce_(2, doc_index.view(1, 1, -1).expand(stop - start, maximum, -1), similarities, reduce="amax", include_self=True)
                scores = (per_document.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)).float().cpu().numpy()
            for local, row in enumerate(range(start, stop)):
                order = stable_topk(scores[local], document_ids, top_documents)
                output.append({"qid": query_ids[row], "rankings": [
                    {"doc_id": document_ids[index], "rank": rank, "score": float(scores[local, index])}
                    for rank, index in enumerate(order, 1)
                ]})
            completed = stop
            if completed % 128 == 0 or completed == len(query_ids):
                logger.status(stage="retrieve-colbert-prototypes", state="RUNNING", completed=completed, total=len(query_ids))
                logger.log(f"progress={completed}/{len(query_ids)}")
        del prototype_tensor, doc_index
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        write_jsonl(rankings_path, output)
        result = write_stage_manifest(output_dir, stage="retrieve-colbert-prototypes", v3_fingerprint=v3_fingerprint,
            config={"top_documents": top_documents, "batch_size": batch_size, "backend": "cuda_scatter_maxsim_v1"},
            files=[rankings_path], counts={"queries": len(output), "documents": len(document_ids)})
    return result


def build_query_memory(
    *,
    split: str,
    train_path: Path,
    query_path: Path,
    folds_path: Path,
    output_dir: Path,
    v3_fingerprint: str,
    neighbors: int = 10,
    documents: int = 10,
) -> dict[str, Any]:
    """Character n-gram label propagation with strict fold isolation for train."""
    if split not in {"train", "public"}:
        raise ValueError("split must be train or public")
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_queries = load_queries(train_path)
    answers = load_answers(train_path)
    target_queries = load_queries(query_path)
    target_ids = sorted(target_queries)
    train_ids = sorted(train_queries)
    folds = load_folds(folds_path) if split == "train" else {}
    fold_for_qid = {str(qid): fold for fold, values in folds.items() for qid in values}
    if split == "train" and set(target_ids) != set(train_ids):
        raise ValueError("Train query-memory requires the same query set as labels")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=120000,
        dtype=np.float32, sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform([train_queries[qid] for qid in train_ids])
    index = {qid: row for row, qid in enumerate(train_ids)}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_rankings.jsonl"
    rows: list[dict[str, Any]] = []
    with stage_run(output_dir, "build-query-memory", total=len(target_ids), v3_fingerprint=v3_fingerprint) as logger:
        target_matrix = matrix if split == "train" else vectorizer.transform([target_queries[qid] for qid in target_ids])
        for position, qid in enumerate(target_ids):
            scores = (target_matrix[position] @ matrix.T).toarray().ravel()
            if split == "train":
                same_fold = fold_for_qid.get(qid)
                if same_fold is None:
                    raise ValueError(f"Missing CV fold for qid={qid}")
                for blocked in folds[same_fold]:
                    scores[index[str(blocked)]] = -np.inf
            top = stable_topk(scores, train_ids, neighbors)
            document_scores: dict[str, float] = defaultdict(float)
            provenance: dict[str, list[str]] = defaultdict(list)
            for rank, neighbor_row in enumerate(top, 1):
                similarity = float(scores[neighbor_row])
                if not np.isfinite(similarity):
                    continue
                neighbor_qid = train_ids[neighbor_row]
                for doc_id in sorted(answers[neighbor_qid]):
                    document_scores[doc_id] += similarity / rank
                    provenance[doc_id].append(neighbor_qid)
            ordered = sorted(document_scores, key=lambda doc: (-document_scores[doc], doc))[:documents]
            rows.append({"qid": qid, "rankings": [
                {"doc_id": doc, "rank": rank, "score": document_scores[doc], "neighbor_qids": provenance[doc]}
                for rank, doc in enumerate(ordered, 1)
            ]})
            if (position + 1) % 256 == 0 or position + 1 == len(target_ids):
                logger.status(stage="build-query-memory", state="RUNNING", completed=position + 1, total=len(target_ids))
                logger.log(f"progress={position + 1}/{len(target_ids)}")
        write_jsonl(output_path, rows)
        result = write_stage_manifest(output_dir, stage="build-query-memory", v3_fingerprint=v3_fingerprint,
            config={"split": split, "analyzer": "char_wb", "ngrams": [3, 5], "neighbors": neighbors, "documents": documents,
                    "fold_isolated": split == "train"}, files=[output_path], counts={"queries": len(rows), "vocabulary": len(vectorizer.vocabulary_)})
    return result


def _rankings_by_qid(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        if qid in output:
            raise ValueError(f"Duplicate qid in ranking source: {qid}")
        output[qid] = [dict(value) for value in row["rankings"]]
    return output


def build_candidate_union(
    *,
    split: str,
    source_rankings_path: Path,
    colbert_rankings_path: Path,
    memory_rankings_path: Path,
    output_dir: Path,
    v3_fingerprint: str,
    limit: int = 64,
) -> dict[str, Any]:
    """Top-20 BM25 + top-20 BGE + top-20 ColBERT + top-10 memory → RRF pool."""
    colbert = _rankings_by_qid(colbert_rankings_path)
    memory = _rankings_by_qid(memory_rankings_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_candidates.jsonl"
    rows: list[dict[str, Any]] = []
    with stage_run(output_dir, "build-candidate-union", total=len(colbert), v3_fingerprint=v3_fingerprint) as logger:
        for number, source in enumerate(read_jsonl(source_rankings_path), 1):
            qid = str(source["qid"])
            if source.get("v3_fingerprint") != v3_fingerprint:
                raise RuntimeError(f"Stale EXP-012 source ranking at qid={qid}")
            if qid not in colbert or qid not in memory:
                raise ValueError(f"Missing EXP-013 ranking source for qid={qid}")
            legacy = source["rankings"]
            channels = {
                "bm25": legacy.get("bm25", [])[:20],
                "bge": legacy.get("bge_leaf", [])[:20],
                "colbert": colbert[qid][:20],
                "memory": memory[qid][:10],
            }
            fused = fuse_rankings(channels, limit=limit)
            row = {"qid": qid, "query": str(source["query"]), "schema_version": "legalir.exp013_slid.candidates.v1",
                   "v3_fingerprint": v3_fingerprint, "candidates": fused}
            validate_candidate_record(row, maximum=limit)
            rows.append(row)
            if number % 256 == 0:
                logger.status(stage="build-candidate-union", state="RUNNING", completed=number, total=len(colbert))
        if len(rows) != len(colbert):
            raise ValueError(f"Source ranking query count mismatch: {len(rows)} != {len(colbert)}")
        write_jsonl(output_path, rows)
        result = write_stage_manifest(output_dir, stage="build-candidate-union", v3_fingerprint=v3_fingerprint,
            config={"split": split, "limit": limit, "sources": {"bm25": 20, "bge": 20, "colbert": 20, "memory": 10}, "rrf_k": 60},
            files=[output_path], counts={"queries": len(rows), "candidates": sum(len(row["candidates"]) for row in rows)})
    return result


def audit_candidate_oracle(
    *, candidates_path: Path, train_path: Path, output_dir: Path, v3_fingerprint: str,
    minimum_recall64: float = 0.985,
) -> dict[str, Any]:
    """Hard evidence before spending GPU time on exact candidate scoring."""
    answers = load_answers(train_path)
    records = list(read_jsonl(candidates_path))
    seen = {str(row["qid"]) for row in records}
    if seen != set(answers):
        raise ValueError("Candidate oracle query set differs from training answers")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "candidate_oracle.json"
    with stage_run(output_dir, "audit-candidate-oracle", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        metrics = oracle_metrics(records, answers)
        report = {"schema_version": "legalir.exp013_slid.oracle.v1", "v3_fingerprint": v3_fingerprint,
                  "metrics": metrics, "minimum_recall64": minimum_recall64,
                  "pass": metrics.get("recall@64", 0.0) >= minimum_recall64,
                  "guidance": "Proceed to exact MaxSim only when pass=true; otherwise improve candidate recall first."}
        atomic_json(report_path, report)
        result = write_stage_manifest(output_dir, stage="audit-candidate-oracle", v3_fingerprint=v3_fingerprint,
            config={"minimum_recall64": minimum_recall64}, files=[report_path], counts={"queries": len(records)})
        logger.log(f"Recall@64={metrics['recall@64']:.6f} gate={'PASS' if report['pass'] else 'FAIL'}")
    return result
