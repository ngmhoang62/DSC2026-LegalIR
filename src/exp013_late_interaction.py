"""Compact late-interaction index primitives for EXP-013.

This is deliberately an exact re-ranker over a small candidate set.  It does
not attempt to scan 435k passages with MaxSim at query time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from exp012b_core import atomic_json, read_jsonl, stage_run, write_jsonl
from exp013_core import MAX_ANCHORS, stable_topk, write_stage_manifest
from exp013_model import colbert_encode, load_colbert_model


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def select_anchor_indices(
    token_ids: Sequence[int], attention_mask: Sequence[int], *, maximum: int = MAX_ANCHORS
) -> np.ndarray:
    """Retain rare tokens first, while preserving deterministic text order.

    The actual ColBERT model already suppresses special/padding vectors.  The
    caller supplies that mask; this function only performs corpus compression.
    """
    active = [i for i, enabled in enumerate(attention_mask) if int(enabled)]
    if len(active) <= maximum:
        return np.asarray(active, dtype=np.int32)
    frequency = Counter(int(token_ids[i]) for i in active)
    selected = sorted(active, key=lambda i: (frequency[int(token_ids[i])], i))[:maximum]
    return np.asarray(sorted(selected), dtype=np.int32)


def quantize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantisation; scales make reconstruction exact-ish."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected [tokens, dimension] vectors")
    scales = np.maximum(np.max(np.abs(values), axis=1), 1e-8) / 127.0
    return np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8), scales.astype(np.float16)


def dequantize_rows(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


def maxsim(query_vectors: np.ndarray, document_vectors: np.ndarray) -> float:
    """ColBERT MaxSim score, normalized by query token count."""
    query = l2_normalize(query_vectors)
    document = l2_normalize(document_vectors)
    if not len(query) or not len(document):
        return float("-inf")
    return float((query @ document.T).max(axis=1).mean())


def document_prototypes(
    document_vectors: np.ndarray, *, maximum: int = 24, seed: int = 42
) -> np.ndarray:
    """Deterministic farthest-point prototypes; avoids an extra sklearn index dependency."""
    values = l2_normalize(document_vectors)
    if len(values) <= maximum:
        return values.astype(np.float16)
    # First vector is stable.  Each next vector maximizes distance to its closest selected one.
    chosen = [0]
    closest = 1.0 - values @ values[0]
    for _ in range(1, maximum):
        index = int(np.argmax(closest))
        chosen.append(index)
        closest = np.minimum(closest, 1.0 - values @ values[index])
    return values[np.asarray(chosen)].astype(np.float16)


def build_document_prototypes(
    leaf_dir: Path,
    output_dir: Path,
    *,
    v3_fingerprint: str,
    maximum: int = 24,
) -> dict[str, Any]:
    """Build doc-level routing vectors from compressed late-interaction leaves.

    ``leaf_dir`` is the output of ``encode-colbert-v3``.  Token vectors are
    held in one memmap and per-passage spans are stored as JSONL, avoiding a
    Python object per vector.
    """
    manifest = __import__("json").loads((leaf_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("v3_fingerprint") != v3_fingerprint:
        raise RuntimeError("Late-interaction leaves have stale v3 fingerprint")
    rows = list(read_jsonl(leaf_dir / "passages.jsonl"))
    vectors = np.load(leaf_dir / "token_vectors.int8.npy", mmap_mode="r")
    scales = np.load(leaf_dir / "token_scales.f16.npy", mmap_mode="r")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        start, end = int(row["token_start"]), int(row["token_end"])
        grouped[str(row["doc_id"])].append(dequantize_rows(vectors[start:end], scales[start:end]))
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = sorted(grouped)
    offsets = [0]
    pieces: list[np.ndarray] = []
    with stage_run(output_dir, "build-document-prototypes", total=len(doc_ids), v3_fingerprint=v3_fingerprint) as logger:
        for number, doc_id in enumerate(doc_ids, 1):
            proto = document_prototypes(np.concatenate(grouped[doc_id], axis=0), maximum=maximum)
            pieces.append(proto)
            offsets.append(offsets[-1] + len(proto))
            if number % 256 == 0:
                logger.status(stage="build-document-prototypes", state="RUNNING", completed=number, total=len(doc_ids))
        prototype_path = output_dir / "prototypes.f16.npy"
        offsets_path = output_dir / "prototype_offsets.i64.npy"
        ids_path = output_dir / "document_ids.json"
        np.save(prototype_path, np.concatenate(pieces, axis=0), allow_pickle=False)
        np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        atomic_json(ids_path, doc_ids)
        result = write_stage_manifest(output_dir, stage="build-document-prototypes", v3_fingerprint=v3_fingerprint,
            config={"maximum_prototypes": maximum, "selection": "deterministic_farthest_point_v1"},
            files=[prototype_path, offsets_path, ids_path], counts={"documents": len(doc_ids), "vectors": offsets[-1]})
    return result


def retrieve_prototype_documents(
    query_vectors: np.ndarray, prototype_vectors: np.ndarray,
    prototype_offsets: np.ndarray, document_ids: Sequence[str], *, limit: int = 20,
) -> list[dict[str, Any]]:
    query = l2_normalize(query_vectors)
    prototypes = l2_normalize(prototype_vectors)
    token_scores = query @ prototypes.T
    scores = np.empty(len(document_ids), dtype=np.float32)
    for row, (start, end) in enumerate(zip(prototype_offsets[:-1], prototype_offsets[1:])):
        scores[row] = token_scores[:, int(start):int(end)].max(axis=1).mean()
    order = stable_topk(scores, document_ids, limit)
    return [{"doc_id": str(document_ids[i]), "rank": rank, "score": float(scores[i])}
            for rank, i in enumerate(order, 1)]


def make_feature_row(
    *, qid: str, doc_id: str, candidate_rank: int, channels: Mapping[str, Mapping[str, float]],
    late_score: float, query_length: int, document_length: int,
) -> dict[str, Any]:
    """Flat feature schema for LambdaMART; never contains gold labels."""
    row: dict[str, Any] = {"qid": str(qid), "doc_id": str(doc_id), "candidate_rank": int(candidate_rank),
                            "late_interaction": float(late_score), "query_tokens": int(query_length),
                            "document_tokens": int(document_length)}
    for channel in ("bm25", "bge", "colbert", "memory"):
        values = channels.get(channel, {})
        row[f"{channel}_rank"] = float(values.get("rank", 999.0))
        row[f"{channel}_score"] = float(values.get("score", 0.0))
    return row


def encode_colbert_v3(
    v3_dir: Path,
    output_dir: Path,
    *,
    v3_fingerprint: str,
    model_name: str,
    device: str,
    batch_size: int = 12,
    allow_download: bool = False,
    trust_remote_code: bool = False,
    maximum_anchors: int = MAX_ANCHORS,
) -> dict[str, Any]:
    """One-time v3 encoding, stored compactly as an int8 token-vector store.

    It has no resume mode by design: a partial memmap is not a trustworthy
    index.  The output directory must be empty or contain a matching manifest.
    """
    existing = output_dir / "manifest.json"
    config = {"model": model_name, "max_anchors": maximum_anchors, "quantization": "int8-row-scale-v1"}
    if existing.exists():
        result = __import__("json").loads(existing.read_text(encoding="utf-8"))
        if result.get("v3_fingerprint") == v3_fingerprint and result.get("config") == config:
            return result
        raise RuntimeError(f"Refusing to overwrite a different late-interaction index: {output_dir}")
    rows = list(read_jsonl(v3_dir / "chunks.jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_colbert_model(model_name, device=device, allow_download=allow_download,
                                          trust_remote_code=trust_remote_code)
    passage_rows: list[dict[str, Any]] = []
    quantized_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    cursor = 0
    with stage_run(output_dir, "encode-colbert-v3", total=len(rows), v3_fingerprint=v3_fingerprint) as logger:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            vectors, ids, masks = colbert_encode(
                model, tokenizer, [str(row["retrieval_text"]) for row in batch],
                task="retrieval.passage", device=device,
            )
            for source, values, token_ids, attention in zip(batch, vectors, ids, masks):
                keep = select_anchor_indices(token_ids, attention, maximum=maximum_anchors)
                compressed = l2_normalize(values[keep])
                packed, scales = quantize_rows(compressed)
                quantized_parts.append(packed)
                scale_parts.append(scales)
                passage_rows.append({"chunk_id": str(source["chunk_id"]), "doc_id": str(source["doc_id"]),
                    "token_start": cursor, "token_end": cursor + len(packed), "original_tokens": len(values)})
                cursor += len(packed)
            completed = min(start + len(batch), len(rows))
            if completed % max(batch_size, 256) == 0 or completed == len(rows):
                logger.status(stage="encode-colbert-v3", state="RUNNING", completed=completed, total=len(rows))
                logger.log(f"progress={completed}/{len(rows)} retained_tokens={cursor}")
        vectors_path = output_dir / "token_vectors.int8.npy"
        scales_path = output_dir / "token_scales.f16.npy"
        rows_path = output_dir / "passages.jsonl"
        np.save(vectors_path, np.concatenate(quantized_parts, axis=0), allow_pickle=False)
        np.save(scales_path, np.concatenate(scale_parts, axis=0), allow_pickle=False)
        write_jsonl(rows_path, passage_rows)
        result = write_stage_manifest(output_dir, stage="encode-colbert-v3", v3_fingerprint=v3_fingerprint,
            config=config, files=[vectors_path, scales_path, rows_path],
            counts={"passages": len(rows), "token_vectors": cursor})
    import torch
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def encode_colbert_queries(
    query_path: Path,
    output_dir: Path,
    *,
    v3_fingerprint: str,
    model_name: str,
    device: str,
    batch_size: int = 32,
    allow_download: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Small reusable query token store; model is released before retrieval."""
    from exp012b_core import load_queries
    queries = load_queries(query_path)
    qids = sorted(queries)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = output_dir / "manifest.json"
    config = {"model": model_name, "task": "retrieval.query", "dimensions": 64, "max_length": 64}
    if existing.exists():
        manifest = __import__("json").loads(existing.read_text(encoding="utf-8"))
        if manifest.get("v3_fingerprint") == v3_fingerprint and manifest.get("config") == config:
            return manifest
        raise RuntimeError(f"Refusing to overwrite a different query token store: {output_dir}")
    model, tokenizer = load_colbert_model(model_name, device=device, allow_download=allow_download,
                                          trust_remote_code=trust_remote_code)
    pieces: list[np.ndarray] = []
    offsets = [0]
    with stage_run(output_dir, "encode-colbert-queries", total=len(qids), v3_fingerprint=v3_fingerprint) as logger:
        for start in range(0, len(qids), batch_size):
            batch_qids = qids[start:start + batch_size]
            vectors, _, _ = colbert_encode(model, tokenizer, [queries[qid] for qid in batch_qids],
                                            task="retrieval.query", device=device, max_length=64, dimensions=64)
            for values in vectors:
                if not len(values):
                    raise ValueError("Query tokenization produced no non-special tokens")
                pieces.append(values.astype(np.float16))
                offsets.append(offsets[-1] + len(values))
            completed = start + len(batch_qids)
            if completed % 256 == 0 or completed == len(qids):
                logger.status(stage="encode-colbert-queries", state="RUNNING", completed=completed, total=len(qids))
                logger.log(f"progress={completed}/{len(qids)} tokens={offsets[-1]}")
        values_path = output_dir / "query_vectors.f16.npy"
        offsets_path = output_dir / "query_offsets.i64.npy"
        qids_path = output_dir / "qids.json"
        np.save(values_path, np.concatenate(pieces, axis=0), allow_pickle=False)
        np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        atomic_json(qids_path, qids)
        result = write_stage_manifest(output_dir, stage="encode-colbert-queries", v3_fingerprint=v3_fingerprint,
            config=config, files=[values_path, offsets_path, qids_path], counts={"queries": len(qids), "token_vectors": offsets[-1]})
    import torch
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result
