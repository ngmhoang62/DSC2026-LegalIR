"""BGE leaf encoding, structural block construction, and hierarchical search."""

from __future__ import annotations

import json
import hashlib
import math
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from exp012b_core import (
    PIPELINE_SCHEMA,
    artifact_manifest,
    atomic_json,
    canonical_json,
    content_hash,
    load_v3_manifest,
    read_jsonl,
    sha256_file,
    stage_run,
)


DEFAULT_BGE_MODEL = "BAAI/bge-m3"


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


def _load_bge(model_name: str, device: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    if device.startswith("cuda"):
        model.half()
    return model


def _normalized(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms == 0):
        raise ValueError("Embedding contains NaN/Inf or zero-norm rows")
    return values / norms


def _encode_with_oom_fallback(
    model: Encoder,
    texts: Sequence[str],
    batch_size: int,
    *,
    device: str,
) -> tuple[np.ndarray, int]:
    current = batch_size
    while True:
        try:
            vectors = model.encode(
                list(texts),
                batch_size=current,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return _normalized(np.asarray(vectors)), current
        except RuntimeError as error:
            if "out of memory" not in str(error).casefold() or current <= 2:
                raise
            current = max(2, current // 2)
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()


def encode_v3_leaves(
    v3_dir: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_BGE_MODEL,
    device: str = "cuda",
    dimension: int = 1024,
    shard_size: int = 8192,
    batch_size: int = 8,
    encoder: Encoder | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    v3 = load_v3_manifest(v3_dir)
    total = int(v3["counts"]["chunks"])
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "leaf_embeddings.f16"
    rows_path = output_dir / "leaf_rows.jsonl"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    config = {
        "model": model_name,
        "dimension": dimension,
        "dtype": "float16",
        "shard_size": shard_size,
        "initial_batch_size": batch_size,
    }
    config_hash = content_hash({"v3": v3["content_fingerprint"], "config": config})
    existing_config = output_dir / "encode_config.json"
    if existing_config.exists():
        previous = json.loads(existing_config.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise RuntimeError("Refusing to resume leaf encoding with changed config")
    elif resume and (embeddings_path.exists() or any(shard_dir.glob("*.json"))):
        raise RuntimeError("Cannot resume unmanifested leaf encoding")
    atomic_json(existing_config, {"config_hash": config_hash, **config})

    model = encoder or _load_bge(model_name, device)
    mode = "r+" if embeddings_path.exists() else "w+"
    vectors = np.memmap(embeddings_path, dtype=np.float16, mode=mode, shape=(total, dimension))
    with stage_run(
        output_dir,
        "encode-bge-leaves",
        total=total,
        v3_fingerprint=v3["content_fingerprint"],
    ) as logger, rows_path.open("w", encoding="utf-8", newline="\n") as row_handle:
        chunk_iterator = iter(read_jsonl(v3_dir / "chunks.jsonl"))
        processed = 0
        actual_batch = batch_size
        for shard_index, start in enumerate(range(0, total, shard_size)):
            end = min(start + shard_size, total)
            rows = [next(chunk_iterator) for _ in range(end - start)]
            for row_number, chunk in enumerate(rows, start=start):
                row_handle.write(
                    canonical_json(
                        {
                            "row": row_number,
                            "chunk_id": chunk["chunk_id"],
                            "doc_id": chunk["doc_id"],
                            "parent_node_id": chunk["parent_node_id"],
                            "source_kind": chunk["source_kind"],
                            "start": chunk["start"],
                            "end": chunk["end"],
                        }
                    )
                    + "\n"
                )
            marker_path = shard_dir / f"shard_{shard_index:04d}.json"
            if resume and marker_path.exists():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                shard_hash = hashlib.sha256(
                    np.asarray(vectors[start:end], dtype=np.float16).tobytes(order="C")
                ).hexdigest()
                if (
                    marker.get("config_hash") == config_hash
                    and marker.get("start") == start
                    and marker.get("end") == end
                    and marker.get("embedding_sha256") == shard_hash
                ):
                    processed = end
                    continue
            # Stable length buckets only inside this shard; vectors are written
            # back to original corpus rows, preserving the global row map.
            order = sorted(range(len(rows)), key=lambda idx: (rows[idx]["token_count"], idx))
            for batch_start in range(0, len(order), shard_size):
                batch_indices = order[batch_start : batch_start + shard_size]
                texts = [rows[idx]["retrieval_text"] for idx in batch_indices]
                encoded, actual_batch = _encode_with_oom_fallback(
                    model, texts, actual_batch, device=device
                )
                if encoded.shape != (len(batch_indices), dimension):
                    raise ValueError(f"Unexpected BGE output shape {encoded.shape}")
                vectors[[start + idx for idx in batch_indices]] = encoded.astype(np.float16)
            vectors.flush()
            embedding_sha256 = hashlib.sha256(
                np.asarray(vectors[start:end], dtype=np.float16).tobytes(order="C")
            ).hexdigest()
            marker = {
                "schema_version": PIPELINE_SCHEMA,
                "config_hash": config_hash,
                "shard": shard_index,
                "start": start,
                "end": end,
                "actual_batch_size": actual_batch,
                "row_identity_hash": content_hash([row["chunk_id"] for row in rows]),
                "embedding_sha256": embedding_sha256,
            }
            atomic_json(marker_path, marker)
            processed = end
            logger.status(
                stage="encode-bge-leaves",
                state="RUNNING",
                completed=processed,
                total=total,
                shard=shard_index,
                actual_batch_size=actual_batch,
            )
        vectors.flush()
        finite_sample = np.asarray(vectors[:: max(1, total // 10000)], dtype=np.float32)
        if np.any(~np.isfinite(finite_sample)):
            raise ValueError("Leaf embedding sample contains NaN/Inf")
        result = artifact_manifest(
            stage="encode-bge-leaves",
            inputs={"v3_fingerprint": v3["content_fingerprint"]},
            config={**config, "actual_batch_size": actual_batch},
            files=[embeddings_path, rows_path],
        )
        result["counts"] = {"rows": total, "dimension": dimension}
        atomic_json(output_dir / "manifest.json", result)
    return result


def load_embedding_memmap(stage_dir: Path, prefix: str) -> tuple[np.memmap, list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    rows_path = stage_dir / f"{prefix}_rows.jsonl"
    embeddings_path = stage_dir / f"{prefix}_embeddings.f16"
    rows = list(read_jsonl(rows_path))
    count = int(manifest["counts"]["rows"])
    dimension = int(manifest["counts"]["dimension"])
    if len(rows) != count:
        raise ValueError(f"{prefix} row map count mismatch")
    array = np.memmap(embeddings_path, dtype=np.float16, mode="r", shape=(count, dimension))
    return array, rows, manifest


def build_structural_blocks(
    leaf_dir: Path,
    output_dir: Path,
    *,
    max_chunks_per_block: int = 32,
) -> dict[str, Any]:
    leaf_vectors, rows, leaf_manifest = load_embedding_memmap(leaf_dir, "leaf")
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["parent_node_id"]].append(int(row["row"]))
    block_specs: list[tuple[str, str, str, list[int]]] = []
    for parent_id in sorted(grouped, key=lambda value: grouped[value][0]):
        indices = grouped[parent_id]
        for part, start in enumerate(range(0, len(indices), max_chunks_per_block)):
            members = indices[start : start + max_chunks_per_block]
            first = rows[members[0]]
            block_specs.append(
                (
                    f"{parent_id}:b{part:03d}",
                    first["doc_id"],
                    parent_id,
                    members,
                )
            )
    dimension = leaf_vectors.shape[1]
    embeddings_path = output_dir / "block_embeddings.f16"
    rows_path = output_dir / "block_rows.jsonl"
    block_vectors = np.memmap(
        embeddings_path, dtype=np.float16, mode="w+", shape=(len(block_specs), dimension)
    )
    with stage_run(output_dir, "build-bge-blocks", total=len(block_specs)) as logger, rows_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for block_row, (block_id, doc_id, parent_id, members) in enumerate(block_specs):
            mean = np.asarray(leaf_vectors[members], dtype=np.float32).mean(axis=0, keepdims=True)
            vector = _normalized(mean)[0]
            block_vectors[block_row] = vector.astype(np.float16)
            handle.write(
                canonical_json(
                    {
                        "row": block_row,
                        "block_id": block_id,
                        "doc_id": doc_id,
                        "parent_node_id": parent_id,
                        "leaf_rows": members,
                        "chunk_ids": [rows[index]["chunk_id"] for index in members],
                    }
                )
                + "\n"
            )
            if (block_row + 1) % 8192 == 0:
                logger.status(
                    stage="build-bge-blocks",
                    state="RUNNING",
                    completed=block_row + 1,
                    total=len(block_specs),
                )
        block_vectors.flush()
        result = artifact_manifest(
            stage="build-bge-blocks",
            inputs={"leaf_fingerprint": leaf_manifest["content_fingerprint"]},
            config={"max_chunks_per_block": max_chunks_per_block},
            files=[embeddings_path, rows_path],
        )
        result["counts"] = {"rows": len(block_specs), "dimension": dimension}
        atomic_json(output_dir / "manifest.json", result)
    return result


def top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(scores)
    limit = min(limit, len(values))
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    selected = np.argpartition(values, -limit)[-limit:]
    return selected[np.argsort(-values[selected], kind="stable")]


def aggregate_dense_documents(
    hits: Iterable[dict[str, Any]],
    *,
    alpha: float = 0.2,
    length_penalty: float = 0.005,
    document_block_counts: dict[str, int] | None = None,
    top_docs: int = 100,
) -> list[dict[str, Any]]:
    by_doc: dict[str, dict[str, float]] = defaultdict(dict)
    for hit in hits:
        parent_scores = by_doc[str(hit["doc_id"])]
        parent = str(hit["parent_node_id"])
        parent_scores[parent] = max(parent_scores.get(parent, -math.inf), float(hit["score"]))
    output: list[dict[str, Any]] = []
    for doc_id, parent_scores in by_doc.items():
        ordered = sorted(parent_scores.values(), reverse=True)
        best = ordered[0]
        padded = (ordered + [best, best])[:3]
        count = (document_block_counts or {}).get(doc_id, len(parent_scores))
        score = best + alpha * (float(np.mean(padded)) - best) - length_penalty * math.log1p(count)
        output.append(
            {"doc_id": doc_id, "score": score, "best_score": best, "unique_parents": len(parent_scores)}
        )
    output.sort(key=lambda row: (-row["score"], row["doc_id"]))
    for rank, row in enumerate(output[:top_docs], start=1):
        row["rank"] = rank
    return output[:top_docs]


def search_blocks(
    query_vector: np.ndarray,
    block_vectors: np.ndarray,
    block_rows: Sequence[dict[str, Any]],
    *,
    top_blocks: int = 2000,
    top_docs: int = 100,
    alpha: float = 0.2,
    length_penalty: float = 0.005,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = _normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    scores = np.asarray(block_vectors, dtype=np.float32) @ query
    indices = top_indices(scores, top_blocks)
    hits = [
        {
            **block_rows[int(index)],
            "score": float(scores[index]),
            "rank": rank,
        }
        for rank, index in enumerate(indices, start=1)
    ]
    counts: dict[str, int] = defaultdict(int)
    for row in block_rows:
        counts[row["doc_id"]] += 1
    documents = aggregate_dense_documents(
        hits,
        alpha=alpha,
        length_penalty=length_penalty,
        document_block_counts=counts,
        top_docs=top_docs,
    )
    return documents, hits


def refine_candidate_leaves(
    query_vector: np.ndarray,
    candidate_doc_ids: set[str],
    leaf_vectors: np.ndarray,
    leaf_rows: Sequence[dict[str, Any]],
    *,
    top_per_doc: int = 3,
    document_rows: dict[str, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    query = _normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    if document_rows is None:
        doc_indices: dict[str, list[int]] = defaultdict(list)
        for row in leaf_rows:
            if row["doc_id"] in candidate_doc_ids:
                doc_indices[row["doc_id"]].append(int(row["row"]))
    else:
        doc_indices = {doc_id: document_rows.get(doc_id, []) for doc_id in candidate_doc_ids}
    ranked_docs: list[dict[str, Any]] = []
    evidence: dict[str, list[dict[str, Any]]] = {}
    for doc_id in sorted(candidate_doc_ids):
        indices = doc_indices.get(doc_id, [])
        if not indices:
            continue
        scores = np.asarray(leaf_vectors[indices], dtype=np.float32) @ query
        local_order = top_indices(scores, min(len(indices), max(top_per_doc * 8, top_per_doc)))
        unique: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        for local in local_order:
            global_row = indices[int(local)]
            row = leaf_rows[global_row]
            if row["parent_node_id"] in seen_parents:
                continue
            unique.append({**row, "score": float(scores[local])})
            seen_parents.add(row["parent_node_id"])
            if len(unique) == top_per_doc:
                break
        best = unique[0]["score"]
        padded = [item["score"] for item in unique] + [best, best]
        ranked_docs.append(
            {
                "doc_id": doc_id,
                "score": best + 0.2 * (float(np.mean(padded[:3])) - best),
                "best_score": best,
                "best_chunk_id": unique[0]["chunk_id"],
            }
        )
        evidence[doc_id] = unique
    ranked_docs.sort(key=lambda row: (-row["score"], row["doc_id"]))
    for rank, row in enumerate(ranked_docs, start=1):
        row["rank"] = rank
    return ranked_docs, evidence


class CudaExactLeafRefiner:
    """Gather only candidate leaves and score them in exact FP32 on CUDA.

    BM25 threads may call this object concurrently; the lock intentionally
    serializes GPU work while allowing CPU lexical search for other queries to
    continue in parallel.
    """

    def __init__(
        self,
        leaf_vectors: np.ndarray,
        leaf_rows: Sequence[dict[str, Any]],
        document_rows: dict[str, list[int]],
        *,
        device: str = "cuda",
    ) -> None:
        if not device.startswith("cuda"):
            raise ValueError("CudaExactLeafRefiner requires a CUDA device")
        import torch

        self.torch = torch
        self.device = device
        self.rows = leaf_rows
        self.document_rows = document_rows
        self.vectors = torch.tensor(
            np.asarray(leaf_vectors), dtype=torch.float32, device=device
        ).contiguous()
        self.lock = threading.Lock()

    def close(self) -> None:
        del self.vectors
        self.torch.cuda.empty_cache()

    def refine(
        self,
        query_vector: np.ndarray,
        candidate_doc_ids: set[str],
        *,
        top_per_doc: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        ordered_docs = sorted(candidate_doc_ids)
        flat_indices: list[int] = []
        spans: dict[str, tuple[int, int]] = {}
        for doc_id in ordered_docs:
            start = len(flat_indices)
            flat_indices.extend(self.document_rows.get(doc_id, ()))
            spans[doc_id] = (start, len(flat_indices))
        if not flat_indices:
            return [], {}
        query = _normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        with self.lock, self.torch.inference_mode():
            index = self.torch.tensor(flat_indices, dtype=self.torch.long, device=self.device)
            query_tensor = self.torch.tensor(query, dtype=self.torch.float32, device=self.device)
            scores = (
                self.vectors.index_select(0, index).matmul(query_tensor).cpu().numpy()
            )
        ranked_docs: list[dict[str, Any]] = []
        evidence: dict[str, list[dict[str, Any]]] = {}
        for doc_id in ordered_docs:
            start, end = spans[doc_id]
            if start == end:
                continue
            local_scores = scores[start:end]
            local_order = top_indices(
                local_scores, min(len(local_scores), max(top_per_doc * 8, top_per_doc))
            )
            unique: list[dict[str, Any]] = []
            seen_parents: set[str] = set()
            doc_rows = self.document_rows[doc_id]
            for local in local_order:
                global_row = doc_rows[int(local)]
                row = self.rows[global_row]
                if row["parent_node_id"] in seen_parents:
                    continue
                unique.append({**row, "score": float(local_scores[local])})
                seen_parents.add(row["parent_node_id"])
                if len(unique) == top_per_doc:
                    break
            best = unique[0]["score"]
            padded = [item["score"] for item in unique] + [best, best]
            ranked_docs.append(
                {
                    "doc_id": doc_id,
                    "score": best + 0.2 * (float(np.mean(padded[:3])) - best),
                    "best_score": best,
                    "best_chunk_id": unique[0]["chunk_id"],
                }
            )
            evidence[doc_id] = unique
        ranked_docs.sort(key=lambda row: (-row["score"], row["doc_id"]))
        for rank, row in enumerate(ranked_docs, start=1):
            row["rank"] = rank
        return ranked_docs, evidence
