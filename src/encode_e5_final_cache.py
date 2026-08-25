"""Resumable, fingerprint-bound VietLegal-E5 embedding cache builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from structural_chunker_v3 import canonical_json, sha256_file


MODEL_ID = "mainguyen9/vietlegal-e5"
MAX_LENGTH = 512
SCHEMA_VERSION = "legalir.e5_embedding_cache.v1"


def _chunks(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _encoder_code_sha256() -> str:
    return sha256_file(Path(__file__))


def _fingerprint(
    corpus_manifest: dict[str, Any], chunks_path: Path, *,
    model_weight_dtype: str, embedding_compute_dtype: str, storage_dtype: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION, "corpus_fingerprint": corpus_manifest["content_fingerprint"],
        "chunks_sha256": sha256_file(chunks_path), "model_id": MODEL_ID,
        "tokenizer": MODEL_ID, "document_prefix": "passage: ", "query_prefix": "query: ",
        "max_length": MAX_LENGTH, "normalization": "l2_mean_pooling",
        "model_weight_dtype": model_weight_dtype,
        "embedding_compute_dtype": embedding_compute_dtype,
        "storage_dtype": storage_dtype,
        "encoder_code_sha256": _encoder_code_sha256(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)


def build_cache(corpus_dir: Path, output_dir: Path, *, batch_size: int = 32, device: str | None = None) -> dict[str, Any]:
    corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    if not (corpus_dir / "_SUCCESS.json").exists():
        raise RuntimeError("Structural corpus lacks _SUCCESS.json")
    chunks_path = corpus_dir / "chunks.jsonl"
    chunk_count = int(corpus_manifest["counts"]["chunks"])
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch runtime has no CUDA support")
    # The selected dtype belongs to the stored vectors, not automatically to
    # model weights. AutoModel keeps the model's checkpoint dtype unless an
    # explicit dtype is requested.
    storage_dtype = "float16"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    embeddings_path = output_dir / "embeddings.f16.npy"
    ids_path = output_dir / "chunk_ids.jsonl"
    manifest_path = output_dir / "manifest.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(requested_device).eval()
    model_weight_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    embedding_compute_dtype = model_weight_dtype
    fingerprint = _fingerprint(
        corpus_manifest, chunks_path, model_weight_dtype=model_weight_dtype,
        embedding_compute_dtype=embedding_compute_dtype, storage_dtype=storage_dtype,
    )
    if (output_dir / "_SUCCESS.json").exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_fingerprint") != fingerprint:
            raise RuntimeError("Refusing cache reuse: fingerprint mismatch")
        return manifest
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"cache_fingerprint": fingerprint, "completed": 0}
    if progress.get("cache_fingerprint") != fingerprint:
        raise RuntimeError("Refusing resume: checkpoint fingerprint mismatch")
    dimension = int(getattr(model.config, "hidden_size"))
    matrix = np.lib.format.open_memmap(embeddings_path, mode="r+" if embeddings_path.exists() else "w+", dtype=np.float16, shape=(chunk_count, dimension))
    completed = int(progress["completed"])
    if completed < 0 or completed > chunk_count:
        raise RuntimeError("Invalid checkpoint completed count")
    source = _chunks(chunks_path)
    for _ in range(completed):
        next(source)
    mode = "a" if completed else "w"
    started = time.perf_counter()
    with ids_path.open(mode, encoding="utf-8", newline="\n") as ids_handle:
        pending: list[dict[str, Any]] = []
        for row in source:
            pending.append(row)
            if len(pending) < batch_size:
                continue
            completed, batch_size = _encode_batch(model, tokenizer, pending, matrix, ids_handle, completed, requested_device, batch_size, progress_path, fingerprint, chunk_count, started)
            pending = []
        if pending:
            completed, batch_size = _encode_batch(model, tokenizer, pending, matrix, ids_handle, completed, requested_device, batch_size, progress_path, fingerprint, chunk_count, started)
    if completed != chunk_count:
        raise RuntimeError(f"Incomplete encoding: {completed}/{chunk_count}")
    matrix.flush()
    manifest = {"schema_version": SCHEMA_VERSION, "cache_fingerprint": fingerprint, "corpus_fingerprint": corpus_manifest["content_fingerprint"], "chunks_sha256": sha256_file(chunks_path), "model_id": MODEL_ID, "tokenizer": MODEL_ID, "document_prefix": "passage: ", "query_prefix": "query: ", "max_length": MAX_LENGTH, "normalization": "l2_mean_pooling", "storage_dtype": storage_dtype, "model_weight_dtype": model_weight_dtype, "embedding_compute_dtype": embedding_compute_dtype, "encoder_code_sha256": _encoder_code_sha256(), "device": requested_device, "dimension": dimension, "chunks": chunk_count, "embedding_file": embeddings_path.name, "chunk_id_file": ids_path.name, "effective_final_batch_size": batch_size, "truncation_count": 0, "artifact_sha256": {"embeddings.f16.npy": sha256_file(embeddings_path), "chunk_ids.jsonl": sha256_file(ids_path)}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "_SUCCESS.json").write_text(json.dumps({"stage": "e5_embedding_cache", "cache_fingerprint": fingerprint}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def repair_completed_manifest(corpus_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Re-publish legacy completed-cache metadata without recomputing vectors.

    This is an owning-stage repair: it verifies unchanged vector/id artifacts,
    derives actual storage precision, then atomically publishes a new manifest
    and success marker. It never edits embeddings or IDs.
    """
    corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks_path = corpus_dir / "chunks.jsonl"
    manifest_path = output_dir / "manifest.json"
    success_path = output_dir / "_SUCCESS.json"
    if not manifest_path.exists() or not success_path.exists():
        raise RuntimeError("Repair requires a completed cache")
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings_path = output_dir / legacy["embedding_file"]
    ids_path = output_dir / legacy["chunk_id_file"]
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.shape != (int(legacy["chunks"]), int(legacy["dimension"])) or matrix.dtype != np.float16:
        raise RuntimeError("Embedding array shape or dtype does not match completed cache")
    with ids_path.open("r", encoding="utf-8") as handle:
        id_count = sum(1 for line in handle if line.strip())
    if id_count != int(legacy["chunks"]):
        raise RuntimeError("Chunk-ID order is incomplete")
    model_weight_dtype = "float32"
    fingerprint = _fingerprint(
        corpus_manifest, chunks_path, model_weight_dtype=model_weight_dtype,
        embedding_compute_dtype=model_weight_dtype, storage_dtype="float16",
    )
    manifest = {**legacy, "cache_fingerprint": fingerprint, "storage_dtype": "float16",
                "model_weight_dtype": model_weight_dtype, "embedding_compute_dtype": model_weight_dtype,
                "encoder_code_sha256": _encoder_code_sha256()}
    manifest.pop("model_dtype", None)
    manifest["artifact_sha256"] = {embeddings_path.name: sha256_file(embeddings_path), ids_path.name: sha256_file(ids_path)}
    progress_path = output_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["cache_fingerprint"] = fingerprint
    for path, payload in ((manifest_path, manifest), (progress_path, progress), (success_path, {"stage": "e5_embedding_cache", "cache_fingerprint": fingerprint})):
        temporary = path.with_name(path.name + ".repairing")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
    return manifest


def _encode_batch(model: Any, tokenizer: Any, rows: list[dict[str, Any]], matrix: np.ndarray, ids_handle: Any, completed: int, device: str, batch_size: int, progress_path: Path, fingerprint: str, total: int, started: float) -> tuple[int, int]:
    texts = ["passage: " + str(row["retrieval_text"]) for row in rows]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=True, return_tensors="pt")
    longest = int(encoded["attention_mask"].sum(dim=1).max().item())
    if longest > MAX_LENGTH:
        raise RuntimeError(f"Document-input truncation invariant failed: {longest} > {MAX_LENGTH}")
    while True:
        try:
            with torch.inference_mode():
                inputs = {key: value.to(device) for key, value in encoded.items()}
                embeddings = torch.nn.functional.normalize(_mean_pool(model(**inputs).last_hidden_state, inputs["attention_mask"]), p=2, dim=1)
            matrix[completed:completed + len(rows)] = embeddings.cpu().numpy().astype(np.float16)
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            for start in range(0, len(rows), batch_size):
                completed, batch_size = _encode_batch(
                    model, tokenizer, rows[start:start + batch_size], matrix, ids_handle,
                    completed, device, batch_size, progress_path, fingerprint, total, started,
                )
            return completed, batch_size
    for row in rows:
        ids_handle.write(canonical_json({"chunk_id": row["chunk_id"], "doc_id": row["doc_id"]}) + "\n")
    completed += len(rows)
    matrix.flush(); ids_handle.flush()
    elapsed = max(time.perf_counter() - started, 1e-6)
    eta = (total - completed) / (completed / elapsed)
    progress_path.write_text(json.dumps({"cache_fingerprint": fingerprint, "completed": completed, "total": total, "effective_batch_size": batch_size, "elapsed_seconds": round(elapsed, 1), "eta_seconds": round(eta, 1)}, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    if completed % max(batch_size, 1000) == 0 or completed == total:
        print(f"Encoded {completed}/{total}; ETA {eta / 60:.1f} min; batch {batch_size}", flush=True)
    return completed, batch_size


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device")
    parser.add_argument("--repair-completed-manifest", action="store_true")
    args = parser.parse_args(argv)
    action = repair_completed_manifest if args.repair_completed_manifest else build_cache
    print(json.dumps(action(args.corpus_dir, args.output_dir) if args.repair_completed_manifest else action(args.corpus_dir, args.output_dir, batch_size=args.batch_size, device=args.device), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
