"""Single orchestration CLI for the v3-native EXP-012b pipeline.

Long stages are resumable and write their own status/log/success markers.  The
CLI intentionally imports model libraries only inside stages that need them.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from exp012b_bm25 import (
    BM25Searcher,
    aggregate_bm25_documents,
    build_fts5_index,
    default_segmenter,
    safe_fts_query,
    tokenize_v3_fields,
)
from exp012b_core import (
    PIPELINE_SCHEMA,
    atomic_json,
    canonical_json,
    content_hash,
    load_answers,
    load_queries,
    load_v3_manifest,
    read_jsonl,
    require_success,
    sha256_file,
    stage_run,
)
from exp012b_dense import (
    _load_bge,
    _normalized,
    aggregate_dense_documents,
    build_structural_blocks,
    encode_v3_leaves,
    load_embedding_memmap,
    refine_candidate_leaves,
    CudaExactLeafRefiner,
)
from exp012b_retrieval import (
    build_chunk_offset_index,
    build_evidence_record,
    candidate_union,
    evaluate_rankings,
    load_chunk_records,
    PersistentChunkReader,
    select_evidence_ids,
    weighted_rrf,
)
from exp012b_reranker import score_evidence_records, train_lora_pairwise, unload_cuda
from exp012b_tuning import evaluate_zero_shot_artifacts
from exp012b_tuning import load_folds, nested_tune_stage1
from exp012b_training_data import mine_fold_pair_cache
from exp012b_lora_eval import (
    build_public_submission,
    evaluate_lora_fold,
    evaluate_oof,
    load_adapter,
    score_lora_fold,
)
from exp012b_performance import configure_process, execution_config, profile_metadata
from exp012b_token_cache import build_token_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache" / "exp012b_v3"
DEFAULT_RESULTS = ROOT / "results" / "exp012b_v3"
FAST_CACHE = ROOT / "cache" / "exp012b_v3_fast"
FAST_RESULTS = ROOT / "results" / "exp012b_v3_fast"
DEFAULT_V3 = ROOT / "cache" / "structural_v3"
DEFAULT_TRAIN = ROOT / "public_test_dataset" / "train.json"
DEFAULT_PUBLIC = ROOT / "public_test_dataset" / "public-official.json"
DEFAULT_AUDIT = ROOT / "results" / "audits" / "chunker_v3" / "audit.json"


STAGE_DIRS = {
    "tokenize-bm25": "bm25_fields",
    "build-bm25": "bm25_index",
    "encode-bge-leaves": "bge_leaves",
    "build-bge-blocks": "bge_blocks",
    "retrieve-hybrid": "rankings",
    "route-evidence": "evidence",
    "score-zero-shot": "zero_shot",
    "train-lora-fold": "lora",
}


def _query_path(split: str) -> Path:
    return DEFAULT_TRAIN if split == "train" else DEFAULT_PUBLIC


def completed_stage(primary: Path, fallback: Path, relative: Path | str) -> Path:
    """Prefer an optimized artifact, otherwise read the frozen reference one."""
    local = primary / relative
    if (local / "_SUCCESS.json").exists():
        return local
    reference = fallback / relative
    if (reference / "_SUCCESS.json").exists():
        return reference
    return local


def preflight(v3_dir: Path, audit_path: Path) -> dict[str, Any]:
    # Preflight is the trust boundary: it always performs full SHA-256.  Its
    # stat receipts let optimized downstream commands avoid re-reading GBs of
    # unchanged structural artifacts while still detecting edits/reformats.
    v3 = load_v3_manifest(v3_dir, full_verify=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if audit.get("status") != "PASS":
        errors.append(f"structural audit status={audit.get('status')}")
    if audit.get("content_fingerprint") != v3["content_fingerprint"]:
        errors.append("structural audit fingerprint differs from v3 cache")
    if int(v3["counts"]["chunks"]) != 435316:
        errors.append(f"unexpected v3 chunk count={v3['counts']['chunks']}")
    if errors:
        raise RuntimeError("Preflight failed: " + "; ".join(errors))
    return {
        "schema_version": PIPELINE_SCHEMA,
        "status": "PASS",
        "v3_fingerprint": v3["content_fingerprint"],
        "counts": v3["counts"],
        "integrity_receipts": v3["integrity_receipts"],
    }


def encode_queries(
    query_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    device: str,
) -> tuple[list[str], np.ndarray]:
    queries = load_queries(query_path)
    qids = sorted(queries)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "query_embeddings.f16.npy"
    rows_path = output_dir / "query_rows.jsonl"
    manifest_path = output_dir / "query_embeddings.manifest.json"
    cache_key = content_hash(
        {"query_sha256": sha256_file(query_path), "model": model_name, "qids": qids}
    )
    if vectors_path.exists() and rows_path.exists() and manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            cached.get("cache_key") == cache_key
            and cached.get("vectors_sha256") == sha256_file(vectors_path)
            and cached.get("rows_sha256") == sha256_file(rows_path)
        ):
            return qids, np.load(vectors_path, mmap_mode="r")
    model = _load_bge(model_name, device)
    vectors = model.encode(
        [queries[qid] for qid in qids],
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = _normalized(np.asarray(vectors)).astype(np.float16)
    np.save(vectors_path, vectors, allow_pickle=False)
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, qid in enumerate(qids):
            handle.write(canonical_json({"row": row, "qid": qid, "query": queries[qid]}) + "\n")
    atomic_json(
        manifest_path,
        {
            "cache_key": cache_key,
            "vectors_sha256": sha256_file(vectors_path),
            "rows_sha256": sha256_file(rows_path),
            "queries": len(qids),
        },
    )
    unload_cuda(model)
    return qids, vectors


def retrieve_hybrid(
    *,
    split: str,
    cache_root: Path,
    v3_dir: Path,
    model_name: str,
    device: str,
    bm25_profile: str = "legal_structure",
    top_passages: int = 2000,
    top_blocks: int = 2000,
    workers: int = 4,
    resume: bool = False,
    leaf_backend: str = "cpu_reference",
    upstream_cache_root: Path | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    v3 = load_v3_manifest(v3_dir)
    output_dir = cache_root / "rankings" / split
    upstream = upstream_cache_root or cache_root
    bm25_dir = completed_stage(cache_root, upstream, "bm25_index")
    leaf_dir = completed_stage(cache_root, upstream, "bge_leaves")
    block_dir = completed_stage(cache_root, upstream, "bge_blocks")
    bm25_database = bm25_dir / "bm25_v3.sqlite"
    leaf_vectors, leaf_rows, leaf_manifest = load_embedding_memmap(leaf_dir, "leaf")
    block_vectors, block_rows, block_manifest = load_embedding_memmap(block_dir, "block")
    query_path = _query_path(split)
    qids, query_vectors = encode_queries(query_path, output_dir, model_name=model_name, device=device)
    queries = load_queries(query_path)
    document_leaf_rows: dict[str, list[int]] = defaultdict(list)
    scope_parent_rows: dict[str, list[int]] = defaultdict(list)
    for row in leaf_rows:
        document_leaf_rows[row["doc_id"]].append(int(row["row"]))
        scope_parent_rows[row["parent_node_id"]].append(int(row["row"]))

    block_counts: dict[str, int] = defaultdict(int)
    for row in block_rows:
        block_counts[row["doc_id"]] += 1

    documents_meta = {row["doc_id"]: row for row in read_jsonl(v3_dir / "documents.jsonl")}
    output_path = output_dir / "hybrid_candidates.jsonl"
    source_path = output_dir / "source_rankings.jsonl"
    predictions_by_channel: dict[str, dict[str, list[str]]] = {
        "bm25": {},
        "bge": {},
        "hybrid": {},
    }
    config_hash = content_hash(
        {
            "v3": v3["content_fingerprint"],
            "leaf": leaf_manifest["content_fingerprint"],
            "block": block_manifest["content_fingerprint"],
            "split": split,
            "model": model_name,
            "bm25_profile": bm25_profile,
            "top_passages": top_passages,
            "top_blocks": top_blocks,
            "leaf_backend": leaf_backend,
        }
    )
    shard_dir = output_dir / "source_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed_shards: list[Path] = []
    with stage_run(
        output_dir,
        "retrieve-hybrid",
        total=len(qids),
        v3_fingerprint=v3["content_fingerprint"],
    ) as logger:
        import torch

        logger.log(f"phase=load-block-matrix workers={workers}")
        # torch.tensor performs an owned copy and avoids the undefined-behaviour
        # warning from wrapping the read-only NumPy memmap with from_numpy.
        block_tensor = torch.tensor(
            np.asarray(block_vectors), device=device, dtype=torch.float16
        )
        cuda_refiner = (
            CudaExactLeafRefiner(
                leaf_vectors, leaf_rows, document_leaf_rows, device=device
            )
            if leaf_backend == "cuda_exact" else None
        )
        thread_state = threading.local()
        created_searchers: list[BM25Searcher] = []
        searcher_lock = threading.Lock()

        def worker_searcher() -> BM25Searcher:
            searcher = getattr(thread_state, "searcher", None)
            if searcher is None:
                searcher = BM25Searcher(bm25_database, profile=bm25_profile)
                thread_state.searcher = searcher
                with searcher_lock:
                    created_searchers.append(searcher)
            return searcher

        def process_query(
            query_index: int,
            qid: str,
            block_docs: list[dict[str, Any]],
            expression: str,
        ) -> tuple[dict[str, Any], list[str], list[str]]:
            query = queries[qid]
            bm25_hits = worker_searcher().search_expression(expression, limit=top_passages)
            bm25_docs = aggregate_bm25_documents(bm25_hits, top_docs=100)
            candidates = candidate_union(bm25_docs, block_docs)
            if cuda_refiner is not None:
                leaf_docs, dense_evidence = cuda_refiner.refine(
                    query_vectors[query_index], candidates, top_per_doc=8
                )
            else:
                leaf_docs, dense_evidence = refine_candidate_leaves(
                    query_vectors[query_index], candidates, leaf_vectors, leaf_rows,
                    top_per_doc=8, document_rows=document_leaf_rows,
                )
            block_doc_ids = {row["doc_id"] for row in block_docs}
            bge_only_docs = [dict(row) for row in leaf_docs if row["doc_id"] in block_doc_ids]
            for bge_rank, row in enumerate(bge_only_docs, start=1):
                row["rank"] = bge_rank
            bm25_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for hit in bm25_hits:
                bm25_by_doc[hit["doc_id"]].append(hit)
            evidence_by_doc: dict[str, dict[str, Any]] = {}
            for doc_id in sorted(candidates):
                selected = select_evidence_ids(
                    dense_evidence.get(doc_id, []), bm25_by_doc.get(doc_id, []), maximum=3
                )
                if not selected:
                    continue
                scope_rows = [
                    row_index
                    for parent in documents_meta[doc_id].get("scope_node_ids", [])
                    for row_index in scope_parent_rows.get(parent, [])
                ]
                scope_chunk_id = None
                if scope_rows:
                    scope_scores = np.asarray(leaf_vectors[scope_rows], dtype=np.float32) @ np.asarray(
                        query_vectors[query_index], dtype=np.float32
                    )
                    scope_chunk_id = leaf_rows[scope_rows[int(np.argmax(scope_scores))]]["chunk_id"]
                evidence_by_doc[doc_id] = {
                    "evidence": selected,
                    "document_label": documents_meta[doc_id]["document_label"],
                    "scope_chunk_id": scope_chunk_id,
                }
            return (
                {
                    "schema_version": PIPELINE_SCHEMA,
                    "v3_fingerprint": v3["content_fingerprint"],
                    "qid": qid,
                    "query": query,
                    "bm25_profile": bm25_profile,
                    "rankings": {"bm25": bm25_docs, "bge_block": block_docs, "bge_leaf": leaf_docs},
                    "evidence_by_doc": evidence_by_doc,
                },
                [row["doc_id"] for row in bm25_docs],
                [row["doc_id"] for row in bge_only_docs],
            )

        retrieval_started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hybrid-retrieve") as executor:
                for shard_index, start in enumerate(range(0, len(qids), 64)):
                    shard_qids = qids[start : start + 64]
                    shard_path = shard_dir / f"source_{shard_index:04d}.jsonl"
                    marker_path = shard_dir / f"source_{shard_index:04d}.json"
                    if resume and shard_path.exists() and marker_path.exists():
                        marker = json.loads(marker_path.read_text(encoding="utf-8"))
                        if (
                            marker.get("config_hash") == config_hash
                            and marker.get("qids") == shard_qids
                            and marker.get("sha256") == sha256_file(shard_path)
                        ):
                            completed_shards.append(shard_path)
                            logger.status(
                                stage="retrieve-hybrid", state="RUNNING", phase="retrieve",
                                completed=start + len(shard_qids), total=len(qids), resumed=True,
                            )
                            continue
                    logger.log(
                        f"phase=retrieve shard={shard_index} queries={len(shard_qids)}"
                    )
                    expressions = {
                        qid: safe_fts_query(default_segmenter(queries[qid])) for qid in shard_qids
                    }
                    with torch.inference_mode():
                        query_batch = torch.tensor(
                            np.asarray(query_vectors[start : start + len(shard_qids)]),
                            device=device, dtype=torch.float16,
                        )
                        scores = torch.matmul(query_batch, block_tensor.T)
                        values, indices = torch.topk(scores, k=min(top_blocks, scores.shape[1]), dim=1)
                        values_cpu = values.float().cpu().numpy()
                        indices_cpu = indices.cpu().numpy()
                    del scores, values, indices, query_batch
                    jobs = []
                    for local, qid in enumerate(shard_qids):
                        hits = [
                            {
                                **block_rows[int(index)],
                                "score": float(values_cpu[local, rank - 1]),
                                "rank": rank,
                            }
                            for rank, index in enumerate(indices_cpu[local], start=1)
                        ]
                        block_docs = aggregate_dense_documents(
                            hits, alpha=0.2, length_penalty=0.005,
                            document_block_counts=block_counts, top_docs=100,
                        )
                        jobs.append(
                            executor.submit(
                                process_query, start + local, qid, block_docs, expressions[qid]
                            )
                        )
                    records = [job.result() for job in jobs]
                    temporary = shard_path.with_suffix(".jsonl.tmp")
                    with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
                        for record, bm25_prediction, bge_prediction in records:
                            handle.write(canonical_json(record) + "\n")
                            predictions_by_channel["bm25"][record["qid"]] = bm25_prediction
                            predictions_by_channel["bge"][record["qid"]] = bge_prediction
                    temporary.replace(shard_path)
                    atomic_json(
                        marker_path,
                        {
                            "config_hash": config_hash,
                            "qids": shard_qids,
                            "sha256": sha256_file(shard_path),
                        },
                    )
                    completed_shards.append(shard_path)
                    completed = start + len(shard_qids)
                    elapsed = time.perf_counter() - retrieval_started
                    rate = completed / elapsed if elapsed else 0.0
                    logger.status(
                        stage="retrieve-hybrid", state="RUNNING", phase="retrieve",
                        completed=completed, total=len(qids), workers=workers,
                        queries_per_second=rate,
                        eta_seconds=(len(qids) - completed) / rate if rate else None,
                    )
        finally:
            for searcher in created_searchers:
                searcher.close()
            if cuda_refiner is not None:
                cuda_refiner.close()
        del block_tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if len(completed_shards) != math.ceil(len(qids) / 64):
            raise RuntimeError("Incomplete retrieval shard set")
        temporary_source = source_path.with_suffix(".jsonl.tmp")
        with temporary_source.open("wb") as output_handle:
            for shard_path in completed_shards:
                with shard_path.open("rb") as shard_handle:
                    while block := shard_handle.read(8 * 1024 * 1024):
                        output_handle.write(block)
        temporary_source.replace(source_path)
        metrics = {}
        # Keep only compact source rankings for nested tuning. Evidence maps
        # stay on disk and are streamed again when materializing Top-50.
        source_by_qid = {row["qid"]: row["rankings"] for row in read_jsonl(source_path)}
        chosen_by_fold: dict[str, dict[str, float]] = {}
        selected_public_weights = {"bm25": 1.0, "bge_block": 1.0, "bge_leaf": 1.5}
        if split == "train":
            def bge_channel_ids(rankings: dict[str, list[dict[str, Any]]]) -> list[str]:
                block_ids = {item["doc_id"] for item in rankings["bge_block"]}
                return [row["doc_id"] for row in rankings["bge_leaf"] if row["doc_id"] in block_ids]

            predictions_by_channel["bm25"] = {
                qid: [row["doc_id"] for row in rankings["bm25"]]
                for qid, rankings in source_by_qid.items()
            }
            predictions_by_channel["bge"] = {
                qid: bge_channel_ids(rankings)
                for qid, rankings in source_by_qid.items()
            }
            answers = load_answers(DEFAULT_TRAIN)
            folds = load_folds(ROOT / "cache" / "cv_folds.json")
            nested = nested_tune_stage1(source_by_qid, answers, folds)
            predictions_by_channel["hybrid"] = nested["predictions"]
            chosen_by_fold = nested["chosen_weights"]
            metrics = {
                channel: evaluate_rankings(predictions, answers)
                for channel, predictions in predictions_by_channel.items()
            }
            frequency: dict[str, int] = defaultdict(int)
            by_key: dict[str, dict[str, float]] = {}
            for weights in chosen_by_fold.values():
                key = canonical_json(weights)
                frequency[key] += 1
                by_key[key] = weights
            selected_public_weights = by_key[sorted(frequency, key=lambda key: (-frequency[key], key))[0]]
            if metrics["hybrid"]["recall@50"] + 1e-12 < max(
                metrics["bm25"]["recall@50"], metrics["bge"]["recall@50"]
            ):
                raise RuntimeError("Hybrid Recall@50 is below the best single channel")
            if metrics["hybrid"]["recall@50"] < 0.965:
                raise RuntimeError(
                    f"Stage-1 gate failed: hybrid Recall@50={metrics['hybrid']['recall@50']:.6f} < 0.965"
                )
        else:
            train_manifest_path = cache_root / "rankings" / "train" / "manifest.json"
            if not train_manifest_path.exists():
                raise RuntimeError("Train Stage-1 tuning must complete before public retrieval")
            selected_public_weights = json.loads(train_manifest_path.read_text(encoding="utf-8"))[
                "selected_public_weights"
            ]

        fold_by_qid = {
            qid: fold for fold, fold_qids in (load_folds(ROOT / "cache" / "cv_folds.json").items() if split == "train" else [])
            for qid in fold_qids
        }
        with output_path.open("w", encoding="utf-8", newline="\n") as output_handle:
            for source_record in read_jsonl(source_path):
                qid = source_record["qid"]
                weights = (
                    chosen_by_fold[fold_by_qid[qid]] if split == "train" else selected_public_weights
                )
                hybrid = weighted_rrf(source_record["rankings"], weights=weights, limit=50)
                for candidate in hybrid:
                    metadata = source_record["evidence_by_doc"].get(candidate["doc_id"])
                    if metadata is None:
                        raise RuntimeError(f"No v3 evidence for {qid}/{candidate['doc_id']}")
                    candidate.update(metadata)
                output_handle.write(
                    canonical_json(
                        {
                            "schema_version": PIPELINE_SCHEMA,
                            "v3_fingerprint": v3["content_fingerprint"],
                            "qid": qid,
                            "query": source_record["query"],
                            "bm25_profile": bm25_profile,
                            "rrf_weights": weights,
                            "candidates": hybrid,
                        }
                    )
                    + "\n"
                )
        manifest = {
            "schema_version": PIPELINE_SCHEMA,
            "stage": "retrieve-hybrid",
            "v3_fingerprint": v3["content_fingerprint"],
            "leaf_fingerprint": leaf_manifest["content_fingerprint"],
            "block_fingerprint": block_manifest["content_fingerprint"],
            "split": split,
            "queries": len(qids),
            "metrics": metrics,
            "chosen_weights_by_fold": chosen_by_fold,
            "selected_public_weights": selected_public_weights,
            "execution": {"leaf_backend": leaf_backend, "workers": workers},
            "artifact_sha256": {
                source_path.name: sha256_file(source_path),
                output_path.name: sha256_file(output_path),
            },
        }
        manifest["content_fingerprint"] = content_hash(manifest)
        atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def route_evidence(
    split: str,
    cache_root: Path,
    v3_dir: Path,
    *,
    resume: bool = False,
    batch_queries: int = 32,
    persistent_reader: bool = False,
    upstream_cache_root: Path | None = None,
) -> dict[str, Any]:
    v3 = load_v3_manifest(v3_dir)
    source_path = cache_root / "rankings" / split / "hybrid_candidates.jsonl"
    ranking_manifest = json.loads((source_path.parent / "manifest.json").read_text(encoding="utf-8"))
    query_count = int(ranking_manifest["queries"])
    lookup_dir = completed_stage(cache_root, upstream_cache_root or cache_root, "chunk_lookup")
    lookup_database = lookup_dir / "chunk_offsets.sqlite"
    output_dir = cache_root / "evidence" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evidence.jsonl"
    pair_count = 0
    config_hash = content_hash(
        {
            "schema": "route-evidence-shards-v1",
            "v3": v3["content_fingerprint"],
            "ranking": ranking_manifest.get("content_fingerprint"),
            "model": "BAAI/bge-reranker-v2-m3",
            "batch_queries": batch_queries,
            "persistent_reader": persistent_reader,
        }
    )
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with stage_run(
        output_dir,
        "route-evidence",
        total=query_count,
        v3_fingerprint=v3["content_fingerprint"],
    ) as logger:
        from transformers import AutoTokenizer

        logger.log("phase=load-tokenizer model=BAAI/bge-reranker-v2-m3")
        tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-v2-m3", local_files_only=True, use_fast=True
        )
        completed_shards: list[Path] = []
        processed = 0
        source_iterator = iter(read_jsonl(source_path))
        reader = (
            PersistentChunkReader(v3_dir, lookup_database, cache_size=16384)
            if persistent_reader else None
        )
        for shard_index, start in enumerate(range(0, query_count, batch_queries)):
            batch = []
            for _ in range(min(batch_queries, query_count - start)):
                try:
                    batch.append(next(source_iterator))
                except StopIteration as error:
                    raise RuntimeError(
                        f"Candidate query count is below manifest at {start}"
                    ) from error
            shard_path = shard_dir / f"evidence_{shard_index:04d}.jsonl"
            marker_path = shard_dir / f"evidence_{shard_index:04d}.json"
            qid_batch = [str(row["qid"]) for row in batch]
            if resume and shard_path.exists() and marker_path.exists():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if (
                    marker.get("config_hash") == config_hash
                    and marker.get("qids") == qid_batch
                    and marker.get("sha256") == sha256_file(shard_path)
                ):
                    completed_shards.append(shard_path)
                    processed += len(batch)
                    pair_count += int(marker["pairs"])
                    logger.status(
                        stage="route-evidence", state="RUNNING", phase="route",
                        completed=processed, total=query_count, pairs=pair_count, resumed=True,
                    )
                    continue
            needed: set[str] = set()
            for record in batch:
                for candidate in record["candidates"]:
                    needed.update(item["chunk_id"] for item in candidate["evidence"])
                    if candidate.get("scope_chunk_id"):
                        needed.add(candidate["scope_chunk_id"])
            chunks = (
                reader.load(needed)
                if reader is not None
                else load_chunk_records(v3_dir, needed, lookup_database=lookup_database)
            )
            shard_pairs = 0
            temporary = shard_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
                for record in batch:
                    routed = []
                    for candidate in record["candidates"]:
                        scope_id = candidate.get("scope_chunk_id")
                        scope_excerpt = chunks[scope_id]["raw_text"] if scope_id else ""
                        evidence_record = build_evidence_record(
                            qid=record["qid"], query=record["query"], candidate=candidate,
                            selected=candidate["evidence"], chunks=chunks,
                            document_label=candidate["document_label"],
                            v3_fingerprint=v3["content_fingerprint"], scope_excerpt=scope_excerpt,
                            tokenizer=tokenizer,
                        )
                        routed.append({**candidate, "evidence": evidence_record["evidence"]})
                        shard_pairs += len(evidence_record["evidence"])
                    handle.write(canonical_json({
                        "schema_version": PIPELINE_SCHEMA,
                        "v3_fingerprint": v3["content_fingerprint"],
                        "qid": record["qid"], "query": record["query"], "candidates": routed,
                    }) + "\n")
            temporary.replace(shard_path)
            atomic_json(
                marker_path,
                {
                    "config_hash": config_hash,
                    "qids": qid_batch,
                    "pairs": shard_pairs,
                    "sha256": sha256_file(shard_path),
                },
            )
            completed_shards.append(shard_path)
            processed += len(batch)
            pair_count += shard_pairs
            logger.status(
                stage="route-evidence", state="RUNNING", phase="route",
                completed=processed, total=query_count, pairs=pair_count,
            )
            logger.log(
                f"phase=route shard={shard_index} queries={len(batch)} pairs={shard_pairs}"
            )
        if reader is not None:
            reader.close()
        try:
            next(source_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Candidate query count exceeds manifest")
        if processed != query_count:
            raise RuntimeError(f"Evidence query count mismatch: {processed}/{query_count}")
        temporary_output = output_path.with_suffix(".jsonl.tmp")
        with temporary_output.open("wb") as output_handle:
            for shard_path in completed_shards:
                with shard_path.open("rb") as shard_handle:
                    while block := shard_handle.read(8 * 1024 * 1024):
                        output_handle.write(block)
        temporary_output.replace(output_path)
        manifest = {
            "schema_version": PIPELINE_SCHEMA,
            "stage": "route-evidence",
            "split": split,
            "v3_fingerprint": v3["content_fingerprint"],
            "queries": query_count,
            "pairs": pair_count,
            "artifact_sha256": {output_path.name: sha256_file(output_path)},
        }
        manifest["content_fingerprint"] = content_hash(manifest)
        atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def show_status(cache_root: Path) -> int:
    found = False
    for status_path in sorted(cache_root.glob("**/status.json")):
        found = True
        status = json.loads(status_path.read_text(encoding="utf-8"))
        extras = []
        if status.get("phase"):
            extras.append(f"phase={status['phase']}")
        if status.get("eta_seconds") is not None:
            extras.append(f"eta={float(status['eta_seconds']) / 60:.1f}m")
        suffix = f" ({', '.join(extras)})" if extras else ""
        print(
            f"{status_path.parent.relative_to(cache_root)}: "
            f"{status.get('state')} {status.get('completed')}/{status.get('total')}{suffix}"
        )
    if not found:
        print("No EXP-012b stages have started.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[
        "preflight", "build-chunk-lookup", "tokenize-bm25", "build-bm25", "encode-bge-leaves",
        "build-bge-blocks", "retrieve-hybrid", "route-evidence",
        "score-zero-shot", "evaluate-zero-shot", "mine-lora", "train-lora-fold",
        "prepare-lora-teacher",
        "build-token-cache",
        "score-lora-fold", "evaluate-lora-fold", "evaluate-oof", "mine-final",
        "train-final", "score-final-public", "public-submission", "status",
    ])
    parser.add_argument("--split", choices=["train", "public"], default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execution-profile", choices=["reference", "optimized"], default="reference",
        help="Reference preserves current behavior; optimized uses isolated fast artifacts.",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--upstream-cache-root", type=Path,
        help="Read-only fallback for completed upstream artifacts; optimized defaults to reference.",
    )
    parser.add_argument("--v3-dir", type=Path, default=DEFAULT_V3)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--pairs-train", type=Path)
    parser.add_argument("--pairs-validation", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Bounded CPU workers for exact retrieval/mining (default: 4)",
    )
    parser.add_argument(
        "--pair-batch-size", type=int, default=4,
        help="LoRA micro-batch; candidates above 4 require fold-0 metric promotion.",
    )
    parser.add_argument(
        "--reranker-batch-size", type=int,
        help="Override the profile base batch for bounded 256-query speed tests.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Experimental LoRA speed profile; validate VRAM and rerun fold 0 before promotion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = execution_config(args.execution_profile)
    args.cache_root = args.cache_root or (
        FAST_CACHE if args.execution_profile == "optimized" else DEFAULT_CACHE
    )
    args.results_root = args.results_root or (
        FAST_RESULTS if args.execution_profile == "optimized" else DEFAULT_RESULTS
    )
    args.upstream_cache_root = args.upstream_cache_root or (
        DEFAULT_CACHE if args.execution_profile == "optimized" else args.cache_root
    )
    configure_process(profile, args.cache_root / "preflight" / "manifest.json")
    reranker_batch_size = args.reranker_batch_size or profile.reranker_batch_size
    if args.stage == "status":
        return show_status(args.cache_root)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        preflight_dir = args.cache_root / "preflight"
        with stage_run(preflight_dir, "preflight"):
            result = preflight(args.v3_dir, args.audit)
            result["execution"] = profile_metadata(profile)
            atomic_json(preflight_dir / "manifest.json", result)
    elif args.stage == "build-chunk-lookup":
        require_success(args.cache_root / "preflight")
        build_chunk_offset_index(args.v3_dir, args.cache_root / "chunk_lookup")
    elif args.stage == "tokenize-bm25":
        require_success(args.cache_root / "preflight")
        tokenize_v3_fields(
            args.v3_dir,
            args.cache_root / "bm25_fields",
            resume=args.resume,
            workers=args.workers if args.execution_profile == "optimized" else 1,
        )
    elif args.stage == "build-bm25":
        fields_dir = completed_stage(args.cache_root, args.upstream_cache_root, "bm25_fields")
        require_success(fields_dir)
        build_fts5_index(
            fields_dir, args.cache_root / "bm25_index", resume=args.resume
        )
    elif args.stage == "encode-bge-leaves":
        require_success(args.cache_root / "preflight")
        encode_v3_leaves(
            args.v3_dir,
            args.cache_root / "bge_leaves",
            model_name=args.model,
            device=args.device,
            resume=args.resume,
            batch_size=profile.bge_batch_size,
        )
    elif args.stage == "build-bge-blocks":
        leaf_dir = completed_stage(args.cache_root, args.upstream_cache_root, "bge_leaves")
        require_success(leaf_dir)
        build_structural_blocks(leaf_dir, args.cache_root / "bge_blocks")
    elif args.stage == "retrieve-hybrid":
        require_success(completed_stage(args.cache_root, args.upstream_cache_root, "bm25_index"))
        require_success(completed_stage(args.cache_root, args.upstream_cache_root, "bge_leaves"))
        require_success(completed_stage(args.cache_root, args.upstream_cache_root, "bge_blocks"))
        retrieve_hybrid(
            split=args.split,
            cache_root=args.cache_root,
            v3_dir=args.v3_dir,
            model_name=args.model,
            device=args.device,
            workers=args.workers,
            resume=args.resume,
            leaf_backend=profile.leaf_backend,
            upstream_cache_root=args.upstream_cache_root,
        )
    elif args.stage == "route-evidence":
        require_success(args.cache_root / "rankings" / args.split)
        # The optimized namespace intentionally reuses the immutable reference
        # byte-offset index unless a fast replacement was explicitly built.
        # Keep this gate consistent with route_evidence()'s fallback resolver.
        require_success(
            completed_stage(args.cache_root, args.upstream_cache_root, "chunk_lookup")
        )
        route_evidence(
            args.split,
            args.cache_root,
            args.v3_dir,
            resume=args.resume,
            batch_queries=128 if args.execution_profile == "optimized" else 32,
            persistent_reader=args.execution_profile == "optimized",
            upstream_cache_root=args.upstream_cache_root,
        )
    elif args.stage == "build-token-cache":
        evidence_dir = completed_stage(
            args.cache_root, args.upstream_cache_root, Path("evidence") / args.split
        )
        require_success(evidence_dir)
        build_token_cache(
            evidence_dir / "evidence.jsonl",
            args.cache_root / "token_cache" / args.split,
            resume=args.resume,
        )
    elif args.stage == "score-zero-shot":
        evidence_dir = completed_stage(
            args.cache_root, args.upstream_cache_root, Path("evidence") / args.split
        )
        require_success(evidence_dir)
        score_evidence_records(
            evidence_dir / "evidence.jsonl",
            args.cache_root / "zero_shot" / args.split,
            device=args.device,
            batch_size=reranker_batch_size,
            resume=args.resume,
            execution_metadata=profile_metadata(profile),
            token_cache_dir=(
                args.cache_root / "token_cache" / args.split
                if args.execution_profile == "optimized"
                and (args.cache_root / "token_cache" / args.split / "manifest.json").exists()
                else None
            ),
        )
    elif args.stage == "evaluate-zero-shot":
        require_success(args.cache_root / "zero_shot" / "train")
        ranking_manifest = json.loads(
            (args.cache_root / "rankings" / "train" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        evaluation_dir = args.results_root / "zero_shot_evaluation"
        with stage_run(
            evaluation_dir,
            "evaluate-zero-shot",
            total=7000,
            v3_fingerprint=ranking_manifest["v3_fingerprint"],
        ) as logger:
            result = evaluate_zero_shot_artifacts(
                args.cache_root / "evidence" / "train" / "evidence.jsonl",
                args.cache_root / "zero_shot" / "train" / "scores.jsonl",
                DEFAULT_TRAIN,
                ROOT / "cache" / "cv_folds.json",
                args.results_root / "zero_shot_metrics.json",
                stage1_metrics=ranking_manifest["metrics"]["hybrid"],
            )
            metrics = result["metrics"]
            gate = json.loads(
                (args.results_root / "zero_shot_metrics.json").read_text(encoding="utf-8")
            )["promotion_gate"]
            logger.log(
                f"Recall@5={metrics['recall@5']:.6f} "
                f"Precision@5={metrics['precision@5']:.6f} "
                f"promotion_gate={'PASS' if gate['overall_pass'] else 'FAIL'}"
            )
    elif args.stage in {"prepare-lora-teacher", "mine-lora"}:
        ranking_dir = completed_stage(
            args.cache_root, args.upstream_cache_root, Path("rankings") / "train"
        )
        leaf_dir = completed_stage(args.cache_root, args.upstream_cache_root, "bge_leaves")
        bm25_dir = completed_stage(args.cache_root, args.upstream_cache_root, "bm25_index")
        require_success(ranking_dir)
        require_success(leaf_dir)
        require_success(bm25_dir)
        require_success(args.results_root / "zero_shot_evaluation")
        zero_shot_report = json.loads(
            (args.results_root / "zero_shot_metrics.json").read_text(encoding="utf-8")
        )
        if not zero_shot_report.get("promotion_gate", {}).get("overall_pass"):
            raise RuntimeError("LoRA mining blocked: zero-shot promotion gate did not pass")
        if args.stage == "mine-lora" and args.fold > 0:
            gate_path = args.results_root / "lora" / "fold_0_metrics.json"
            if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get(
                "continuation_gate"
            ):
                raise RuntimeError("Fold 1-4 blocked until Fold 0 continuation gate passes")
        mine_fold_pair_cache(
            fold=-1 if args.stage == "prepare-lora-teacher" else args.fold,
            train_path=DEFAULT_TRAIN,
            folds_path=ROOT / "cache" / "cv_folds.json",
            source_rankings_path=ranking_dir / "source_rankings.jsonl",
            query_rows_path=ranking_dir / "query_rows.jsonl",
            query_vectors_path=ranking_dir / "query_embeddings.f16.npy",
            leaf_dir=leaf_dir,
            bm25_database=bm25_dir / "bm25_v3.sqlite",
            v3_dir=args.v3_dir,
            output_dir=(
                args.cache_root / "lora_pairs" / "teacher_prepare"
                if args.stage == "prepare-lora-teacher"
                else args.cache_root / "lora_pairs" / f"fold_{args.fold}"
            ),
            device=args.device,
            resume=args.resume,
            workers=args.workers,
        )
    elif args.stage == "train-lora-fold":
        pair_dir = args.cache_root / "lora_pairs" / f"fold_{args.fold}"
        require_success(pair_dir)
        train_pairs = args.pairs_train or pair_dir / "train_pairs.jsonl"
        validation_pairs = args.pairs_validation or pair_dir / "validation_pairs.jsonl"
        train_lora_pairwise(
            train_pairs,
            validation_pairs,
            args.cache_root / "lora" / f"fold_{args.fold}",
            device=args.device,
            resume=args.resume,
            stage_name="train-lora-fold",
            pair_batch_size=args.pair_batch_size,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            pretokenize=profile.pretokenize_training,
        )
    elif args.stage == "score-lora-fold":
        adapter_stage = completed_stage(
            args.cache_root, args.upstream_cache_root, Path("lora") / f"fold_{args.fold}"
        )
        require_success(adapter_stage)
        score_lora_fold(
            fold=args.fold,
            evidence_path=completed_stage(
                args.cache_root, args.upstream_cache_root, Path("evidence") / "train"
            ) / "evidence.jsonl",
            folds_path=ROOT / "cache" / "cv_folds.json",
            adapter_dir=adapter_stage / "best_adapter",
            output_dir=args.cache_root / "lora_scores" / f"fold_{args.fold}",
            device=args.device,
            resume=args.resume,
            merge_adapter=profile.merge_lora,
            batch_size=reranker_batch_size,
            token_cache_dir=(
                args.cache_root / "token_cache" / "train"
                if args.execution_profile == "optimized"
                and (args.cache_root / "token_cache" / "train" / "manifest.json").exists()
                else None
            ),
        )
    elif args.stage == "evaluate-lora-fold":
        require_success(args.cache_root / "lora_scores" / f"fold_{args.fold}")
        result_dir = args.results_root / "lora"
        result_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir = result_dir / f"fold_{args.fold}_evaluation"
        with stage_run(evaluation_dir, "evaluate-lora-fold", total=1) as logger:
            result = evaluate_lora_fold(
                fold=args.fold,
                candidates_path=args.cache_root / "rankings" / "train" / "hybrid_candidates.jsonl",
                scores_path=args.cache_root / "lora_scores" / f"fold_{args.fold}" / "scores.jsonl",
                train_path=DEFAULT_TRAIN,
                folds_path=ROOT / "cache" / "cv_folds.json",
                zero_shot_metrics_path=args.results_root / "zero_shot_metrics.json",
                output_path=result_dir / f"fold_{args.fold}_metrics.json",
                logger=logger,
            )
            logger.log(
                f"Recall@5={result['metrics']['recall@5']:.6f} "
                f"Precision@5={result['metrics']['precision@5']:.6f} "
                f"continuation_gate={'PASS' if result['continuation_gate'] else 'FAIL'}"
            )
    elif args.stage == "evaluate-oof":
        result_dir = args.results_root / "lora"
        for fold in range(5):
            require_success(result_dir / f"fold_{fold}_evaluation")
            require_success(args.cache_root / "lora_scores" / f"fold_{fold}")
        evaluation_dir = result_dir / "oof_evaluation"
        with stage_run(evaluation_dir, "evaluate-oof", total=1) as logger:
            result = evaluate_oof(
                [result_dir / f"fold_{fold}_metrics.json" for fold in range(5)],
                DEFAULT_TRAIN,
                args.results_root / "oof_metrics.json",
                candidates_path=args.cache_root / "rankings" / "train" / "hybrid_candidates.jsonl",
                fold_score_paths=[
                    args.cache_root / "lora_scores" / f"fold_{fold}" / "scores.jsonl"
                    for fold in range(5)
                ],
                folds_path=ROOT / "cache" / "cv_folds.json",
                zero_shot_metrics_path=args.results_root / "zero_shot_metrics.json",
                logger=logger,
            )
            logger.log(
                f"Recall@5={result['metrics']['recall@5']:.6f} "
                f"Precision@5={result['metrics']['precision@5']:.6f} "
                f"promotion_gate={'PASS' if result['promotion_gate']['overall_pass'] else 'FAIL'}"
            )
    elif args.stage == "mine-final":
        require_success(args.results_root / "lora" / "oof_evaluation")
        oof = json.loads((args.results_root / "oof_metrics.json").read_text(encoding="utf-8"))
        gate = oof.get("promotion_gate", {})
        if not (gate.get("overall_pass") if isinstance(gate, dict) else gate):
            raise RuntimeError("Final mining blocked: OOF promotion gate did not pass")
        mine_fold_pair_cache(
            fold=-1,
            train_path=DEFAULT_TRAIN,
            folds_path=ROOT / "cache" / "cv_folds.json",
            source_rankings_path=args.cache_root / "rankings" / "train" / "source_rankings.jsonl",
            query_rows_path=args.cache_root / "rankings" / "train" / "query_rows.jsonl",
            query_vectors_path=args.cache_root / "rankings" / "train" / "query_embeddings.f16.npy",
            leaf_dir=args.cache_root / "bge_leaves",
            bm25_database=args.cache_root / "bm25_index" / "bm25_v3.sqlite",
            v3_dir=args.v3_dir,
            output_dir=args.cache_root / "lora_pairs" / "final",
            device=args.device,
            resume=args.resume,
            workers=args.workers,
        )
    elif args.stage == "train-final":
        require_success(args.cache_root / "lora_pairs" / "final")
        for fold in range(5):
            require_success(args.cache_root / "lora" / f"fold_{fold}")
        best_epochs = [
            json.loads((args.cache_root / "lora" / f"fold_{fold}" / "manifest.json").read_text(encoding="utf-8"))[
                "best_epoch"
            ]
            for fold in range(5)
        ]
        epoch = sorted(set(best_epochs), key=lambda value: (-best_epochs.count(value), value))[0]
        train_lora_pairwise(
            args.cache_root / "lora_pairs" / "final" / "train_pairs.jsonl",
            args.cache_root / "lora_pairs" / "final" / "validation_pairs.jsonl",
            args.cache_root / "lora" / "final",
            device=args.device,
            epochs=epoch,
            train_on_validation=True,
            resume=args.resume,
            stage_name="train-final",
            pair_batch_size=args.pair_batch_size,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            pretokenize=profile.pretokenize_training,
        )
    elif args.stage == "score-final-public":
        final_adapter_stage = completed_stage(
            args.cache_root, args.upstream_cache_root, Path("lora") / "final"
        )
        require_success(final_adapter_stage)
        public_evidence = completed_stage(
            args.cache_root, args.upstream_cache_root, Path("evidence") / "public"
        )
        require_success(public_evidence)
        adapter_dir = final_adapter_stage / "best_adapter"
        score_evidence_records(
            public_evidence / "evidence.jsonl",
            args.cache_root / "lora_scores" / "public",
            model_name=str(adapter_dir),
            device=args.device,
            batch_size=reranker_batch_size,
            resume=args.resume,
            stage_name="score-final-public",
            model_loader=lambda: load_adapter(
                adapter_dir, device=args.device, merge=profile.merge_lora
            ),
            execution_metadata=profile_metadata(profile),
            token_cache_dir=(
                args.cache_root / "token_cache" / "public"
                if args.execution_profile == "optimized"
                and (args.cache_root / "token_cache" / "public" / "manifest.json").exists()
                else None
            ),
        )
    elif args.stage == "public-submission":
        require_success(args.cache_root / "lora_scores" / "public")
        require_success(args.results_root / "lora" / "oof_evaluation")
        require_success(args.cache_root / "rankings" / "public")
        submission_dir = args.results_root / "submission"
        with stage_run(submission_dir, "public-submission", total=1) as logger:
            _, zip_path = build_public_submission(
                candidates_path=args.cache_root / "rankings" / "public" / "hybrid_candidates.jsonl",
                scores_path=args.cache_root / "lora_scores" / "public" / "scores.jsonl",
                zero_shot_metrics_path=args.results_root / "zero_shot_metrics.json",
                oof_report_path=args.results_root / "oof_metrics.json",
                output_dir=submission_dir,
            )
            logger.log(f"submission={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
