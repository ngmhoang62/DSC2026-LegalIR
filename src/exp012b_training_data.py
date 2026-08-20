"""Fold-isolated gold routing and BM25/BGE/hybrid hard-negative mining."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from exp012b_bm25 import BM25Searcher, default_segmenter, safe_fts_query
from exp012b_core import atomic_json, canonical_json, content_hash, load_answers, load_queries, read_jsonl, sha256_file, stage_run
from exp012b_dense import load_embedding_memmap, top_indices
from exp012b_retrieval import load_chunk_records, render_evidence_bundle, select_evidence_ids, weighted_rrf
from exp012b_reranker import _load_reranker, _score_pairs_with_stats, mine_pairwise_examples, unload_cuda
from exp012b_tuning import load_folds


def deterministic_internal_split(
    qids: Sequence[str],
    answers: Mapping[str, set[str]],
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    by_cardinality: dict[int, list[str]] = defaultdict(list)
    for qid in qids:
        by_cardinality[len(answers[qid])].append(qid)
    train: list[str] = []
    validation: list[str] = []
    for cardinality, members in sorted(by_cardinality.items()):
        ordered = sorted(
            members,
            key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(),
        )
        validation_count = max(1, round(len(ordered) * validation_fraction)) if len(ordered) > 1 else 0
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    return sorted(train), sorted(validation)


def choose_channel_negatives(
    rankings: Mapping[str, Sequence[dict[str, Any]]], answers: set[str]
) -> list[tuple[str, str]]:
    bm25_ids = [str(row["doc_id"]) for row in rankings["bm25"]]
    bge_ids = [str(row["doc_id"]) for row in rankings["bge_leaf"]]
    bge_set, bm25_set = set(bge_ids), set(bm25_ids)

    def first(candidates: Sequence[str], excluded_channel: set[str] | None = None) -> str | None:
        for doc_id in candidates:
            if doc_id not in answers and (excluded_channel is None or doc_id not in excluded_channel):
                return doc_id
        for doc_id in candidates:
            if doc_id not in answers:
                return doc_id
        return None

    hybrid = [row["doc_id"] for row in weighted_rrf(rankings, limit=50)]
    values = [
        ("bm25", first(bm25_ids, bge_set)),
        ("bge", first(bge_ids, bm25_set)),
        ("hybrid", first(hybrid)),
    ]
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for source, doc_id in values:
        if doc_id is not None and doc_id not in seen:
            output.append((source, doc_id))
            seen.add(doc_id)
    return output


def mine_fold_pair_cache(
    *,
    fold: int,
    train_path: Path,
    folds_path: Path,
    source_rankings_path: Path,
    query_rows_path: Path,
    query_vectors_path: Path,
    leaf_dir: Path,
    bm25_database: Path,
    v3_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    model_name: str = "BAAI/bge-reranker-v2-m3",
    resume: bool = False,
    workers: int = 4,
    teacher_shard_queries: int = 64,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if teacher_shard_queries < 1:
        raise ValueError("teacher_shard_queries must be positive")
    answers = load_answers(train_path)
    queries = load_queries(train_path)
    folds = load_folds(folds_path)
    fold_name = "final" if fold < 0 else f"fold_{fold}"
    if fold >= 0 and fold_name not in folds:
        raise ValueError(f"Unknown fold {fold_name}")
    heldout = set() if fold < 0 else set(folds[fold_name])
    eligible = sorted(set(answers) - heldout)
    train_qids, validation_qids = deterministic_internal_split(eligible, answers)
    split_by_qid = {qid: "train" for qid in train_qids} | {qid: "validation" for qid in validation_qids}
    ranking_manifest = json.loads((source_rankings_path.parent / "manifest.json").read_text(encoding="utf-8"))
    bm25_manifest = json.loads((bm25_database.parent / "manifest.json").read_text(encoding="utf-8"))
    leaf_manifest = json.loads((leaf_dir / "manifest.json").read_text(encoding="utf-8"))
    route_key = content_hash(
        {
            "route_schema": "document-bounded-bm25-v2",
            "leaf": leaf_manifest["content_fingerprint"],
            "rankings": ranking_manifest.get("content_fingerprint"),
            "bm25": bm25_manifest["content_fingerprint"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir.parents[1]
    mining_dir = cache_root / "lora_mining"
    route_dir = cache_root / "lora_routes" / route_key
    teacher_dir = cache_root / "lora_teacher"
    for directory in (mining_dir, route_dir, teacher_dir):
        directory.mkdir(parents=True, exist_ok=True)
    teacher_key = content_hash({"route_key": route_key, "model": model_name, "scoring": "qid-shards-v1"})

    with stage_run(output_dir, "mine-lora", total=len(eligible)) as logger:
        # Build the small hard-negative cache once.  Future folds avoid parsing
        # the 1.13GB source ranking artifact again.
        negative_key = content_hash(
            {
                "schema": "channel-negatives-v1",
                "rankings": ranking_manifest.get("content_fingerprint"),
                "train": sha256_file(train_path),
            }
        )
        negative_path = mining_dir / "negative_sources.jsonl"
        negative_manifest_path = mining_dir / "negative_sources.manifest.json"
        negative_cache_valid = False
        if negative_path.exists() and negative_manifest_path.exists():
            marker = json.loads(negative_manifest_path.read_text(encoding="utf-8"))
            negative_cache_valid = (
                marker.get("cache_key") == negative_key
                and marker.get("sha256") == sha256_file(negative_path)
            )
        if not negative_cache_valid:
            logger.log("phase=select-negatives source=source_rankings.jsonl")
            temporary = negative_path.with_suffix(".jsonl.tmp")
            selected_count = 0
            with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
                for row_number, row in enumerate(read_jsonl(source_rankings_path), start=1):
                    qid = str(row["qid"])
                    if qid not in answers:
                        continue
                    negatives = choose_channel_negatives(row["rankings"], answers[qid])
                    handle.write(canonical_json({"qid": qid, "negatives": negatives}) + "\n")
                    selected_count += 1
                    if row_number % 256 == 0:
                        logger.status(
                            stage="mine-lora", state="RUNNING", phase="select-negatives",
                            completed=row_number, total=len(answers),
                        )
            temporary.replace(negative_path)
            if selected_count != len(answers):
                raise RuntimeError(f"Negative cache coverage mismatch: {selected_count}/{len(answers)}")
            atomic_json(
                negative_manifest_path,
                {"cache_key": negative_key, "sha256": sha256_file(negative_path), "queries": selected_count},
            )
        negative_sources = {
            str(row["qid"]): [(str(source), str(doc_id)) for source, doc_id in row["negatives"]]
            for row in read_jsonl(negative_path)
            if str(row["qid"]) in split_by_qid
        }
        missing_rankings = sorted(set(eligible) - negative_sources.keys())
        if missing_rankings:
            raise ValueError(f"Missing Stage-1 rankings for qids: {missing_rankings[:10]}")

        def target_documents(qid: str) -> list[str]:
            return sorted(answers[qid] | {doc_id for _, doc_id in negative_sources[qid]})

        def load_valid_route(qid: str) -> dict[str, Any] | None:
            path = route_dir / f"{qid}.json"
            if not path.exists():
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = target_documents(qid)
            if (
                record.get("route_key") == route_key
                and record.get("target_docs") == expected
                and set(record.get("selections", {})) == set(expected)
            ):
                return record
            return None

        pending_routes = [qid for qid in eligible if load_valid_route(qid) is None]
        cached_routes = len(eligible) - len(pending_routes)
        if pending_routes:
            logger.log("phase=prepare-routing loading query/leaf indexes and BM25 document bounds")
            query_index = {row["qid"]: int(row["row"]) for row in read_jsonl(query_rows_path)}
            query_vectors = np.load(query_vectors_path, mmap_mode="r")
            leaf_vectors, leaf_rows, loaded_leaf_manifest = load_embedding_memmap(leaf_dir, "leaf")
            if loaded_leaf_manifest.get("content_fingerprint") != leaf_manifest.get("content_fingerprint"):
                raise RuntimeError("Leaf manifest changed while mining")
            document_rows: dict[str, list[int]] = defaultdict(list)
            for row in leaf_rows:
                document_rows[str(row["doc_id"])].append(int(row["row"]))
            with BM25Searcher(bm25_database) as range_searcher:
                document_ranges = range_searcher.load_document_ranges()
            if set(document_rows) != set(document_ranges):
                raise RuntimeError("Dense/BM25 document coverage mismatch")
            logger.log(
                f"phase=segment-queries pending={len(pending_routes)} cached={cached_routes} workers={workers}"
            )
            expressions: dict[str, str] = {}
            segment_started = time.perf_counter()
            for number, qid in enumerate(pending_routes, start=1):
                expressions[qid] = safe_fts_query(default_segmenter(queries[qid]))
                if number % 128 == 0:
                    elapsed = time.perf_counter() - segment_started
                    logger.status(
                        stage="mine-lora", state="RUNNING", phase="segment-queries",
                        completed=number, total=len(pending_routes),
                        queries_per_second=number / elapsed if elapsed else 0.0,
                    )

            thread_state = threading.local()
            created_searchers: list[BM25Searcher] = []
            searcher_lock = threading.Lock()

            def worker_searcher() -> BM25Searcher:
                searcher = getattr(thread_state, "searcher", None)
                if searcher is None:
                    searcher = BM25Searcher(bm25_database, document_ranges=document_ranges)
                    thread_state.searcher = searcher
                    with searcher_lock:
                        created_searchers.append(searcher)
                return searcher

            def route_query(qid: str) -> dict[str, Any]:
                query_vector = np.asarray(query_vectors[query_index[qid]], dtype=np.float32)
                selections_by_doc: dict[str, list[dict[str, Any]]] = {}
                searcher = worker_searcher()
                for doc_id in target_documents(qid):
                    indices = document_rows.get(doc_id, [])
                    if not indices:
                        raise ValueError(f"No v3 leaves for {qid}/{doc_id}")
                    scores = np.asarray(leaf_vectors[indices], dtype=np.float32) @ query_vector
                    dense_hits = []
                    for rank, local in enumerate(top_indices(scores, min(24, len(indices))), start=1):
                        row = leaf_rows[indices[int(local)]]
                        dense_hits.append({**row, "score": float(scores[local]), "rank": rank})
                    lexical_hits = searcher.search_document_expression(
                        expressions[qid], doc_id, limit=8
                    )
                    selections_by_doc[doc_id] = select_evidence_ids(
                        dense_hits, lexical_hits, maximum=3
                    )
                return {
                    "schema_version": "legalir.exp012b_v3.route.v2",
                    "route_key": route_key,
                    "qid": qid,
                    "target_docs": target_documents(qid),
                    "selections": selections_by_doc,
                }

            logger.log("phase=route-documents exact_bm25=rowid-bounded")
            route_started = time.perf_counter()
            completed_routes = cached_routes
            try:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lora-route") as executor:
                    futures = {executor.submit(route_query, qid): qid for qid in pending_routes}
                    for future in as_completed(futures):
                        qid = futures[future]
                        atomic_json(route_dir / f"{qid}.json", future.result())
                        completed_routes += 1
                        routed_now = completed_routes - cached_routes
                        if routed_now % 32 == 0 or completed_routes == len(eligible):
                            elapsed = time.perf_counter() - route_started
                            rate = routed_now / elapsed if elapsed else 0.0
                            logger.status(
                                stage="mine-lora", state="RUNNING", phase="route-documents",
                                completed=completed_routes, total=len(eligible),
                                cached=cached_routes, queries_per_second=rate,
                                eta_seconds=(len(eligible) - completed_routes) / rate if rate else None,
                            )
                        if routed_now % 256 == 0:
                            logger.log(
                                f"phase=route-documents completed={completed_routes}/{len(eligible)} "
                                f"queries_per_second={routed_now / max(time.perf_counter()-route_started, 1e-9):.2f}"
                            )
            finally:
                for searcher in created_searchers:
                    searcher.close()
        else:
            logger.log(f"phase=route-documents cache_hit={cached_routes}/{len(eligible)}")

        def load_valid_teacher(qid: str) -> dict[str, Any] | None:
            path = teacher_dir / f"{qid}.json"
            if not path.exists():
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = set(target_documents(qid))
            if record.get("teacher_key") == teacher_key and set(record.get("bundles", {})) >= expected:
                return record
            return None

        pending_teacher = [qid for qid in eligible if load_valid_teacher(qid) is None]
        cached_teacher_count = len(eligible) - len(pending_teacher)
        actual_batch = 24
        lookup_database = cache_root / "chunk_lookup" / "chunk_offsets.sqlite"
        documents = {row["doc_id"]: row for row in read_jsonl(v3_dir / "documents.jsonl")}
        if pending_teacher:
            logger.log(
                f"phase=teacher-load-model pending={len(pending_teacher)} cached={cached_teacher_count}"
            )
            model, tokenizer = _load_reranker(model_name, device)
            logger.log("phase=teacher-score model_loaded=true")
            teacher_started = time.perf_counter()
            teacher_pairs_scored = 0
            try:
                for start in range(0, len(pending_teacher), teacher_shard_queries):
                    shard_qids = pending_teacher[start : start + teacher_shard_queries]
                    routes = {
                        qid: json.loads((route_dir / f"{qid}.json").read_text(encoding="utf-8"))
                        for qid in shard_qids
                    }
                    needed_chunks = {
                        selection["chunk_id"]
                        for route in routes.values()
                        for selections in route["selections"].values()
                        for selection in selections
                    }
                    chunks = load_chunk_records(
                        v3_dir, needed_chunks,
                        lookup_database=lookup_database if lookup_database.exists() else None,
                    )
                    identities: list[tuple[str, str, dict[str, Any], str]] = []
                    pairs: list[tuple[str, str]] = []
                    for qid in shard_qids:
                        for doc_id in routes[qid]["target_docs"]:
                            for selection in routes[qid]["selections"][doc_id]:
                                bundle = render_evidence_bundle(
                                    queries[qid], documents[doc_id]["document_label"],
                                    chunks[selection["chunk_id"]],
                                )
                                identities.append((qid, doc_id, selection, bundle))
                                pairs.append((queries[qid], bundle))
                    logits, actual_batch, score_stats = _score_pairs_with_stats(
                        model, tokenizer, pairs, device=device, batch_size=actual_batch
                    )
                    best: dict[tuple[str, str], dict[str, Any]] = {}
                    for (qid, doc_id, selection, bundle), score in zip(identities, logits):
                        key = (qid, doc_id)
                        current = best.get(key)
                        if current is None or float(score) > current["score"]:
                            best[key] = {
                                "chunk_id": selection["chunk_id"],
                                "bundle_text": bundle,
                                "score": float(score),
                            }
                    for qid in shard_qids:
                        atomic_json(
                            teacher_dir / f"{qid}.json",
                            {
                                "schema_version": "legalir.exp012b_v3.teacher.v2",
                                "teacher_key": teacher_key,
                                "qid": qid,
                                "bundles": {
                                    doc_id: best[(qid, doc_id)] for doc_id in target_documents(qid)
                                },
                            },
                        )
                    teacher_pairs_scored += len(pairs)
                    completed_teacher = cached_teacher_count + start + len(shard_qids)
                    elapsed = time.perf_counter() - teacher_started
                    logger.status(
                        stage="mine-lora", state="RUNNING", phase="teacher-score",
                        completed=completed_teacher, total=len(eligible),
                        cached=cached_teacher_count, pairs_scored=teacher_pairs_scored,
                        pairs_per_second=teacher_pairs_scored / elapsed if elapsed else 0.0,
                        actual_batch_size=actual_batch,
                    )
                    logger.log(
                        f"phase=teacher-score completed={completed_teacher}/{len(eligible)} "
                        f"shard_queries={len(shard_qids)} pairs={len(pairs)} "
                        f"pairs_per_second={score_stats['pairs_per_second']:.2f}"
                    )
            finally:
                unload_cuda(model, tokenizer)
        else:
            logger.log(f"phase=teacher-score cache_hit={cached_teacher_count}/{len(eligible)}")

        logger.log("phase=write-pairs")
        paths = {name: output_dir / f"{name}_pairs.jsonl" for name in ("train", "validation")}
        temporary_paths = {name: path.with_suffix(".jsonl.tmp") for name, path in paths.items()}
        handles = {
            name: path.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024)
            for name, path in temporary_paths.items()
        }
        counts = defaultdict(int)
        try:
            for qid in eligible:
                teacher = load_valid_teacher(qid)
                if teacher is None:
                    raise RuntimeError(f"Missing valid teacher cache for {qid}")
                bundles = teacher["bundles"]
                positive = {doc_id: bundles[doc_id] for doc_id in answers[qid]}
                negative_bundle = {
                    doc_id: bundles[doc_id] for _, doc_id in negative_sources[qid]
                }
                # Selection already happened against full rankings above. The
                # mining helper only needs the selected doc and its provenance.
                selected_by_source = dict(negative_sources[qid])
                channel_rankings = {
                    source: ([{"doc_id": selected_by_source[source]}] if source in selected_by_source else [])
                    for source in ("bm25", "bge", "hybrid")
                }
                rows = mine_pairwise_examples(
                    qid=qid,
                    query=queries[qid],
                    answers=answers[qid],
                    positive_bundles=positive,
                    channel_rankings=channel_rankings,
                    best_bundle_by_doc=negative_bundle,
                )
                destination = split_by_qid[qid]
                for row in rows:
                    handles[destination].write(canonical_json(row) + "\n")
                    counts[destination] += 1
        finally:
            for handle in handles.values():
                handle.close()
        for name, temporary in temporary_paths.items():
            temporary.replace(paths[name])
        manifest = {
            "schema_version": "legalir.exp012b_v3.lora_pairs.v2",
            "fold": fold,
            "heldout_qids": len(heldout),
            "train_qids": len(train_qids),
            "validation_qids": len(validation_qids),
            "counts": dict(counts),
            "leaf_fingerprint": leaf_manifest["content_fingerprint"],
            "route_key": route_key,
            "teacher_key": teacher_key,
            "workers": workers,
            "resume_requested": resume,
            "actual_reranker_batch_size": actual_batch,
            "pair_fingerprint": content_hash({name: sha256_file(path) for name, path in paths.items()}),
        }
        atomic_json(output_dir / "manifest.json", manifest)
    return manifest
