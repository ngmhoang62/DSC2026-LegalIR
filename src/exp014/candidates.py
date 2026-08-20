"""Candidate generation and integrity audit for EXP-014."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from exp012b_core import (
    atomic_json,
    load_answers,
    load_queries,
    read_jsonl,
    sha256_file,
    stage_run,
    write_jsonl,
)
from exp012b_tuning import load_folds
from exp014.core import SCHEMA, hash_payload, load_document_metadata, oracle, write_manifest
from exp014.entity_matcher import match_document_entities


def _ranking_rows(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        if qid in output:
            raise ValueError(f"duplicate qid in ranking source: {qid}")
        output[qid] = dict(row)
    return output


def build_query_memory(*, split: str, train_path: Path, query_path: Path, folds_path: Path, output_dir: Path,
                       v3_fingerprint: str, neighbors: int = 10, documents: int = 10) -> dict[str, Any]:
    """Character TF-IDF retrieval with label access blocked for a held-out fold."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_queries, answers = load_queries(train_path), load_answers(train_path)
    targets = load_queries(query_path)
    target_ids, train_ids = sorted(targets), sorted(train_queries)
    folds = load_folds(folds_path) if split == "train" else {}
    fold_for = {str(qid): fold for fold, qids in folds.items() for qid in qids}
    if split == "train" and set(target_ids) != set(train_ids):
        raise ValueError("train query memory must target exactly the labeled train set")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=120000,
                                 dtype=np.float32, sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform([train_queries[qid] for qid in train_ids])
    target = matrix if split == "train" else vectorizer.transform([targets[qid] for qid in target_ids])
    train_index = {qid: position for position, qid in enumerate(train_ids)}
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{split}_memory.jsonl"
    with stage_run(output_dir, "build-query-memory", total=len(target_ids), v3_fingerprint=v3_fingerprint) as logger:
        started = time.perf_counter()
        for position, qid in enumerate(target_ids, 1):
            similarities = (target[position - 1] @ matrix.T).toarray().ravel()
            blocked: set[str] = set()
            if split == "train":
                fold = fold_for.get(qid)
                if fold is None:
                    raise ValueError(f"missing fold for qid={qid}")
                blocked = {str(value) for value in folds[fold]}
                for blocked_qid in blocked:
                    similarities[train_index[blocked_qid]] = -np.inf
            order = sorted(range(len(train_ids)), key=lambda index: (-float(similarities[index]), train_ids[index]))[:neighbors]
            scores: dict[str, float] = defaultdict(float)
            provenance: dict[str, list[str]] = defaultdict(list)
            for rank, index in enumerate(order, 1):
                if not np.isfinite(similarities[index]):
                    continue
                neighbor = train_ids[index]
                if neighbor in blocked:
                    raise RuntimeError(f"fold leakage at qid={qid}, neighbor={neighbor}")
                for doc_id in sorted(answers[neighbor]):
                    scores[doc_id] += float(similarities[index]) / rank
                    provenance[doc_id].append(neighbor)
            ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:documents]
            rows.append({"qid": qid, "rankings": [
                {"doc_id": doc_id, "rank": rank, "score": float(scores[doc_id]), "neighbor_qids": provenance[doc_id]}
                for rank, doc_id in enumerate(ranked, 1)
            ]})
            if position % 64 == 0 or position == len(target_ids):
                logger.status(stage="build-query-memory", state="RUNNING", completed=position, total=len(target_ids))
            if position % 256 == 0 or position == len(target_ids):
                rate = position / max(time.perf_counter() - started, 1e-9)
                logger.log(f"progress={position}/{len(target_ids)} rate={rate:.2f}_queries_per_second eta_seconds={(len(target_ids)-position)/max(rate,1e-9):.0f}")
        write_jsonl(result_path, rows)
        return write_manifest(output_dir, stage="build-query-memory", v3_fingerprint=v3_fingerprint,
            config={"split": split, "neighbors": neighbors, "documents": documents, "fold_isolated": split == "train"},
            files=[result_path], counts={"queries": len(rows), "vocabulary": len(vectorizer.vocabulary_)})


def _source_value(row: Mapping[str, Any], name: str, limit: int) -> dict[str, dict[str, Any]]:
    return {str(value["doc_id"]): dict(value) for value in row.get("rankings", {}).get(name, [])[:limit]}


def build_candidates(*, split: str, rankings_path: Path, memory_path: Path, v3_dir: Path, output_dir: Path,
                     v3_fingerprint: str, bge_limit: int = 120, bm25_limit: int = 50, memory_limit: int = 10) -> dict[str, Any]:
    ranking_manifest = json.loads((rankings_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if ranking_manifest.get("v3_fingerprint") != v3_fingerprint:
        raise RuntimeError("source-ranking manifest has stale structural-v3 fingerprint")
    ranking_expected = ranking_manifest.get("artifact_sha256", {}).get(rankings_path.name)
    if not ranking_expected:
        raise RuntimeError("source-ranking manifest does not fingerprint source_rankings.jsonl")
    source, memory = _ranking_rows(rankings_path), _ranking_rows(memory_path)
    if set(source) != set(memory):
        raise ValueError("source ranking and memory query sets differ")
    metadata = load_document_metadata(v3_dir)
    ranking_fingerprint, memory_fingerprint = sha256_file(rankings_path), sha256_file(memory_path)
    if ranking_fingerprint != ranking_expected:
        raise RuntimeError("source_rankings.jsonl hash differs from frozen EXP-012b manifest")
    
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{split}_candidates.jsonl"
    with stage_run(output_dir, "build-candidates", total=len(source), v3_fingerprint=v3_fingerprint) as logger:
        started = time.perf_counter()
        for position, qid in enumerate(sorted(source), 1):
            source_row = source[qid]
            query_text = str(source_row["query"])
            if source_row.get("v3_fingerprint") != v3_fingerprint:
                raise RuntimeError(f"stale source rankings at qid={qid}")
            channels = {
                "bge": _source_value(source_row, "bge_leaf", bge_limit),
                "bm25": _source_value(source_row, "bm25", bm25_limit),
                "memory": {str(value["doc_id"]): dict(value) for value in memory[qid].get("rankings", [])[:memory_limit]},
            }
            candidate_ids = sorted(set().union(*(set(values) for values in channels.values())))
            candidates: list[dict[str, Any]] = []
            for doc_id in candidate_ids:
                if doc_id not in metadata:
                    raise ValueError(f"candidate absent from structural-v3: {doc_id}")
                item = metadata[doc_id]
                entity_info = match_document_entities(query_text, item)
                sources = {name: {
                    "rank": int(value["rank"]), "score": float(value.get("score", 0.0)),
                    "best_chunk_id": value.get("best_chunk_id"), "neighbor_qids": value.get("neighbor_qids", []),
                } for name, values in channels.items() if (value := values.get(doc_id)) is not None}
                best = min(item["rank"] for item in sources.values())
                
                candidates.append({
                    "doc_id": doc_id, "sources": sources, "channel_count": len(sources), "best_source_rank": best,
                    "bge_rank": int(sources.get("bge", {}).get("rank", 999)),
                    "bm25_rank": int(sources.get("bm25", {}).get("rank", 999)),
                    "memory_rank": int(sources.get("memory", {}).get("rank", 999)),
                    "entity_exact_match": float(entity_info["exact_match"]),
                    "entity_name_match": float(entity_info["name_match"]),
                    "entity_match_count": float(entity_info["match_count"]),
                    "chunk_count": int(item["chunk_count"]), "article_count": int(item["article_count"]),
                    "document_length": int(item.get("passage_length", 0)),
                    "parse_mode": str(item.get("parse_mode", "unknown")), "document_type": str(item["document_type"]),
                    "hierarchy_available": int(item["hierarchy_available"]),
                })
            
            # Deterministic ordering priority: Entity Exact Match > Channel Count > Best Rank > BGE Rank > BM25 Rank > Memory Rank > Doc ID
            candidates.sort(key=lambda item: (
                -item["entity_exact_match"],
                -item["channel_count"],
                item["best_source_rank"],
                item["bge_rank"],
                item["bm25_rank"],
                item["memory_rank"],
                item["doc_id"]
            ))
            for rank, item in enumerate(candidates, 1):
                item["rank"] = rank
            rows.append({
                "schema_version": SCHEMA, "v3_fingerprint": v3_fingerprint, "qid": qid,
                "query": query_text, "source_fingerprint": ranking_fingerprint,
                "memory_fingerprint": memory_fingerprint, "candidates": candidates
            })
            if position % 64 == 0 or position == len(source):
                logger.status(stage="build-candidates", state="RUNNING", completed=position, total=len(source))
            if position % 256 == 0 or position == len(source):
                rate = position / max(time.perf_counter() - started, 1e-9)
                logger.log(f"progress={position}/{len(source)} rate={rate:.2f}_queries_per_second eta_seconds={(len(source)-position)/max(rate,1e-9):.0f}")
        write_jsonl(result_path, rows)
        return write_manifest(output_dir, stage="build-candidates", v3_fingerprint=v3_fingerprint,
            config={"split": split, "sources": {"bge_leaf": bge_limit, "bm25": bm25_limit, "query_memory": memory_limit}}, files=[result_path],
            inputs={"source_rankings_sha256": ranking_fingerprint, "query_memory_sha256": memory_fingerprint},
            counts={"queries": len(rows), "candidates": sum(len(row["candidates"]) for row in rows)})


def audit_candidates(*, candidates_path: Path, memory_path: Path, train_path: Path, folds_path: Path,
                     v3_dir: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    records = list(read_jsonl(candidates_path))
    answers, folds = load_answers(train_path), load_folds(folds_path)
    fold_for = {str(qid): fold for fold, values in folds.items() for qid in values}
    v3_docs = {str(row["doc_id"]) for row in read_jsonl(v3_dir / "documents.jsonl")}
    memory = _ranking_rows(memory_path)
    errors: list[str] = []
    sizes: list[int] = []
    seen: set[str] = set()
    source_fingerprints: set[str] = set()
    memory_fingerprints: set[str] = set()
    for record in records:
        qid = str(record["qid"])
        if qid in seen:
            errors.append(f"duplicate_qid:{qid}")
        seen.add(qid)
        if record.get("v3_fingerprint") != v3_fingerprint:
            errors.append(f"stale:{qid}")
        source_fingerprints.add(str(record.get("source_fingerprint", "")))
        memory_fingerprints.add(str(record.get("memory_fingerprint", "")))
        docs = [str(row["doc_id"]) for row in record["candidates"]]
        sizes.append(len(docs))
        if len(docs) != len(set(docs)):
            errors.append(f"duplicate_document:{qid}")
        expected_order = sorted(record["candidates"], key=lambda item: (
            -float(item["entity_exact_match"]),
            -int(item["channel_count"]),
            int(item["best_source_rank"]),
            int(item["bge_rank"]),
            int(item["bm25_rank"]),
            int(item["memory_rank"]),
            str(item["doc_id"])
        ))
        if docs != [str(row["doc_id"]) for row in expected_order] or [int(row["rank"]) for row in record["candidates"]] != list(range(1, len(docs) + 1)):
            errors.append(f"nondeterministic_rank:{qid}")
        if set(docs) - v3_docs:
            errors.append(f"missing_v3_document:{qid}")
        if qid in fold_for:
            forbidden = {str(value) for value in folds[fold_for[qid]]}
            for row in memory.get(qid, {}).get("rankings", []):
                if forbidden & {str(value) for value in row.get("neighbor_qids", [])}:
                    errors.append(f"memory_leakage:{qid}")
    if set(seen) != set(answers):
        errors.append("query_set_mismatch")
    if len(source_fingerprints) != 1 or "" in source_fingerprints:
        errors.append("source_fingerprint_mismatch")
    if len(memory_fingerprints) != 1 or "" in memory_fingerprints:
        errors.append("memory_fingerprint_mismatch")
    missing_ground_truth = sorted({doc_id for values in answers.values() for doc_id in values} - v3_docs)
    if missing_ground_truth:
        errors.append("ground_truth_missing_v3:" + ",".join(missing_ground_truth[:5]))
    
    metrics = oracle(records, answers)
    target_recall = metrics.get("recall@180", metrics.get("recall@150", 0.0))
    if target_recall < 0.990:
        errors.append(f"pool_recall={target_recall:.6f}_below_0.990")
    if float(np.mean(sizes)) > 145:
        errors.append(f"mean_pool_over_145:{np.mean(sizes):.2f}")
    if max(sizes, default=0) > 180:
        errors.append(f"max_pool_over_180:{max(sizes)}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "candidate_audit.json"
    report = {
        "schema_version": SCHEMA, "v3_fingerprint": v3_fingerprint, "status": "PASS" if not errors else "FAIL",
        "errors": errors, "metrics": metrics,
        "pool_size": {"mean": float(np.mean(sizes)), "max": max(sizes), "min": min(sizes)},
        "candidate_fingerprint": hash_payload([record.get("qid") for record in records])
    }
    with stage_run(output_dir, "audit-candidates", total=len(records), v3_fingerprint=v3_fingerprint) as logger:
        atomic_json(report_path, report)
        if errors:
            raise RuntimeError("candidate audit failed: " + ", ".join(errors[:5]))
        logger.log(f"Recall@pool={target_recall:.6f} mean_pool={np.mean(sizes):.2f}")
        return write_manifest(output_dir, stage="audit-candidates", v3_fingerprint=v3_fingerprint,
                              config={"minimum_recall": 0.990}, files=[report_path], counts={"queries": len(records)})
