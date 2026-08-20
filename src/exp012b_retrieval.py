"""Hybrid BM25/BGE fusion, metrics, and v3 evidence routing."""

from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from exp012b_core import (
    EVIDENCE_SCHEMA,
    artifact_manifest,
    atomic_json,
    load_v3_manifest,
    read_jsonl,
    stage_run,
)


DEFAULT_RRF_WEIGHTS = {"bm25": 1.0, "bge_block": 1.0, "bge_leaf": 1.5}


def weighted_rrf(
    rankings: Mapping[str, Sequence[dict[str, Any]]],
    *,
    weights: Mapping[str, float] = DEFAULT_RRF_WEIGHTS,
    rrf_k: int = 60,
    limit: int = 50,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    sources: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source, ranking in rankings.items():
        weight = float(weights.get(source, 1.0))
        seen: set[str] = set()
        for fallback_rank, item in enumerate(ranking, start=1):
            doc_id = str(item["doc_id"])
            if doc_id in seen:
                raise ValueError(f"Duplicate document {doc_id} in {source} ranking")
            seen.add(doc_id)
            rank = int(item.get("rank", fallback_rank))
            scores[doc_id] += weight / (rrf_k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
            sources[doc_id][source] = {
                "rank": rank,
                "score": float(item.get("score", 0.0)),
            }
    ordered = sorted(scores, key=lambda doc: (-scores[doc], best_rank[doc], doc))[:limit]
    return [
        {
            "doc_id": doc_id,
            "rank": rank,
            "rrf_score": scores[doc_id],
            "best_source_rank": best_rank[doc_id],
            "sources": sources[doc_id],
        }
        for rank, doc_id in enumerate(ordered, start=1)
    ]


def candidate_union(*rankings: Sequence[dict[str, Any]]) -> set[str]:
    return {str(row["doc_id"]) for ranking in rankings for row in ranking}


def recall_at_k(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], k: int) -> float:
    recalls: list[float] = []
    for qid, gold in answers.items():
        if not gold:
            continue
        found = len(set(map(str, predictions.get(qid, [])[:k])) & gold)
        recalls.append(found / len(gold))
    return float(np.mean(recalls)) if recalls else 0.0


def precision_at_k(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], k: int) -> float:
    values = [
        len(set(map(str, predictions.get(qid, [])[:k])) & gold) / k
        for qid, gold in answers.items()
        if gold
    ]
    return float(np.mean(values)) if values else 0.0


def evaluate_rankings(
    predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], ks: Sequence[int] = (5, 20, 30, 50)
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(predictions, answers, k)
        if k == 5:
            metrics[f"precision@{k}"] = precision_at_k(predictions, answers, k)
    return metrics


def select_evidence_ids(
    dense_hits: Sequence[dict[str, Any]],
    bm25_hits: Sequence[dict[str, Any]],
    *,
    maximum: int = 3,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_parents: set[str] = set()

    def add(hit: dict[str, Any], provenance: str) -> bool:
        chunk_id = str(hit["chunk_id"])
        if chunk_id in selected_ids:
            return False
        selected.append(
            {
                "chunk_id": chunk_id,
                "parent_node_id": str(hit["parent_node_id"]),
                "provenance": provenance,
                "score": float(hit.get("score", 0.0)),
                "rank": int(hit.get("rank", 0)),
            }
        )
        selected_ids.add(chunk_id)
        selected_parents.add(str(hit["parent_node_id"]))
        return True

    if dense_hits:
        add(dense_hits[0], "bge_primary")
    for hit in dense_hits[1:]:
        if hit["parent_node_id"] not in selected_parents and add(hit, "bge_structural_complement"):
            break
    for hit in bm25_hits:
        if add(hit, "bm25_lexical_complement"):
            break
    for hit in dense_hits:
        if len(selected) >= maximum:
            break
        add(hit, "bge_fallback")
    return selected[:maximum]


def adaptive_evidence_limit(parent_rank: int) -> int:
    if parent_rank <= 10:
        return 3
    if parent_rank <= 30:
        return 2
    return 1


def build_chunk_offset_index(v3_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Index byte offsets, avoiding a second copy of the large v3 text."""
    v3_manifest = load_v3_manifest(v3_dir)
    source = v3_dir / "chunks.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "chunk_offsets.sqlite"
    database.unlink(missing_ok=True)
    count = 0
    with stage_run(
        output_dir,
        "build-chunk-lookup",
        total=int(v3_manifest["counts"]["chunks"]),
        v3_fingerprint=v3_manifest["content_fingerprint"],
    ) as logger:
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                "CREATE TABLE chunk_offsets(chunk_id TEXT PRIMARY KEY, byte_offset INTEGER NOT NULL)"
            )
            with source.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    row = json.loads(line)
                    connection.execute(
                        "INSERT INTO chunk_offsets(chunk_id, byte_offset) VALUES(?, ?)",
                        (str(row["chunk_id"]), offset),
                    )
                    count += 1
                    if count % 8192 == 0:
                        connection.commit()
                        logger.status(
                            stage="build-chunk-lookup",
                            state="RUNNING",
                            completed=count,
                            total=int(v3_manifest["counts"]["chunks"]),
                        )
            connection.commit()
            connection.execute("CREATE INDEX offset_order ON chunk_offsets(byte_offset)")
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()
        expected = int(v3_manifest["counts"]["chunks"])
        if count != expected:
            raise RuntimeError(f"Chunk lookup count mismatch: {count}/{expected}")
        result = artifact_manifest(
            stage="build-chunk-lookup",
            inputs={
                "v3_fingerprint": v3_manifest["content_fingerprint"],
                "chunks_sha256": v3_manifest["artifact_sha256"][source.name],
            },
            config={"format": "jsonl-byte-offset-v1"},
            files=[database],
        )
        result["counts"] = {"chunks": count}
        atomic_json(output_dir / "manifest.json", result)
    return result


def load_chunk_records(
    v3_dir: Path,
    needed_chunk_ids: set[str],
    *,
    lookup_database: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if not needed_chunk_ids:
        return {}
    if lookup_database is not None:
        offsets: dict[str, int] = {}
        connection = sqlite3.connect(f"file:{lookup_database.as_posix()}?mode=ro", uri=True)
        try:
            wanted = sorted(needed_chunk_ids)
            for start in range(0, len(wanted), 900):
                batch = wanted[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                offsets.update(
                    (str(chunk_id), int(offset))
                    for chunk_id, offset in connection.execute(
                        f"SELECT chunk_id, byte_offset FROM chunk_offsets WHERE chunk_id IN ({placeholders})",
                        batch,
                    )
                )
        finally:
            connection.close()
        missing_offsets = sorted(needed_chunk_ids - offsets.keys())
        if missing_offsets:
            raise ValueError(f"Unknown v3 evidence chunks: {missing_offsets[:10]}")
        found: dict[str, dict[str, Any]] = {}
        with (v3_dir / "chunks.jsonl").open("rb") as handle:
            for chunk_id, offset in sorted(offsets.items(), key=lambda item: item[1]):
                handle.seek(offset)
                row = json.loads(handle.readline())
                if str(row.get("chunk_id")) != chunk_id:
                    raise RuntimeError(f"Stale chunk offset index at {chunk_id}")
                found[chunk_id] = row
        return found
    found: dict[str, dict[str, Any]] = {}
    for chunk in read_jsonl(v3_dir / "chunks.jsonl"):
        if chunk["chunk_id"] in needed_chunk_ids:
            found[chunk["chunk_id"]] = chunk
    missing = sorted(needed_chunk_ids - found.keys())
    if missing:
        raise ValueError(f"Unknown v3 evidence chunks: {missing[:10]}")
    return found


class PersistentChunkReader:
    """One connection/file handle per routing stage with a bounded row cache."""

    def __init__(
        self,
        v3_dir: Path,
        lookup_database: Path,
        *,
        cache_size: int = 8192,
        immutable: bool = True,
    ) -> None:
        self.source = (v3_dir / "chunks.jsonl").open("rb")
        immutable_flag = "&immutable=1" if immutable else ""
        self.connection = sqlite3.connect(
            f"file:{lookup_database.as_posix()}?mode=ro{immutable_flag}", uri=True
        )
        self.cache_size = max(0, cache_size)
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def close(self) -> None:
        self.connection.close()
        self.source.close()

    def __enter__(self) -> "PersistentChunkReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def load(self, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for chunk_id in sorted(chunk_ids):
            cached = self.cache.get(chunk_id)
            if cached is None:
                missing.append(chunk_id)
            else:
                self.cache.move_to_end(chunk_id)
                found[chunk_id] = cached
        offsets: dict[str, int] = {}
        for start in range(0, len(missing), 900):
            batch = missing[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            offsets.update(
                (str(chunk_id), int(offset))
                for chunk_id, offset in self.connection.execute(
                    f"SELECT chunk_id, byte_offset FROM chunk_offsets WHERE chunk_id IN ({placeholders})",
                    batch,
                )
            )
        unknown = sorted(set(missing) - offsets.keys())
        if unknown:
            raise ValueError(f"Unknown v3 evidence chunks: {unknown[:10]}")
        for chunk_id, offset in sorted(offsets.items(), key=lambda item: item[1]):
            self.source.seek(offset)
            row = json.loads(self.source.readline())
            if str(row.get("chunk_id")) != chunk_id:
                raise RuntimeError(f"Stale chunk offset index at {chunk_id}")
            found[chunk_id] = row
            if self.cache_size:
                self.cache[chunk_id] = row
                self.cache.move_to_end(chunk_id)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return found


def render_evidence_bundle(
    query: str,
    document_label: str,
    chunk: dict[str, Any],
    *,
    scope_excerpt: str = "",
    tokenizer: Any | None = None,
    max_length: int = 512,
    max_query_tokens: int = 128,
) -> str:
    hierarchy: list[str] = []
    for line in str(chunk["retrieval_text"]).splitlines():
        if line.startswith(("[Chương]", "[Mục]", "[Điều]")):
            hierarchy.append(line.split("]", 1)[1].strip())
    def truncate(value: str, budget: int) -> str:
        if tokenizer is None:
            return value.strip()
        encoded = tokenizer(value, add_special_tokens=False, return_attention_mask=False)
        ids = encoded["input_ids"][:budget]
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    query_for_pair = truncate(query, max_query_tokens)
    fields = [f"[Văn bản] {truncate(document_label, 32)}"]
    if scope_excerpt.strip():
        fields.append(f"[Phạm vi áp dụng] {truncate(scope_excerpt, 64)}")
    if hierarchy:
        fields.append(f"[Cấu trúc] {truncate(' > '.join(hierarchy), 48)}")
    prefix = "\n".join(fields + ["[Nội dung]"])
    if tokenizer is None:
        return prefix + " " + str(chunk["raw_text"])
    query_ids = tokenizer(query_for_pair, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    content_budget = max(1, max_length - len(query_ids) - len(prefix_ids) - special_tokens)
    content = truncate(str(chunk["raw_text"]), content_budget)
    bundle = prefix + " " + content
    # Boundary tokenization can be non-additive. Coarsen only passage content
    # until the real encoded pair fits; protected metadata/query remain intact.
    while len(tokenizer(query_for_pair, bundle, add_special_tokens=True)["input_ids"]) > max_length:
        content_budget -= 1
        if content_budget <= 0:
            raise ValueError("Protected evidence metadata exceeds reranker token budget")
        content = truncate(str(chunk["raw_text"]), content_budget)
        bundle = prefix + " " + content
    return bundle


def build_evidence_record(
    *,
    qid: str,
    query: str,
    candidate: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    chunks: Mapping[str, dict[str, Any]],
    document_label: str,
    v3_fingerprint: str,
    scope_excerpt: str = "",
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    limit = adaptive_evidence_limit(int(candidate["rank"]))
    evidence = []
    for selection in selected[:limit]:
        chunk = chunks[selection["chunk_id"]]
        evidence.append(
            {
                **selection,
                "doc_id": candidate["doc_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "bundle_text": render_evidence_bundle(
                    query,
                    document_label,
                    chunk,
                    scope_excerpt=scope_excerpt,
                    tokenizer=tokenizer,
                ),
            }
        )
    if not evidence:
        raise ValueError(f"Candidate {candidate['doc_id']} has no evidence")
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "v3_fingerprint": v3_fingerprint,
        "qid": str(qid),
        "query": query,
        "candidate": candidate,
        "evidence": evidence,
    }


def validate_candidate_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    queries = 0
    candidates = 0
    evidence = 0
    seen_qids: set[str] = set()
    for record in records:
        qid = str(record["qid"])
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid {qid}")
        seen_qids.add(qid)
        queries += 1
        rows = record["candidates"]
        doc_ids = [str(row["doc_id"]) for row in rows]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"Duplicate candidate document for qid {qid}")
        candidates += len(rows)
        for row in rows:
            ev = row.get("evidence", [])
            if len(ev) > 3:
                raise ValueError(f"More than three evidence passages for {qid}/{row['doc_id']}")
            ids = [item["chunk_id"] for item in ev]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate evidence passage for {qid}/{row['doc_id']}")
            evidence += len(ev)
    return {"queries": queries, "candidates": candidates, "evidence": evidence}
