"""Vietnamese-segmented, field-weighted BM25 index over structural-v3 passages."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

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


BM25_PROFILES = {
    "balanced": (1.0, 1.0, 1.0, 1.0),
    "legal_structure": (1.5, 2.0, 1.25, 1.0),
    "heading_priority": (1.0, 3.0, 1.0, 1.0),
}
_QUERY_TOKEN = re.compile(r"[\w_]+", flags=re.UNICODE)


def default_segmenter(text: str) -> str:
    from underthesea import word_tokenize

    return str(word_tokenize(text, format="text")).casefold()


def _segment_text_batch(texts: Sequence[str]) -> list[str]:
    """Process-pool entrypoint; Underthesea is imported once per worker."""
    return [default_segmenter(text) for text in texts]


def safe_fts_query(segmented_text: str) -> str:
    tokens = _QUERY_TOKEN.findall(segmented_text.casefold())
    if not tokens:
        return '"__legalir_no_token__"'
    # Every token is quoted so FTS operators in user text remain inert. OR is
    # deliberate: BM25 should retrieve partial lexical evidence, not hard-gate
    # on all query terms being present.
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class BM25Searcher:
    """Persistent read-only FTS5 searcher with exact document-bounded scans."""

    def __init__(
        self,
        database_path: Path,
        *,
        profile: str = "legal_structure",
        segmenter: Callable[[str], str] = default_segmenter,
        document_ranges: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        if profile not in BM25_PROFILES:
            raise ValueError(f"Unknown BM25 profile: {profile}")
        self.database_path = Path(database_path)
        self.profile = profile
        self.weights = BM25_PROFILES[profile]
        self.segmenter = segmenter
        self.connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro"
            + ("&immutable=1" if __import__("os").environ.get("LEGALIR_EXECUTION_PROFILE") == "optimized" else ""),
            uri=True,
            check_same_thread=False,
        )
        if __import__("os").environ.get("LEGALIR_EXECUTION_PROFILE") == "optimized":
            self.connection.execute("PRAGMA mmap_size=536870912")
            self.connection.execute("PRAGMA cache_size=-65536")
        self.document_ranges = dict(document_ranges) if document_ranges is not None else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "BM25Searcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def prepare_query(self, query: str) -> str:
        return safe_fts_query(self.segmenter(query))

    def load_document_ranges(self) -> dict[str, tuple[int, int]]:
        if self.document_ranges is not None:
            return self.document_ranges
        ranges: dict[str, tuple[int, int]] = {}
        for doc_id, first_row, last_row, count in self.connection.execute(
            "SELECT doc_id, min(rowid), max(rowid), count(*) FROM passages GROUP BY doc_id"
        ):
            first, last, size = int(first_row), int(last_row), int(count)
            if last - first + 1 != size:
                raise RuntimeError(f"Non-contiguous BM25 rows for document {doc_id}")
            ranges[str(doc_id)] = (first, last)
        self.document_ranges = ranges
        return ranges

    @staticmethod
    def _records(rows: Iterable[Sequence[Any]]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "parent_node_id": row[2],
                "score": float(row[3]),
                "rank": rank,
            }
            for rank, row in enumerate(rows, start=1)
        ]

    def search_expression(self, expression: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT chunk_id, doc_id, parent_node_id,
                      bm25(passages, ?, ?, ?, ?) AS score
               FROM passages WHERE passages MATCH ?
               ORDER BY score ASC, chunk_id ASC LIMIT ?""",
            (*self.weights, expression, limit),
        ).fetchall()
        return self._records(rows)

    def search(self, query: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        return self.search_expression(self.prepare_query(query), limit=limit)

    def search_document_expression(
        self, expression: str, doc_id: str, *, limit: int = 8
    ) -> list[dict[str, Any]]:
        bounds = self.load_document_ranges().get(str(doc_id))
        if bounds is None:
            return []
        first_row, last_row = bounds
        rows = self.connection.execute(
            """SELECT chunk_id, doc_id, parent_node_id,
                      bm25(passages, ?, ?, ?, ?) AS score
               FROM passages
               WHERE passages MATCH ? AND rowid BETWEEN ? AND ?
               ORDER BY score ASC, chunk_id ASC LIMIT ?""",
            (*self.weights, expression, first_row, last_row, limit),
        ).fetchall()
        if any(str(row[1]) != str(doc_id) for row in rows):
            raise RuntimeError(f"BM25 document row-range mismatch for {doc_id}")
        return self._records(rows)

    def search_document(self, query: str, doc_id: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return self.search_document_expression(self.prepare_query(query), doc_id, limit=limit)


def _extract_fields(
    chunk: dict[str, Any], document_label: str, scope_text: str
) -> tuple[str, str, str, str]:
    retrieval = str(chunk.get("retrieval_text", ""))
    hierarchy_lines: list[str] = []
    content = str(chunk.get("raw_text", ""))
    for line in retrieval.splitlines():
        stripped = line.strip()
        if stripped.startswith(("[Chương]", "[Mục]", "[Điều]")):
            hierarchy_lines.append(stripped.split("]", 1)[-1].strip())
    return document_label, " ".join(hierarchy_lines), scope_text, content


def tokenize_v3_fields(
    v3_dir: Path,
    output_dir: Path,
    *,
    segmenter: Callable[[str], str] = default_segmenter,
    resume: bool = False,
    shard_size: int = 8192,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = load_v3_manifest(v3_dir)
    documents = {row["doc_id"]: row for row in read_jsonl(v3_dir / "documents.jsonl")}
    scope_ids = {
        node_id
        for document in documents.values()
        for node_id in document.get("scope_node_ids", [])
    }
    scope_by_doc: dict[str, list[str]] = defaultdict(list)
    for node in read_jsonl(v3_dir / "nodes.jsonl"):
        if node["node_id"] in scope_ids:
            # The scope article may be long. Heading plus its opening text is
            # enough for the lexical field; full content remains indexed in
            # the passage column and is never discarded.
            scope_by_doc[node["doc_id"]].append(
                (str(node.get("heading_text", "")) + " " + str(node.get("raw_text", ""))[:1000]).strip()
            )
    segmented_scope = {
        doc_id: segmenter(" ".join(values)) for doc_id, values in scope_by_doc.items()
    }
    segmented_labels = {
        doc_id: segmenter(str(document.get("document_label", "")))
        for doc_id, document in documents.items()
    }
    segmented_hierarchy: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bm25_fields.jsonl"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    config_hash = content_hash(
        {
            "v3": manifest["content_fingerprint"],
            "segmenter": getattr(segmenter, "__name__", type(segmenter).__name__),
            "shard_size": shard_size,
        }
    )
    executor = (
        ProcessPoolExecutor(max_workers=workers)
        if workers > 1 and segmenter is default_segmenter else None
    )
    with stage_run(
        output_dir,
        "tokenize-bm25",
        total=manifest["counts"]["chunks"],
        v3_fingerprint=manifest["content_fingerprint"],
    ) as logger:
        chunks = read_jsonl(v3_dir / "chunks.jsonl")
        shard_paths: list[Path] = []
        count = 0
        for shard_index, start in enumerate(range(0, manifest["counts"]["chunks"], shard_size)):
            rows = []
            for _ in range(min(shard_size, manifest["counts"]["chunks"] - start)):
                rows.append(next(chunks))
            shard_path = shard_dir / f"fields_{shard_index:04d}.jsonl"
            marker_path = shard_dir / f"fields_{shard_index:04d}.json"
            if resume and shard_path.exists() and marker_path.exists():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("config_hash") == config_hash and marker.get("sha256") == sha256_file(shard_path):
                    shard_paths.append(shard_path)
                    count += len(rows)
                    continue
            extracted = [
                _extract_fields(
                    chunk,
                    str(documents[chunk["doc_id"]].get("document_label", "")),
                    segmented_scope.get(chunk["doc_id"], ""),
                )
                for chunk in rows
            ]
            hierarchy_values = [fields[1] for fields in extracted]
            for value in dict.fromkeys(hierarchy_values):
                if value not in segmented_hierarchy:
                    segmented_hierarchy[value] = segmenter(value)
            raw_contents = [fields[3] for fields in extracted]
            if executor is None:
                passage_contents = [segmenter(value) for value in raw_contents]
            else:
                width = max(1, (len(raw_contents) + workers - 1) // workers)
                futures = [
                    executor.submit(_segment_text_batch, raw_contents[pos : pos + width])
                    for pos in range(0, len(raw_contents), width)
                ]
                passage_contents = [value for future in futures for value in future.result()]
            temporary = shard_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as out:
                for local, (chunk, fields, passage_content) in enumerate(
                    zip(rows, extracted, passage_contents), start=1
                ):
                    row_id = start + local
                    row = {
                        "schema_version": PIPELINE_SCHEMA,
                        "row_id": row_id,
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "parent_node_id": chunk["parent_node_id"],
                        "document_label": segmented_labels[chunk["doc_id"]],
                        "hierarchy_text": segmented_hierarchy[fields[1]],
                        "scope_text": fields[2],
                        "passage_content": passage_content,
                    }
                    out.write(canonical_json(row) + "\n")
            temporary.replace(shard_path)
            atomic_json(
                marker_path,
                {
                    "config_hash": config_hash,
                    "start": start,
                    "end": start + len(rows),
                    "sha256": sha256_file(shard_path),
                },
            )
            shard_paths.append(shard_path)
            count += len(rows)
            logger.status(
                stage="tokenize-bm25",
                state="RUNNING",
                completed=count,
                total=manifest["counts"]["chunks"],
                shard=shard_index,
            )
        temporary_output = output_path.with_suffix(".jsonl.tmp")
        with temporary_output.open("wb") as output_handle:
            for shard_path in shard_paths:
                with shard_path.open("rb") as shard_handle:
                    while block := shard_handle.read(8 * 1024 * 1024):
                        output_handle.write(block)
        temporary_output.replace(output_path)
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
        result = artifact_manifest(
            stage="tokenize-bm25",
            inputs={"v3_fingerprint": manifest["content_fingerprint"]},
            config={
                "segmenter": getattr(segmenter, "__name__", type(segmenter).__name__),
                "shard_size": shard_size,
            },
            files=[output_path],
        )
        result["counts"] = {"passages": count}
        atomic_json(output_dir / "manifest.json", result)
    if executor is not None:
        executor.shutdown(wait=True)
    return result


def build_fts5_index(
    fields_dir: Path, output_dir: Path, *, commit_every: int = 8192, resume: bool = False
) -> dict[str, Any]:
    fields_manifest = json.loads((fields_dir / "manifest.json").read_text(encoding="utf-8"))
    fields_path = fields_dir / "bm25_fields.jsonl"
    if sha256_file(fields_path) != fields_manifest["artifact_sha256"][fields_path.name]:
        raise ValueError("BM25 fields cache fingerprint mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "bm25_v3.sqlite"
    config_hash = content_hash(
        {"fields": fields_manifest["content_fingerprint"], "tokenizer": "unicode61 tokenchars _"}
    )
    config_path = output_dir / "build_config.json"
    expected = fields_manifest.get("counts", {}).get("passages")
    if resume and database_path.exists():
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")).get(
            "config_hash"
        ) != config_hash:
            raise RuntimeError("Cannot resume BM25 index with changed or missing config")
    else:
        database_path.unlink(missing_ok=True)
    atomic_json(config_path, {"config_hash": config_hash})
    with stage_run(output_dir, "build-bm25") as logger:
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                    document_label, hierarchy_text, scope_text, passage_content,
                    chunk_id UNINDEXED, doc_id UNINDEXED, parent_node_id UNINDEXED,
                    tokenize = \"unicode61 tokenchars '_'\", columnsize=1
                )"""
            )
            existing = int(connection.execute("SELECT count(*) FROM passages").fetchone()[0])
            maximum = connection.execute("SELECT max(rowid) FROM passages").fetchone()[0]
            if existing and int(maximum) != existing:
                raise RuntimeError("BM25 resume requires contiguous rowids")
            count = 0
            for count, row in enumerate(read_jsonl(fields_path), start=1):
                if count <= existing:
                    continue
                connection.execute(
                    "INSERT INTO passages(rowid, document_label, hierarchy_text, scope_text, passage_content, chunk_id, doc_id, parent_node_id) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        row["row_id"],
                        row["document_label"],
                        row["hierarchy_text"],
                        row["scope_text"],
                        row["passage_content"],
                        row["chunk_id"],
                        row["doc_id"],
                        row["parent_node_id"],
                    ),
                )
                if count % commit_every == 0:
                    connection.commit()
                    logger.status(stage="build-bm25", state="RUNNING", completed=count, total=None)
            connection.commit()
            connection.execute("INSERT INTO passages(passages) VALUES('optimize')")
            connection.commit()
            integrity = connection.execute(
                "INSERT INTO passages(passages, rank) VALUES('integrity-check', 1)"
            )
            list(integrity)
            indexed = connection.execute("SELECT count(*) FROM passages").fetchone()[0]
            if expected is not None and indexed != expected:
                raise RuntimeError(f"Incomplete BM25 index: indexed={indexed}, expected={expected}")
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()
        wal = database_path.with_name(database_path.name + "-wal")
        shm = database_path.with_name(database_path.name + "-shm")
        wal.unlink(missing_ok=True)
        shm.unlink(missing_ok=True)
        result = artifact_manifest(
            stage="build-bm25",
            inputs={"fields_fingerprint": fields_manifest["content_fingerprint"]},
            config={"tokenizer": "unicode61 tokenchars _", "columnsize": 1},
            files=[database_path],
        )
        result["counts"] = {"passages": indexed}
        atomic_json(output_dir / "manifest.json", result)
    return result


def search_bm25(
    database_path: Path,
    query: str,
    *,
    profile: str = "legal_structure",
    limit: int = 2000,
    segmenter: Callable[[str], str] = default_segmenter,
) -> list[dict[str, Any]]:
    with BM25Searcher(database_path, profile=profile, segmenter=segmenter) as searcher:
        return searcher.search(query, limit=limit)


def search_bm25_document(
    database_path: Path,
    query: str,
    doc_id: str,
    *,
    profile: str = "legal_structure",
    limit: int = 8,
    segmenter: Callable[[str], str] = default_segmenter,
) -> list[dict[str, Any]]:
    """BM25 passages restricted to a known parent document (positive routing)."""
    with BM25Searcher(database_path, profile=profile, segmenter=segmenter) as searcher:
        return searcher.search_document(query, str(doc_id), limit=limit)


def aggregate_bm25_documents(hits: Iterable[dict[str, Any]], top_docs: int = 100) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        by_doc[str(hit["doc_id"])].append(hit)
    scored: list[dict[str, Any]] = []
    for doc_id, doc_hits in by_doc.items():
        unique: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        for hit in sorted(doc_hits, key=lambda item: (item["rank"], item["chunk_id"])):
            if hit["parent_node_id"] not in seen_parents:
                unique.append(hit)
                seen_parents.add(hit["parent_node_id"])
            if len(unique) == 3:
                break
        reciprocal = sum(1.0 / (60 + item["rank"]) for item in unique)
        scored.append(
            {
                "doc_id": doc_id,
                "score": reciprocal,
                "best_chunk_id": unique[0]["chunk_id"],
                "best_passage_rank": unique[0]["rank"],
                "unique_parent_hits": len(unique),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["best_passage_rank"], item["doc_id"]))
    for rank, row in enumerate(scored[:top_docs], start=1):
        row["rank"] = rank
    return scored[:top_docs]
