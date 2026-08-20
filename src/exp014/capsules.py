"""Evidence capsule construction for the EXP-014 shortlist candidates."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from exp012b_core import read_jsonl, sha256_file, stage_run, write_jsonl
from exp012b_bm25 import BM25Searcher
from exp012b_retrieval import PersistentChunkReader
from exp014.core import DEFAULT_RERANKER, hash_payload, load_document_metadata, write_manifest


def _truncate(tokenizer: Any, text: str, budget: int) -> str:
    ids = tokenizer(str(text), add_special_tokens=False, return_attention_mask=False)["input_ids"][:budget]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _hierarchy(chunk: Mapping[str, Any]) -> str:
    items = []
    for line in str(chunk.get("retrieval_text", "")).splitlines():
        if line.startswith(("[Chương]", "[Mục]", "[Điều]")):
            items.append(line.split("]", 1)[-1].strip())
    return " > ".join(items)


def _render(tokenizer: Any, query: str, document_label: str, chunks: list[dict[str, Any]], *, max_length: int = 768,
            scope_text: str = "") -> tuple[str, str]:
    query_value = _truncate(tokenizer, query, 128)
    first = chunks[0]
    fields = [f"[Văn bản] {_truncate(tokenizer, document_label, 64)}"]
    if scope_text.strip():
        fields.append(f"[Phạm vi] {_truncate(tokenizer, scope_text, 96)}")
    hierarchy = _hierarchy(first)
    if hierarchy:
        fields.append(f"[Cấu trúc] {_truncate(tokenizer, hierarchy, 64)}")
    body_prefix = "\n".join(fields)
    rendered_chunks = []
    for index, chunk in enumerate(chunks, 1):
        label = "[Nội dung]" if index == 1 else "[Bổ sung]"
        rendered_chunks.append(f"{label} {str(chunk['raw_text']).strip()}")
    document = body_prefix + "\n" + "\n".join(rendered_chunks)
    
    # Clean budget truncation without prompt template overhead
    budget = max_length - 16
    document = _truncate(tokenizer, document, budget)
    return query_value, document


def _select_ids(candidate: Mapping[str, Any], source: Mapping[str, Any], doc_chunks: Mapping[str, list[str]]) -> list[tuple[str, str]]:
    doc_id = str(candidate["doc_id"])
    selected: list[tuple[str, str]] = []
    evidence = source.get("evidence_by_doc", {}).get(doc_id, {}).get("evidence", [])
    bge = [row for row in evidence if str(row.get("provenance", "")).startswith("bge")]
    bm25 = [row for row in evidence if str(row.get("provenance", "")).startswith("bm25")]
    if bge:
        selected.append((str(bge[0]["chunk_id"]), str(bge[0].get("parent_node_id", ""))))
    primary_parent = selected[0][1] if selected else None
    lexical = next((row for row in bm25 if str(row.get("parent_node_id", "")) != primary_parent), None)
    if lexical is not None:
        selected.append((str(lexical["chunk_id"]), str(lexical.get("parent_node_id", ""))))
    elif len(bge) > 1:
        complement = next((row for row in bge[1:] if str(row.get("parent_node_id", "")) != primary_parent), None)
        if complement is not None:
            selected.append((str(complement["chunk_id"]), str(complement.get("parent_node_id", ""))))
    if selected:
        return selected[:2]
    for name in ("bge", "bm25"):
        chunk_id = candidate.get("sources", {}).get(name, {}).get("best_chunk_id")
        if chunk_id and all(str(chunk_id) != prior for prior, _ in selected):
            selected.append((str(chunk_id), ""))
        if len(selected) == 2:
            return selected
    return selected[:2]


class ScopedEvidenceFallback:
    """Exact BGE and BM25 refinement restricted to memory-only parent documents."""

    def __init__(self, *, wanted_docs: set[str], ranking_dir: Path, bge_dir: Path, bm25_db: Path, logger: Any | None = None):
        if logger is not None:
            logger.log(f"phase=scoped-fallback-index documents={len(wanted_docs)}")
        rows = list(read_jsonl(ranking_dir / "query_rows.jsonl"))
        self.query_row = {str(row["qid"]): int(row["row"]) for row in rows}
        self.query_vectors = np.load(ranking_dir / "query_embeddings.f16.npy", mmap_mode="r")
        self.leaf_vectors = np.memmap(bge_dir / "leaf_embeddings.f16", dtype=np.float16, mode="r")
        manifest = json.loads((bge_dir / "manifest.json").read_text(encoding="utf-8"))
        dimensions = int(manifest.get("counts", {}).get("dimension", manifest.get("config", {}).get("dimension", 1024)))
        self.leaf_vectors = self.leaf_vectors.reshape(-1, dimensions)
        self.rows: dict[int, dict[str, Any]] = {}
        self.doc_rows: dict[str, list[int]] = defaultdict(list)
        for row in read_jsonl(bge_dir / "leaf_rows.jsonl"):
            doc_id = str(row["doc_id"])
            if doc_id in wanted_docs:
                index = int(row["row"])
                self.rows[index] = row
                self.doc_rows[doc_id].append(index)
        self.bm25 = BM25Searcher(bm25_db, profile="legal_structure")
        if logger is not None:
            logger.log("phase=scoped-fallback-bm25-ranges")
        self.bm25.load_document_ranges()
        self.bm25_query: dict[str, str] = {}
        if logger is not None:
            logger.log("phase=scoped-fallback-ready")

    def close(self) -> None:
        self.bm25.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object):
        self.close()

    def select(self, *, qid: str, query: str, doc_id: str) -> list[tuple[str, str]]:
        indices = self.doc_rows.get(doc_id, [])
        dense: list[tuple[str, str, float]] = []
        if indices and qid in self.query_row:
            query_vector = np.asarray(self.query_vectors[self.query_row[qid]], dtype=np.float32)
            scores = np.asarray(self.leaf_vectors[indices], dtype=np.float32) @ query_vector
            order = np.argsort(-scores, kind="stable")
            seen: set[str] = set()
            for local in order:
                row = self.rows[indices[int(local)]]
                parent = str(row["parent_node_id"])
                if parent not in seen:
                    dense.append((str(row["chunk_id"]), parent, float(scores[int(local)])))
                    seen.add(parent)
                if len(dense) == 3:
                    break
        expression = self.bm25_query.get(qid)
        if expression is None:
            expression = self.bm25.prepare_query(query)
            self.bm25_query[qid] = expression
        lexical = self.bm25.search_document_expression(expression, doc_id, limit=8)
        selected: list[tuple[str, str]] = []
        if dense:
            selected.append((dense[0][0], dense[0][1]))
        primary_parent = selected[0][1] if selected else None
        bm25 = next((row for row in lexical if str(row["parent_node_id"]) != primary_parent), None)
        if bm25 is not None:
            selected.append((str(bm25["chunk_id"]), str(bm25["parent_node_id"])))
        elif len(dense) > 1:
            selected.append((dense[1][0], dense[1][1]))
        return selected[:2]


def build_capsules(*, split: str, candidates_path: Path, preranker_path: Path, rankings_path: Path,
                   v3_dir: Path, lookup_db: Path, output_dir: Path, v3_fingerprint: str,
                   tokenizer_name: str = DEFAULT_RERANKER, allow_download: bool = True, max_length: int = 768, shortlist_k: int = 25,
                   resume: bool = False, ranking_dir: Path | None = None, bge_dir: Path | None = None,
                   bm25_db: Path | None = None) -> dict[str, Any]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=not allow_download, trust_remote_code=True)
    candidates = {str(row["qid"]): row for row in read_jsonl(candidates_path)}
    predictions = {str(row["qid"]): row for row in read_jsonl(preranker_path)}
    source = {str(row["qid"]): row for row in read_jsonl(rankings_path)}
    if set(candidates) != set(predictions) or set(candidates) != set(source):
        raise ValueError("candidate/preranker/source query sets differ")
    docs = load_document_metadata(v3_dir)
    doc_chunks = json.loads((v3_dir / "doc_to_chunk_ids.json").read_text(encoding="utf-8"))
    scope_ids = {str(node_id) for row in docs.values() for node_id in row.get("scope_node_ids", [])}
    scope_nodes = {str(row["node_id"]): str(row.get("raw_text", "")) for row in read_jsonl(v3_dir / "nodes.jsonl") if str(row["node_id"]) in scope_ids}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_capsules.jsonl"
    build_fingerprint = hash_payload({"v3": v3_fingerprint, "split": split, "tokenizer": tokenizer_name,
                                      "max_length": max_length, "shortlist_k": shortlist_k,
                                      "candidates": sha256_file(candidates_path), "preranker": sha256_file(preranker_path),
                                      "source_rankings": sha256_file(rankings_path)})
    existing = {str(row["qid"]): row for row in read_jsonl(path) if row.get("build_fingerprint") == build_fingerprint} if resume and path.exists() else {}
    wanted_fallback_pairs: set[tuple[str, str]] = set()
    for qid, prediction in predictions.items():
        if qid in existing:
            continue
        candidate_by_doc = {str(row["doc_id"]): row for row in candidates[qid]["candidates"]}
        for ranked in prediction["candidates"][:shortlist_k]:
            doc_id = str(ranked["doc_id"])
            if doc_id not in source[qid].get("evidence_by_doc", {}) or not _select_ids(candidate_by_doc[doc_id], source[qid], doc_chunks):
                wanted_fallback_pairs.add((qid, doc_id))
    wanted_fallback_docs = {doc_id for _, doc_id in wanted_fallback_pairs}
    if wanted_fallback_pairs and not all((ranking_dir, bge_dir, bm25_db)):
        raise RuntimeError("memory-only capsules require ranking_dir, bge_dir, and bm25_db for scoped refinement")
    with stage_run(output_dir, "build-capsules", total=len(candidates), v3_fingerprint=v3_fingerprint) as logger:
        started = time.perf_counter()
        fallback_context = ScopedEvidenceFallback(wanted_docs=wanted_fallback_docs, ranking_dir=ranking_dir, bge_dir=bge_dir, bm25_db=bm25_db, logger=logger) if wanted_fallback_pairs else None
        with (fallback_context if fallback_context is not None else nullcontext()) as scoped, PersistentChunkReader(v3_dir, lookup_db) as reader:
            for position, qid in enumerate(sorted(candidates), 1):
                if qid in existing:
                    continue
                candidate_by_doc = {str(row["doc_id"]): row for row in candidates[qid]["candidates"]}
                ordered = predictions[qid]["candidates"]
                selected = ordered[:shortlist_k]
                need: set[str] = set()
                selections: dict[str, list[tuple[str, str]]] = {}
                for rank_row in selected:
                    doc_id = str(rank_row["doc_id"])
                    needs_scoped = (qid, doc_id) in wanted_fallback_pairs
                    ids = [] if needs_scoped else _select_ids(candidate_by_doc[doc_id], source[qid], doc_chunks)
                    if needs_scoped and scoped is not None:
                        ids = scoped.select(qid=qid, query=str(candidates[qid]["query"]), doc_id=doc_id)
                    if not ids:
                        raise RuntimeError(f"no capsule evidence qid={qid} doc={doc_id}")
                    selections[doc_id] = ids
                    need.update(chunk_id for chunk_id, _ in ids)
                chunks = reader.load(need)
                rows = []
                for rank_row in selected:
                    doc_id = str(rank_row["doc_id"])
                    evidence = [chunks[chunk_id] for chunk_id, _ in selections[doc_id]]
                    scope_text = "\n".join(scope_nodes.get(str(node_id), "") for node_id in docs[doc_id].get("scope_node_ids", []))
                    query, document = _render(tokenizer, candidates[qid]["query"], str(docs[doc_id].get("document_label", doc_id)), evidence, max_length=max_length, scope_text=scope_text)
                    rows.append({"doc_id": doc_id, "lambda_rank": int(rank_row["rank"]), "lambda_score": float(rank_row["lambda_score"]),
                                 "query": query, "document": document,
                                 "evidence": [{"chunk_id": chunk_id, "parent_node_id": str(chunks[chunk_id].get("parent_node_id", parent)),
                                               "provenance": "scoped_bge_bm25" if (qid, doc_id) in wanted_fallback_pairs else "exp012b"}
                                              for chunk_id, parent in selections[doc_id]]})
                existing[qid] = {"schema_version": "legalir.exp014.capsules.v1", "v3_fingerprint": v3_fingerprint,
                                "build_fingerprint": build_fingerprint, "qid": qid, "candidates": rows}
                if position % 64 == 0 or position == len(candidates):
                    logger.status(stage="build-capsules", state="RUNNING", completed=position, total=len(candidates))
                if position % 64 == 0 or position == len(candidates):
                    write_jsonl(path, [existing[known_qid] for known_qid in sorted(existing)])
                if position % 256 == 0 or position == len(candidates):
                    rate = position / max(time.perf_counter() - started, 1e-9)
                    logger.log(f"progress={position}/{len(candidates)} rate={rate:.2f}_queries_per_second eta_seconds={(len(candidates)-position)/max(rate,1e-9):.0f}")
        records = [existing[qid] for qid in sorted(existing)]
        write_jsonl(path, records)
        return write_manifest(output_dir, stage="build-capsules", v3_fingerprint=v3_fingerprint,
            config={"split": split, "max_length": max_length, "max_candidates": shortlist_k, "tokenizer": tokenizer_name},
            inputs={"build_fingerprint": build_fingerprint}, files=[path],
            counts={"queries": len(records), "capsules": sum(len(row["candidates"]) for row in records)})
