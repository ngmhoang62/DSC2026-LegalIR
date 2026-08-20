"""Packed, reusable reranker input IDs for EXP-012b evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from exp012b_core import PIPELINE_SCHEMA, atomic_json, canonical_json, content_hash, read_jsonl, sha256_file, stage_run


def build_token_cache(
    evidence_path: Path,
    output_dir: Path,
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    batch_pairs: int = 512,
    resume: bool = False,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    evidence_manifest = json.loads((evidence_path.parent / "manifest.json").read_text(encoding="utf-8"))
    source_fingerprint = evidence_manifest["content_fingerprint"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "input_ids.u32"
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"
    config_hash = content_hash(
        {"source": source_fingerprint, "model": model_name, "max_length": 512, "query_tokens": 128}
    )
    if resume and manifest_path.exists() and ids_path.exists() and records_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            cached.get("config_hash") == config_hash
            and cached.get("artifact_sha256", {}).get(ids_path.name) == sha256_file(ids_path)
            and cached.get("artifact_sha256", {}).get(records_path.name) == sha256_file(records_path)
        ):
            return cached
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
    query_cache: dict[str, str] = {}
    pending: list[tuple[str, str, str, str, str]] = []
    offset = 0
    pairs = 0
    queries = 0

    with stage_run(output_dir, "build-token-cache", total=int(evidence_manifest["queries"])) as logger, ids_path.open("wb") as ids_handle, records_path.open("w", encoding="utf-8", newline="\n") as records_handle:
        def flush() -> None:
            nonlocal offset, pairs
            if not pending:
                return
            qtexts: list[str] = []
            passages: list[str] = []
            for _, _, _, query, passage in pending:
                if query not in query_cache:
                    query_ids = tokenizer(query, add_special_tokens=False, return_attention_mask=False)["input_ids"][:128]
                    query_cache[query] = tokenizer.decode(query_ids, skip_special_tokens=True).strip()
                qtexts.append(query_cache[query])
                passages.append(passage)
            encoded = tokenizer(
                qtexts, passages, add_special_tokens=True, truncation="only_second",
                max_length=512, padding=False, return_attention_mask=False,
            )["input_ids"]
            for (qid, doc_id, chunk_id, _, passage), input_ids in zip(pending, encoded):
                values = np.asarray(input_ids, dtype=np.uint32)
                ids_handle.write(values.tobytes(order="C"))
                records_handle.write(canonical_json({
                    "qid": qid, "doc_id": doc_id, "chunk_id": chunk_id,
                    "bundle_hash": content_hash(passage), "offset": offset,
                    "length": len(values),
                }) + "\n")
                offset += len(values)
                pairs += 1
            pending.clear()

        for record in read_jsonl(evidence_path):
            queries += 1
            for candidate in record["candidates"]:
                for evidence in candidate["evidence"]:
                    pending.append((str(record["qid"]), str(candidate["doc_id"]), str(evidence["chunk_id"]), str(record["query"]), str(evidence["bundle_text"])))
                    if len(pending) >= batch_pairs:
                        flush()
            if queries % 64 == 0:
                logger.status(stage="build-token-cache", state="RUNNING", completed=queries, total=int(evidence_manifest["queries"]), pairs=pairs)
        flush()
        result = {
            "schema_version": PIPELINE_SCHEMA,
            "stage": "build-token-cache",
            "config_hash": config_hash,
            "source_fingerprint": source_fingerprint,
            "model": model_name,
            "counts": {"queries": queries, "pairs": pairs, "tokens": offset},
            "artifact_sha256": {ids_path.name: sha256_file(ids_path), records_path.name: sha256_file(records_path)},
        }
        result["content_fingerprint"] = content_hash(result)
        atomic_json(manifest_path, result)
    return result


class TokenCacheReader:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        tokens = int(self.manifest["counts"]["tokens"])
        self.ids = np.memmap(cache_dir / "input_ids.u32", dtype=np.uint32, mode="r", shape=(tokens,))

    def query_records(self, qids: set[str] | None = None) -> Iterator[dict[str, Any]]:
        current_qid: str | None = None
        current: list[dict[str, Any]] = []
        for row in read_jsonl(self.cache_dir / "records.jsonl"):
            qid = str(row["qid"])
            if qids is not None and qid not in qids:
                continue
            if current_qid is not None and qid != current_qid:
                yield {"qid": current_qid, "pairs": current}
                current = []
            current_qid = qid
            start = int(row["offset"])
            row["input_ids"] = np.asarray(self.ids[start : start + int(row["length"])], dtype=np.int64)
            current.append(row)
        if current_qid is not None:
            yield {"qid": current_qid, "pairs": current}
