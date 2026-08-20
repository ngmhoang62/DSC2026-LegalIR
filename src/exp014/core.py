"""Shared deterministic infrastructure for EXP-014 Neuro-Symbolic Legal Retrieval."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Ensure src/ is in sys.path for root module imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from exp012b_core import (
    atomic_json,
    canonical_json,
    load_answers,
    load_queries,
    load_v3_manifest,
    read_jsonl,
    sha256_file,
    stage_run,
    write_jsonl,
)
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds


SCHEMA = "legalir.exp014.v1"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_success(path: Path, *, v3_fingerprint: str) -> dict[str, Any]:
    marker = path / "_SUCCESS.json"
    if not marker.exists():
        raise RuntimeError(f"Upstream stage incomplete: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("v3_fingerprint") != v3_fingerprint:
        raise RuntimeError(f"Upstream stage has stale structural-v3 fingerprint: {path}")
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Upstream stage has no manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("v3_fingerprint") != v3_fingerprint:
        raise RuntimeError(f"Upstream manifest is stale or has wrong schema: {path}")
    if payload.get("stage") != manifest.get("stage"):
        raise RuntimeError(f"Success marker/manifest stage mismatch: {path}")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        artifact = path / name
        if not artifact.exists() or sha256_file(artifact) != expected:
            raise RuntimeError(f"Stale/corrupt EXP-014 artifact: {artifact}")
    return payload


def write_manifest(
    directory: Path, *, stage: str, v3_fingerprint: str, config: Mapping[str, Any],
    files: Iterable[Path], counts: Mapping[str, int], inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hashes = {path.name: sha256_file(path) for path in files}
    payload = {
        "schema_version": SCHEMA,
        "stage": stage,
        "v3_fingerprint": v3_fingerprint,
        "inputs": dict(inputs or {}),
        "config": dict(config),
        "counts": dict(counts),
        "artifact_sha256": hashes,
    }
    payload["content_fingerprint"] = hash_payload(payload)
    atomic_json(directory / "manifest.json", payload)
    return payload


def preflight(v3_dir: Path, audit_path: Path) -> dict[str, Any]:
    v3 = load_v3_manifest(v3_dir, full_verify=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("structural-v3 audit must be PASS")
    if audit.get("content_fingerprint") != v3.get("content_fingerprint"):
        raise RuntimeError("structural-v3 audit fingerprint does not match manifest")
    if int(v3.get("counts", {}).get("documents", 0)) != 8532:
        raise RuntimeError("unexpected structural-v3 document count")
    return v3


def rank_ids(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda doc_id: (-float(scores[doc_id]), str(doc_id)))


def predictions_from_records(records: Iterable[Mapping[str, Any]], field: str = "candidates") -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in records:
        qid = str(row["qid"])
        if qid in output:
            raise ValueError(f"duplicate qid: {qid}")
        values = row[field]
        output[qid] = [str(value["doc_id"] if isinstance(value, Mapping) else value) for value in values]
    return output


def oracle(records: Iterable[Mapping[str, Any]], answers: Mapping[str, set[str]], *, field: str = "candidates") -> dict[str, float]:
    predictions = predictions_from_records(records, field)
    if set(predictions) != set(answers):
        raise ValueError("prediction/answer query sets differ")
    return evaluate_rankings(predictions, answers, ks=(5, 10, 20, 24, 25, 30, 32, 50, 120, 150, 180))


def load_document_metadata(v3_dir: Path) -> dict[str, dict[str, Any]]:
    documents = {str(row["doc_id"]): dict(row) for row in read_jsonl(v3_dir / "documents.jsonl")}
    doc_chunks = json.loads((v3_dir / "doc_to_chunk_ids.json").read_text(encoding="utf-8"))
    article_nodes: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(v3_dir / "chunks.jsonl"):
        if ":article:" in str(row["chunk_id"]):
            article_nodes[str(row["doc_id"])].add(str(row["parent_node_id"]))
    for doc_id, row in documents.items():
        row["chunk_count"] = len(doc_chunks.get(doc_id, []))
        row["article_count"] = len(article_nodes[doc_id])
        row["document_type"] = str(row.get("document_type") or row.get("parse_mode") or "unknown")
        row["hierarchy_available"] = int(row.get("parse_mode") != "fallback")
    return documents


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def bootstrap_recall_gain(
    current: Mapping[str, Sequence[str]], baseline: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]],
    *, draws: int = 2000, seed: int = 42,
) -> dict[str, float]:
    qids = sorted(answers)
    current_values = np.asarray([
        len(set(current[qid][:5]) & answers[qid]) / max(1, len(answers[qid])) for qid in qids
    ], dtype=np.float64)
    baseline_values = np.asarray([
        len(set(baseline[qid][:5]) & answers[qid]) / max(1, len(answers[qid])) for qid in qids
    ], dtype=np.float64)
    gains = current_values - baseline_values
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = float(gains[rng.integers(0, len(gains), len(gains))].mean())
    return {"mean": float(gains.mean()), "lower_95": float(np.quantile(samples, 0.025)), "upper_95": float(np.quantile(samples, 0.975))}
