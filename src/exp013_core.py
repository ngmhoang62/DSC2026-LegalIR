"""Deterministic utilities and schemas for EXP-013 selective late interaction.

The module intentionally has no model import.  Every artifact carries the v3
fingerprint that produced it, making a stale structural cache a hard failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from exp012b_core import (
    atomic_json,
    canonical_json,
    content_hash,
    load_answers,
    load_queries,
    load_v3_manifest,
    read_jsonl,
    sha256_file,
    stage_run,
    write_jsonl,
)
from exp012b_retrieval import evaluate_rankings, weighted_rrf


EXP013_SCHEMA = "legalir.exp013_slid.v1"
DEFAULT_MODEL = "jinaai/jina-colbert-v2"
DEFAULT_DIMENSIONS = 64
MAX_ANCHORS = 48


def require_exp013_success(stage_dir: Path, v3_fingerprint: str | None = None) -> dict[str, Any]:
    marker_path = stage_dir / "_SUCCESS.json"
    if not marker_path.exists():
        raise RuntimeError(f"Upstream stage incomplete: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if v3_fingerprint and marker.get("v3_fingerprint") != v3_fingerprint:
        raise RuntimeError(f"Upstream stage has stale v3 fingerprint: {stage_dir}")
    return marker


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def v3_preflight(v3_dir: Path, audit_path: Path) -> dict[str, Any]:
    """The trust boundary for all EXP-013 build stages."""
    v3 = load_v3_manifest(v3_dir, full_verify=True)
    audit = read_manifest(audit_path)
    if audit.get("status") != "PASS":
        raise RuntimeError(f"Structural-v3 audit is not PASS: {audit.get('status')}")
    if audit.get("content_fingerprint") != v3.get("content_fingerprint"):
        raise RuntimeError("Structural-v3 audit does not match current manifest")
    if int(v3.get("counts", {}).get("documents", 0)) != 8532:
        raise RuntimeError("Unexpected structural-v3 document count")
    return v3


def stable_topk(scores: np.ndarray, ids: Sequence[str], limit: int) -> list[int]:
    """Descending score, then ascending ID.  Works without a FAISS dependency."""
    if scores.ndim != 1 or len(scores) != len(ids):
        raise ValueError("scores/ids shape mismatch")
    if limit < 1:
        return []
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), str(ids[i])))
    return order[:limit]


def validate_candidate_record(record: Mapping[str, Any], *, maximum: int = 96) -> None:
    candidates = list(record.get("candidates", []))
    ids = [str(row["doc_id"]) for row in candidates]
    if not candidates or len(candidates) > maximum or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid candidate pool for qid={record.get('qid')}")
    for expected_rank, row in enumerate(candidates, 1):
        if int(row["rank"]) != expected_rank:
            raise ValueError(f"Non-contiguous candidate rank for qid={record.get('qid')}")


def write_stage_manifest(
    output_dir: Path,
    *,
    stage: str,
    v3_fingerprint: str,
    config: Mapping[str, Any],
    files: Iterable[Path],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    hashes = {path.name: sha256_file(path) for path in files}
    result = {
        "schema_version": EXP013_SCHEMA,
        "stage": stage,
        "v3_fingerprint": v3_fingerprint,
        "config": dict(config),
        "counts": dict(counts),
        "artifact_sha256": hashes,
    }
    result["content_fingerprint"] = content_hash(result)
    atomic_json(output_dir / "manifest.json", result)
    return result


def oracle_metrics(records: Iterable[Mapping[str, Any]], answers: Mapping[str, set[str]]) -> dict[str, float]:
    predictions = {
        str(row["qid"]): [str(candidate["doc_id"]) for candidate in row["candidates"]]
        for row in records
    }
    return evaluate_rankings(predictions, answers, ks=(5, 10, 20, 30, 50, 64, 96))


def fuse_rankings(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]], *, limit: int = 64
) -> list[dict[str, Any]]:
    """RRF used only as a candidate recall mechanism, never as a final ranker."""
    normalised = {
        source: [dict(item) for item in rows]
        for source, rows in rankings.items() if rows
    }
    return weighted_rrf(normalised, weights={source: 1.0 for source in normalised}, limit=limit)
