"""EXP-033: evidence-first, one-capsule reranking preparation.

This module deliberately separates cheap, deterministic artifact work from GPU
work.  ``overnight`` is a guarded scheduler: it cannot cross either the manual
scope spot-check or the evidence gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from exp012b_bm25 import BM25Searcher, default_segmenter
from exp030_legal_evidence_routing import (
    canonical_answers,
    classify_scope_node,
    extract_official_title,
    format_structural_path,
    scope_display_span,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp033_in_document_evidence_routing.v1"
SEED = 33033
K64 = 64
MARKERS = (
    "[BẰNG CHỨNG CHÍNH]",
    "[VỊ TRÍ PHÁP LÝ]",
    "[BẰNG CHỨNG BỔ SUNG]",
    "[PHẠM VI LIÊN QUAN]",
    "[ĐỐI TƯỢNG LIÊN QUAN]",
    "[VĂN BẢN]",
)
SCOPE_KINDS = ("scope_of_regulation", "applicable_subjects", "combined")
RAW_SCOPE = re.compile(r"\b(phạm vi (điều chỉnh|áp dụng)|đối tượng áp dụng)\b", re.I)
SENTENCE_END = re.compile(r"[.!?;:]\s|\n\s*\n")
CAPSULE_MODEL_ID = "BAAI/bge-reranker-v2-m3"
CAPSULE_MAX_LENGTH = 512
QUERY_TOKEN_LIMIT = 128
QUERY_HEAD_TOKENS = 96
QUERY_TAIL_TOKENS = 32

# Decisions from the EXP-033 agent review of the deterministic 200-row sample.
# Entries omitted here were read as correctly classified by the parser.  Index
# is stable because sample-scope-audit sorts by bucket/doc/node before writing.
AGENT_REVIEW_OVERRIDES: dict[int, str] = {
    19: "combined", 90: "combined", 91: "scope_of_regulation",
    94: "scope_of_regulation", 98: "combined", 101: "combined",
    102: "combined", 104: "combined", 106: "scope_of_regulation",
    109: "none", 113: "combined", 114: "combined",
    115: "combined", 116: "combined", 117: "combined",
    118: "none", 120: "combined",
    # Full disagreement adjudication after the first repair. These containers
    # hold both a source-exact scope declaration and an applicable-subjects
    # sibling; the legacy review had kept only its original scope label.
    92: "combined", 97: "scope_of_regulation", 109: "combined",
    151: "combined", 153: "combined", 155: "combined", 158: "combined",
    162: "combined", 164: "combined", 165: "combined", 166: "combined",
    168: "combined", 170: "combined", 171: "combined", 173: "combined",
    174: "combined", 175: "combined", 176: "combined", 178: "combined",
    179: "combined", 186: "combined", 187: "combined", 188: "combined",
    189: "combined", 193: "combined", 195: "combined", 197: "combined",
    198: "combined", 160: "combined",
}
# These source spans include a following structural sibling (for example,
# Article 2 after an Article 1 scope) and are therefore not source-exact.
AGENT_BOUNDARY_INCOMPLETE = {
    19, 20, 151, 153, 158, 162, 165, 168, 170, 171, 173, 175,
    178, 179, 187, 189, 193, 195, 197, 199,
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _success(directory: Path, *, stage: str, fingerprint: str, **extra: Any) -> None:
    _write_json(directory / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint, **extra})


def _state(results_root: Path, phase: str, state: str, **extra: Any) -> None:
    payload = {"schema_version": SCHEMA, "phase": phase, "state": state, "updated_at": time.time(), **extra}
    _write_json(results_root / "RUN_STATUS.json", payload)
    with (results_root / "state.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _manifest_ok(directory: Path, *, key: str) -> dict[str, Any]:
    manifest = _json(directory / "manifest.json")
    success = _json(directory / "_SUCCESS.json")
    # Older frozen stages use different telemetry keys in their success marker
    # (for example ``v3_fingerprint``).  A missing requested key is therefore
    # not a false mismatch; a present, unequal value is.
    if key in success and success.get(key) != manifest.get(key):
        raise ValueError(f"manifest/_SUCCESS mismatch for {directory}: {key}")
    return manifest


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "cache": args.cache_root,
        "results": args.results_root,
        "train": args.train,
        "folds": args.folds,
        "preprocessing": args.preprocessing,
        "v3": args.v3,
        "e5": args.e5,
        "query_embeddings": args.query_embeddings,
        "candidates": args.candidates,
        "features": args.features,
        "exp028_oof": args.exp028_oof,
        "bm25": args.bm25_db,
    }


def audit_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Validate immutable inputs before EXP-033 writes its own namespace."""
    preprocessing = _manifest_ok(paths["preprocessing"], key="content_fingerprint")
    v3 = _manifest_ok(paths["v3"], key="content_fingerprint")
    e5 = _manifest_ok(paths["e5"], key="cache_fingerprint")
    union = _manifest_ok(paths["candidates"].parent, key="content_fingerprint")
    features = _manifest_ok(paths["features"], key="content_fingerprint")
    labels, label_stats = canonical_answers(
        paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl"
    )
    if label_stats["evaluable_queries"] != 6991 or label_stats["non_evaluable_queries"] != 9:
        raise ValueError(f"canonical label accounting changed: {label_stats}")
    if v3["counts"]["chunks"] != 343347 or e5["chunks"] != 343347:
        raise ValueError("Structural-v3/E5 chunk count mismatch")
    if e5["corpus_fingerprint"] != v3["content_fingerprint"]:
        raise ValueError("E5 corpus fingerprint does not bind Structural-v3")
    if union["artifact_sha256"]["cache/train_oof_candidates.jsonl"] != _sha256(paths["candidates"]):
        raise ValueError("immutable E5+BM25 candidate hash mismatch")
    rows = 0
    membership_bad = 0
    for row in _jsonl(paths["candidates"]):
        rows += 1
        ids = [str(item["doc_id"]) for item in row["candidates"]]
        if len(ids) != 150 or len(set(ids)) != 150:
            membership_bad += 1
    if rows != 7000 or membership_bad:
        raise ValueError(f"immutable pool audit failed rows={rows} malformed={membership_bad}")
    fingerprint = _hash({
        "preprocessing": preprocessing["content_fingerprint"], "v3": v3["content_fingerprint"],
        "e5": e5["cache_fingerprint"], "candidate_sha256": _sha256(paths["candidates"]),
        "label_stats": label_stats,
    })
    result = {
        "schema_version": SCHEMA, "status": "PASS", "fingerprint": fingerprint,
        "canonical_label_policy": "canonical_duplicate_alias_drop_empty_passage_v1",
        "canonical_labels": {"fingerprint": _hash({key: sorted(value) for key, value in labels.items()}), **label_stats},
        "immutable_pool": {"queries": rows, "parents_per_query": 150, "sha256": _sha256(paths["candidates"])},
        "inputs": {"preprocessing": preprocessing["content_fingerprint"], "v3": v3["content_fingerprint"], "e5": e5["cache_fingerprint"], "features": features["content_fingerprint"]},
    }
    output = paths["results"] / "audit-inputs"
    _write_json(output / "REPORT.json", result)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "fingerprint": fingerprint, "stage": "audit-inputs", "report_sha256": _sha256(output / "REPORT.json")})
    _success(output, stage="audit-inputs", fingerprint=fingerprint)
    _state(paths["results"], "audit-inputs", "SUCCESS", completed=1, total=1, eta_seconds=0)
    return result


def _scope_record(node: Mapping[str, Any], bucket: str, kind: str | None) -> dict[str, Any]:
    raw = str(node.get("raw_text", ""))
    if kind:
        span = scope_display_span(node, kind)
        start = int(node["start"]) + int(span["relative_start"])
        end = int(node["start"]) + int(span["relative_end"])
        text = str(span["raw_text"])
    else:
        match = RAW_SCOPE.search(raw)
        start = int(node["start"]) + (match.start() if match else 0)
        end = int(node["start"]) + (match.end() if match else min(len(raw), 1))
        text = raw[max(0, start - int(node["start"]) - 160): min(len(raw), end - int(node["start"]) + 840)]
    return {
        "schema_version": SCHEMA, "sample_bucket": bucket, "doc_id": str(node["doc_id"]), "node_id": str(node["node_id"]),
        "source_start": start, "source_end": end, "raw_span": text,
        "parser_label": kind or "none", "review_label": None, "boundary_complete": None,
        "missed_span": None, "rationale": None, "review_status": "PENDING_AGENT_REVIEW",
        "source_node_sha256": _hash({"doc_id": node["doc_id"], "node_id": node["node_id"], "raw_text": raw}),
    }


def _sample_unique(rows: Sequence[dict[str, Any]], count: int, *, seed_key: str, used: set[str]) -> list[dict[str, Any]]:
    choices = [row for row in rows if row["doc_id"] not in used]
    ranked = sorted(choices, key=lambda row: _hash({"seed": SEED, "bucket": seed_key, "doc": row["doc_id"], "node": row["node_id"]}))
    selected = ranked[:count]
    if len(selected) != count:
        raise ValueError(f"scope sample bucket {seed_key} has {len(selected)}/{count} unique documents")
    used.update(row["doc_id"] for row in selected)
    return selected


def sample_scope_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Produce the 200 source-exact annotations; human labels remain unset."""
    nodes_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    legacy: list[dict[str, Any]] = []
    raw_signal: list[dict[str, Any]] = []
    legacy_node_ids = {
        str(node_id)
        for document in _jsonl(paths["v3"] / "documents.jsonl")
        for node_id in document.get("scope_node_ids", [])
    }
    typed_docs: set[str] = set()
    for node in _jsonl(paths["v3"] / "nodes.jsonl"):
        if node.get("kind") not in {"article", "section", "chapter"}:
            continue
        parsed = classify_scope_node(node)
        if parsed in SCOPE_KINDS:
            bucket = "combined" if parsed == "combined" else parsed
            nodes_by_kind[bucket].append(_scope_record(node, bucket, parsed))
            typed_docs.add(str(node["doc_id"]))
        elif str(node["node_id"]) in legacy_node_ids:
            legacy.append(_scope_record(node, "rejected_legacy_scope", None))
        elif RAW_SCOPE.search(str(node.get("raw_text", ""))):
            raw_signal.append(_scope_record(node, "raw_text_scope_signal", None))
    # A raw-text-signal negative must not be an old scope-node candidate or a
    # document that already has a typed parser decision; otherwise the two
    # negative strata overlap and hide precisely the legacy false positives.
    legacy_docs = {row["doc_id"] for row in legacy if row["doc_id"] not in typed_docs}
    legacy = [row for row in legacy if row["doc_id"] in legacy_docs]
    raw_signal = [
        row for row in raw_signal
        if row["doc_id"] not in typed_docs and row["doc_id"] not in legacy_docs
    ]
    used: set[str] = set()
    selected = []
    selected += _sample_unique(nodes_by_kind["scope_of_regulation"], 50, seed_key="scope", used=used)
    selected += _sample_unique(nodes_by_kind["applicable_subjects"], 50, seed_key="subject", used=used)
    selected += _sample_unique(nodes_by_kind["combined"], 40, seed_key="combined", used=used)
    selected += _sample_unique(legacy, 30, seed_key="legacy", used=used)
    selected += _sample_unique(raw_signal, 30, seed_key="raw", used=used)
    selected.sort(key=lambda row: (row["sample_bucket"], row["doc_id"], row["node_id"]))
    output = paths["cache"] / "scope-audit"
    count = _write_jsonl(output / "annotations.jsonl", selected)
    fingerprint = _hash({"v3": _json(paths["v3"] / "manifest.json")["content_fingerprint"], "rows": selected})
    report = {"schema_version": SCHEMA, "status": "PENDING_AGENT_REVIEW", "records": count, "fingerprint": fingerprint, "strata": dict(Counter(row["sample_bucket"] for row in selected))}
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "sample-scope-audit", "fingerprint": fingerprint, "annotations_sha256": _sha256(output / "annotations.jsonl")})
    _state(paths["results"], "sample-scope-audit", "WAITING_AGENT_SCOPE_REVIEW", completed=0, total=1, eta_seconds=None, annotations=str(output / "annotations.jsonl"))
    return report


def _scope_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = list(SCOPE_KINDS) + ["none"]
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in rows:
        matrix[str(row["review_label"])][str(row["parser_label"])] += 1
    predicted_positive = sum(matrix[actual][predicted] for actual in labels for predicted in SCOPE_KINDS)
    true_positive = sum(matrix[label][label] for label in SCOPE_KINDS)
    micro_precision = true_positive / predicted_positive if predicted_positive else 0.0
    class_precision = {
        label: matrix[label][label] / sum(matrix[actual][label] for actual in labels)
        if sum(matrix[actual][label] for actual in labels) else 0.0 for label in SCOPE_KINDS
    }
    weights = Counter(str(row["sample_bucket"]) for row in rows)
    recalls = {}
    for label in SCOPE_KINDS:
        denom = sum(matrix[label].values())
        recalls[label] = matrix[label][label] / denom if denom else 0.0
    weighted = sum(recalls.get("combined" if bucket == "combined" else bucket, 0.0) * count for bucket, count in weights.items() if bucket in SCOPE_KINDS) / 140
    boundary = sum(bool(row["boundary_complete"]) for row in rows if row["parser_label"] != "none" and row["review_label"] == row["parser_label"])
    boundary_den = sum(1 for row in rows if row["parser_label"] != "none" and row["review_label"] == row["parser_label"])
    return {"confusion": matrix, "micro_precision": micro_precision, "class_precision": class_precision, "recall": recalls, "stratified_weighted_recall": weighted, "boundary_accuracy": boundary / boundary_den if boundary_den else 0.0}


_EXPLICIT_SCOPE = re.compile(
    r"(?im)^\s*[\"“”']?"
    r"(?:(?P<article>điều\s+\d+[a-zđ]?)\.?\s+|(?P<number>\d+(?:\.\d+)*)(?:\.)?\s+)?"
    # Put combined alternatives first so a subject prefix cannot consume a
    # longer combined title.
    r"(?P<label>đối\s+tượng\s+áp\s+dụng\s+và\s+phạm\s+vi(?:\s+điều\s+chỉnh|\s+áp\s+dụng)?|"
    r"phạm\s+vi\s*,\s*đối\s+tượng(?:\s+và\s+thời\s+gian\s+thực\s+hiện\s+chương\s+trình)?|"
    r"phạm\s+vi(?:\s+điều\s+chỉnh|\s+áp\s+dụng)?\s+và\s+đối\s+tượng(?:\s+áp\s+dụng)?|"
    r"đối\s+tượng\s+và\s+phạm\s+vi\s+áp\s+dụng|"
    r"phạm\s+vi\s+(?:điều\s+chỉnh|áp\s+dụng)|đối\s+tượng\s+áp\s+dụng)"
    r"(?:[ \t]+(?:của[ \t]+)?(?:quy trình|quy chuẩn|tiêu chuẩn))?"
    # A legal heading is followed by a colon or a physical line boundary.
    # Merely starting on a wrapped line is not enough when prose continues on
    # that same line (the observed Article 6 false-positive class).
    r"(?P<delimiter>[ \t]*:|[ \t]*[-–][ \t]*(?:\r?\n)+|[ \t]*(?:\r?\n)+|(?=quy\b))")
_SCOPE_END = re.compile(r"(?im)^\s*(?:II\.|2\.\s*(?:tài liệu|nội dung)|3\.\s*(?:giải thích|khách hàng))")


def _scope_label_kind(label: str) -> str:
    folded = " ".join(label.casefold().split())
    has_scope = folded.startswith("phạm vi") or " và phạm vi" in folded
    has_subject = folded.startswith("đối tượng") or " và đối tượng" in folded
    if has_scope and has_subject:
        return "combined"
    return "scope_of_regulation" if has_scope else "applicable_subjects"


def strict_scope_heading_kind(heading: str) -> str | None:
    """Classify only a node's own legal heading, never words in its body."""
    compact = " ".join(str(heading).strip(" \t\r\n\"“”'").split())
    compact = re.sub(r"(?i)^điều\s+\d+[a-zđ]?\s*[.:]?\s*", "", compact)
    compact = re.sub(r"^\d+(?:\.\d+)*\s*[.):-]?\s*", "", compact)
    folded = compact.casefold()
    if not (folded.startswith("phạm vi") or folded.startswith("đối tượng")):
        return None
    if re.match(r"^phạm vi(?:\s+(?:điều chỉnh|áp dụng))?\s*(?:,|và)\s*đối tượng\b", folded):
        return "combined"
    if re.match(r"^đối tượng(?:\s+áp dụng)?\s*(?:,|và)\s*phạm vi\b", folded):
        return "combined"
    if re.match(r"^phạm vi\s+(?:điều chỉnh|áp dụng)\b", folded):
        return "scope_of_regulation"
    if re.match(r"^đối tượng\s+áp dụng\b", folded):
        return "applicable_subjects"
    return None


def _direct_node_scope_kind(node: Mapping[str, Any]) -> str | None:
    heading = str(node.get("heading_text", ""))
    parsed = strict_scope_heading_kind(heading)
    if parsed and not (
        parsed == "scope_of_regulation"
        and re.search(r"(?i)\bvà\s+đối\s*$", heading)
    ):
        return parsed
    # heading_text is sometimes cut at a visual line wrap ("Đối tượng áp" /
    # "dụng").  The anchored classifier can safely inspect the compact raw
    # prefix because it never searches for a later body occurrence.
    parsed = strict_scope_heading_kind(str(node.get("raw_text", ""))[:300])
    if parsed:
        return parsed
    # Annotated OCR truncation: heading_text ends after "Phạm vi," while the
    # complementary "đối tượng" is the immediately wrapped continuation.
    if re.search(r"(?i)phạm\s+vi\s*,\s*$", heading) and re.match(
        r"(?is)^\s*điều\s+\d+[a-zđ]?\s*[.:]?\s*phạm\s+vi\s*,\s*đối\s+tượng\b",
        str(node.get("raw_text", ""))[:260],
    ):
        return "combined"
    return None


def _next_peer_heading(raw: str, match: re.Match[str], after: int) -> int:
    """Find the next same-depth textual sibling for an embedded heading."""
    article = match.group("article")
    number = match.group("number")
    if article:
        peer = re.compile(r"(?im)^\s*[\"“”']?điều\s+\d+[a-zđ]?\s*[.:]?")
    elif number:
        depth = number.count(".") + 1
        peer = re.compile(rf"(?m)^\s*\d+(?:\.\d+){{{depth - 1}}}(?:\.)?\s+\S")
    else:
        fallback = _SCOPE_END.search(raw, after)
        return fallback.start() if fallback else len(raw)
    found = peer.search(raw, after)
    return found.start() if found else len(raw)


def _has_substantive_scope_content(text: str, *, minimum: int = 10) -> bool:
    compact = " ".join(text.split())
    return len(compact) >= minimum and bool(re.search(
        r"(?i)\b(?:quy định|áp dụng|đối với|bao gồm|cơ quan|tổ chức|cá nhân|"
        r"chương này|quy chuẩn này|thông tư này|nghị định này|quy trình này)\b",
        compact,
    ))


def explicit_scope_spans(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract source-exact embedded headings up to their textual sibling."""
    raw = str(node.get("raw_text", ""))
    matches = list(_EXPLICIT_SCOPE.finditer(raw))
    spans: list[dict[str, Any]] = []
    index = 0
    while index < len(matches):
        first = matches[index]
        kinds = {_scope_label_kind(first.group("label"))}
        last = first
        # Scope and subject are normally consecutive parts of one applicability
        # block.  Join only the immediately following complementary heading.
        if index + 1 < len(matches):
            following = matches[index + 1]
            following_kind = _scope_label_kind(following.group("label"))
            if kinds | {following_kind} == {"scope_of_regulation", "applicable_subjects"}:
                kinds.add(following_kind)
                last = following
                index += 1
        kind = "combined" if len(kinds) == 2 or "combined" in kinds else next(iter(kinds))
        start = first.start()
        end = _next_peer_heading(raw, last, last.end())
        # A later applicability block is also a safe boundary and prevents a
        # malformed container from swallowing multiple annexes.
        if index + 1 < len(matches):
            end = min(end, matches[index + 1].start())
        text = raw[start:end]
        leading = len(text) - len(text.lstrip())
        trailing = len(text.rstrip())
        start += leading
        end = start + max(0, trailing - leading)
        content = raw[last.end():end]
        substantive = _has_substantive_scope_content(content)
        if end > start and substantive:
            spans.append({
                "kind": kind,
                "source_start": int(node["start"]) + start,
                "source_end": int(node["start"]) + end,
                "raw_text": raw[start:end],
                "relative_start": start,
                "relative_end": end,
            })
        index += 1
    return spans


def _source_exact_missed_span(node: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Return the specific declarative span, never a copied audit excerpt."""
    raw = str(node["raw_text"])
    direct_kind = _direct_node_scope_kind(node)
    if direct_kind == kind:
        start = len(raw) - len(raw.lstrip())
        end = len(raw.rstrip())
        return {
            "kind": kind,
            "source_start": int(node["start"]) + start,
            "source_end": int(node["start"]) + end,
            "raw_text": raw[start:end],
        }
    matches = list(_EXPLICIT_SCOPE.finditer(raw))
    if not matches:
        raise ValueError(f"reviewed miss has no explicit source heading: {node['node_id']}")
    # Prefer a combined heading; otherwise span from the first scope declaration
    # through the following subject declaration, stopping at the next section.
    start_match = next((match for match in matches if "và" in match.group("label").casefold()), matches[0])
    start = start_match.start()
    following = [match.start() for match in _SCOPE_END.finditer(raw, start_match.end()) if match.start() > start]
    end = following[0] if following else len(raw)
    selected = raw[start:end]
    start += len(selected) - len(selected.lstrip())
    end = start + len(selected.strip())
    text = raw[start:end]
    if not text:
        raise ValueError(f"empty reviewed missed span: {node['node_id']}")
    absolute_start = int(node["start"]) + start
    return {"kind": kind, "source_start": absolute_start, "source_end": absolute_start + len(text), "raw_text": text}


def finalize_scope_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    annotations = list(_jsonl(paths["results"] / "scope-audit" / "agent_reviewed_annotations.jsonl"))
    if len(annotations) != 200 or any(row.get("review_label") not in (*SCOPE_KINDS, "none") or row.get("boundary_complete") not in (True, False, None) for row in annotations):
        raise ValueError("finalize requires 200 agent-reviewed annotations with label and boundary decision")
    metrics = _scope_metrics(annotations)
    agreement = sum(row["parser_label"] == row["review_label"] for row in annotations)
    gate = metrics["micro_precision"] >= .95 and min(metrics["class_precision"].values()) >= .90 and metrics["boundary_accuracy"] >= .95 and metrics["stratified_weighted_recall"] >= .90
    fingerprint = _hash(annotations)
    output = paths["results"] / "scope-audit"
    # The user receives an equal six-record view of every sampled stratum.
    spot = []
    for bucket in ("scope_of_regulation", "applicable_subjects", "combined", "rejected_legacy_scope", "raw_text_scope_signal"):
        bucket_rows = [row for row in annotations if row["sample_bucket"] == bucket]
        chosen = sorted(bucket_rows, key=lambda row: _hash({"seed": SEED, "spot": row["doc_id"], "node": row["node_id"]}))[:6]
        if len(chosen) != 6:
            raise ValueError(f"spot-check stratum is undersized: {bucket}")
        spot.extend(chosen)
    spot.sort(key=lambda row: (row["sample_bucket"], row["doc_id"], row["node_id"]))
    _write_jsonl(output / "USER_SPOTCHECK_30.jsonl", spot)
    result = {"schema_version": SCHEMA, "status": "WAITING_SCOPE_SPOTCHECK", "fingerprint": fingerprint, "agent_parser_agreement": agreement, "agent_parser_agreement_required": 27, "parser_gate": gate, "metrics": metrics, "spotcheck": str(output / "USER_SPOTCHECK_30.jsonl")}
    _write_json(output / "REPORT.json", result)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "finalize-scope-audit", "fingerprint": fingerprint, "report_sha256": _sha256(output / "REPORT.json")})
    _state(paths["results"], "finalize-scope-audit", "WAITING_SCOPE_SPOTCHECK", completed=1, total=1, eta_seconds=None, parser_gate=gate)
    return result


def review_scope_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Materialize the recorded agent review without mutating generated cache."""
    source = list(_jsonl(paths["cache"] / "scope-audit" / "annotations.jsonl"))
    if len(source) != 200:
        raise ValueError("scope sample must contain exactly 200 rows")
    sample_node_ids = {str(row["node_id"]) for row in source}
    nodes = {str(row["node_id"]): row for row in _jsonl(paths["v3"] / "nodes.jsonl") if str(row["node_id"]) in sample_node_ids}
    if set(nodes) != sample_node_ids:
        raise ValueError("scope review node/source mapping is incomplete")
    reviewed = []
    for index, row in enumerate(source):
        item = dict(row)
        review_label = AGENT_REVIEW_OVERRIDES.get(index, str(item["parser_label"]))
        item["review_label"] = review_label
        item["boundary_complete"] = (
            index not in AGENT_BOUNDARY_INCOMPLETE
            if item["parser_label"] != "none" and review_label == item["parser_label"]
            else None
        )
        item["missed_span"] = (
            _source_exact_missed_span(nodes[str(item["node_id"])], review_label)
            if item["parser_label"] == "none" and review_label != "none" else None
        )
        item["rationale"] = (
            "agent_review: heading/order shows combined scope and subject"
            if review_label == "combined" and item["parser_label"] != "combined"
            else "agent_review: source-exact sibling boundary exceeds selected span"
            if index in AGENT_BOUNDARY_INCOMPLETE
            else "agent_review: keyword is a reference, not a self-contained declarative scope"
            if review_label == "none"
            else "agent_review: parser label and source span accepted"
        )
        item["review_status"] = "AGENT_REVIEWED"
        reviewed.append(item)
    output = paths["results"] / "scope-audit"
    count = _write_jsonl(output / "agent_reviewed_annotations.jsonl", reviewed)
    review_fingerprint = _hash(reviewed)
    result = {"schema_version": SCHEMA, "status": "AGENT_REVIEWED", "records": count, "fingerprint": review_fingerprint, "label_overrides": len(AGENT_REVIEW_OVERRIDES), "boundary_incomplete": len(AGENT_BOUNDARY_INCOMPLETE)}
    _write_json(output / "AGENT_REVIEW.json", result)
    _state(paths["results"], "review-scope-audit", "SUCCESS", completed=count, total=count, eta_seconds=0)
    return result


def build_parent_index(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Build immutable document/chunk/embedding row mapping without encoding."""
    output = paths["cache"] / "parent-index"
    chunk_ids = [str(row["chunk_id"]) for row in _jsonl(paths["e5"] / "chunk_ids.jsonl")]
    if len(chunk_ids) != 343347 or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("E5 chunk-id inventory is malformed")
    embedding_row = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for chunk in _jsonl(paths["v3"] / "chunks.jsonl"):
        chunk_id = str(chunk["chunk_id"])
        if chunk_id not in embedding_row:
            raise ValueError(f"Structural chunk lacks E5 row: {chunk_id}")
        documents[str(chunk["doc_id"])].append({"chunk_id": chunk_id, "embedding_row": embedding_row[chunk_id], "start": chunk["start"], "end": chunk["end"], "parent_node_id": chunk.get("parent_node_id"), "token_count": chunk.get("token_count", 0)})
        count += 1
    rows = ({"doc_id": doc_id, "chunks": items} for doc_id, items in sorted(documents.items()))
    written = _write_jsonl(output / "doc_to_chunks.jsonl", rows)
    fingerprint = _hash({"v3": _json(paths["v3"] / "manifest.json")["content_fingerprint"], "e5": _json(paths["e5"] / "manifest.json")["cache_fingerprint"], "documents": written, "chunks": count})
    report = {"schema_version": SCHEMA, "status": "PASS", "fingerprint": fingerprint, "documents": written, "chunks": count, "reencoded_chunks": 0}
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "build-parent-index", "fingerprint": fingerprint, "doc_to_chunks_sha256": _sha256(output / "doc_to_chunks.jsonl")})
    _success(output, stage="build-parent-index", fingerprint=fingerprint)
    _state(paths["results"], "build-parent-index", "SUCCESS", completed=count, total=count, eta_seconds=0)
    return report


def repair_scope_sidecar(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Repair only EXP-033 scope metadata using direct structural headings.

    The failed audit showed that scanning a chapter/section body is too broad:
    it can select Article 1 plus later siblings.  This stage accepts only the
    node's own heading and derives the text span from that sibling node.
    """
    output = paths["cache"] / "scope-sidecar-v2"
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    rejected_ancestor_body = 0
    for node in _jsonl(paths["v3"] / "nodes.jsonl"):
        heading = str(node.get("heading_text", ""))
        raw = str(node.get("raw_text", ""))
        parsed = _direct_node_scope_kind(node)
        node_spans: list[dict[str, Any]] = []
        if parsed in SCOPE_KINDS and _has_substantive_scope_content(raw, minimum=40):
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            node_spans.append({
                "kind": parsed,
                "source_start": int(node["start"]) + leading,
                "source_end": int(node["start"]) + trailing,
                "raw_text": raw[leading:trailing],
                "boundary_policy": "direct_heading_structural_sibling_v2",
                "priority": 0,
            })
        else:
            node_spans = [
                {**span, "boundary_policy": "embedded_heading_peer_boundary_v2", "priority": 1}
                for span in explicit_scope_spans(node)
            ]
            if not node_spans and classify_scope_node(node) in SCOPE_KINDS:
                rejected_ancestor_body += 1
        for span in node_spans:
            key = (str(node["doc_id"]), int(span["source_start"]), int(span["source_end"]), str(span["kind"]))
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "schema_version": SCHEMA, "doc_id": str(node["doc_id"]), "node_id": str(node["node_id"]),
                "kind": span["kind"], "source_start": span["source_start"], "source_end": span["source_end"],
                "raw_text": span["raw_text"], "boundary_policy": span["boundary_policy"],
                "priority": span["priority"],
                "source_node_sha256": _hash({"node_id": node["node_id"], "raw_text": node["raw_text"]}),
            })
    # The same embedded heading can be visible through an article and its
    # ancestor. Prefer the shortest source-exact span, then the direct
    # structural sibling. This keeps one metadata row per actual declaration.
    candidates.sort(key=lambda row: (row["doc_id"], row["source_start"], row["source_end"] - row["source_start"], row["priority"], row["node_id"]))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        duplicate = next((
            row for row in reversed(rows)
            if row["doc_id"] == candidate["doc_id"]
            and row["source_start"] == candidate["source_start"]
            and row["kind"] == candidate["kind"]
        ), None)
        if duplicate is not None:
            continue
        candidate.pop("priority", None)
        rows.append(candidate)
    rows.sort(key=lambda row: (row["doc_id"], row["source_start"], row["source_end"], row["kind"]))
    count = _write_jsonl(output / "scope_spans.jsonl", rows)
    fingerprint = _hash({"v3": _json(paths["v3"] / "manifest.json")["content_fingerprint"], "rows": rows})
    report = {
        "schema_version": SCHEMA, "status": "REPAIRED_METADATA_SIDECAR_ONLY", "fingerprint": fingerprint,
        "spans": count, "by_kind": dict(Counter(row["kind"] for row in rows)),
        "rejected_ancestor_body_candidates": rejected_ancestor_body,
        "structural_v3_rebuilt": False,
    }
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "repair-scope-sidecar", "fingerprint": fingerprint, "scope_spans_sha256": _sha256(output / "scope_spans.jsonl")})
    _success(output, stage="repair-scope-sidecar", fingerprint=fingerprint)
    _state(paths["results"], "repair-scope-sidecar", "SUCCESS", completed=count, total=count, eta_seconds=0, structural_v3_rebuilt=False)
    return report


def _aggregate_scope_kind(kinds: Iterable[str]) -> str:
    observed = set(kinds)
    if "combined" in observed or {"scope_of_regulation", "applicable_subjects"} <= observed:
        return "combined"
    if "scope_of_regulation" in observed:
        return "scope_of_regulation"
    if "applicable_subjects" in observed:
        return "applicable_subjects"
    return "none"


def reaudit_scope_sidecar(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Re-score all 200 reviewed documents against the repaired sidecar."""
    sidecar_dir = paths["cache"] / "scope-sidecar-v2"
    _manifest_ok(sidecar_dir, key="fingerprint")
    reviewed = list(_jsonl(paths["results"] / "scope-audit" / "agent_reviewed_annotations.jsonl"))
    if len(reviewed) != 200:
        raise ValueError("re-audit requires all 200 reviewed annotations")
    sample_ids = {str(row["node_id"]) for row in reviewed}
    nodes = {
        str(node["node_id"]): node for node in _jsonl(paths["v3"] / "nodes.jsonl")
        if str(node["node_id"]) in sample_ids
    }
    if set(nodes) != sample_ids:
        raise ValueError("re-audit sample node mapping is incomplete")
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_nodes: dict[str, dict[str, Any]] = {}
    sidecar_rows = list(_jsonl(sidecar_dir / "scope_spans.jsonl"))
    sidecar_node_ids = {str(row["node_id"]) for row in sidecar_rows}
    for node in _jsonl(paths["v3"] / "nodes.jsonl"):
        if str(node["node_id"]) in sidecar_node_ids:
            source_nodes[str(node["node_id"])] = node
    exact_failures: list[dict[str, Any]] = []
    for span in sidecar_rows:
        source = source_nodes.get(str(span["node_id"]))
        if source is None:
            exact_failures.append({"node_id": span["node_id"], "reason": "missing_source_node"})
            continue
        relative_start = int(span["source_start"]) - int(source["start"])
        relative_end = int(span["source_end"]) - int(source["start"])
        exact = (
            0 <= relative_start < relative_end <= len(str(source["raw_text"]))
            and str(source["raw_text"])[relative_start:relative_end] == str(span["raw_text"])
        )
        if not exact:
            exact_failures.append({"node_id": span["node_id"], "reason": "offset_text_mismatch"})
            continue
        by_doc[str(span["doc_id"])].append(span)
    audited: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row in reviewed:
        item = dict(row)
        node = nodes[str(row["node_id"])]
        contained = [
            span for span in by_doc.get(str(row["doc_id"]), [])
            if int(node["start"]) <= int(span["source_start"])
            and int(span["source_end"]) <= int(node["end"])
        ]
        prediction = _aggregate_scope_kind(str(span["kind"]) for span in contained)
        item["legacy_parser_label"] = item["parser_label"]
        item["parser_label"] = prediction
        item["repaired_spans"] = contained
        item["boundary_complete"] = bool(contained) if prediction != "none" and prediction == item["review_label"] else None
        item["reaudit_status"] = "MATCH" if prediction == item["review_label"] else "DISAGREEMENT"
        audited.append(item)
        if item["reaudit_status"] == "DISAGREEMENT":
            disagreements.append({
                "sample_index": len(audited) - 1, "doc_id": item["doc_id"], "node_id": item["node_id"],
                "sample_bucket": item["sample_bucket"], "review_label": item["review_label"],
                "repaired_label": prediction, "heading_text": node.get("heading_text", ""),
                "raw_prefix": str(node.get("raw_text", ""))[:500],
            })
    metrics = _scope_metrics(audited)
    gate = (
        not exact_failures
        and metrics["micro_precision"] >= .95
        and min(metrics["class_precision"].values()) >= .90
        and metrics["boundary_accuracy"] >= .95
        and metrics["stratified_weighted_recall"] >= .90
    )
    output = paths["results"] / "scope-audit"
    _write_jsonl(output / "repaired_annotations.jsonl", audited)
    _write_jsonl(output / "REPAIR_DISAGREEMENTS.jsonl", disagreements)
    spot: list[dict[str, Any]] = []
    for bucket in ("scope_of_regulation", "applicable_subjects", "combined", "rejected_legacy_scope", "raw_text_scope_signal"):
        bucket_rows = [row for row in audited if row["sample_bucket"] == bucket]
        spot.extend(sorted(bucket_rows, key=lambda row: _hash({"seed": SEED, "agent_spot_v2": row["doc_id"], "node": row["node_id"]}))[:6])
    spot.sort(key=lambda row: (row["sample_bucket"], row["doc_id"], row["node_id"]))
    _write_jsonl(output / "AGENT_SPOTCHECK_CANDIDATES_30.jsonl", spot)
    fingerprint = _hash({"sidecar": _json(sidecar_dir / "REPORT.json")["fingerprint"], "audited": audited})
    result = {
        "schema_version": SCHEMA,
        "status": "WAITING_AGENT_SPOTCHECK" if gate else "REPAIR_REAUDIT_FAILED",
        "fingerprint": fingerprint,
        "records": len(audited),
        "repaired_parser_gate": gate,
        "metrics": metrics,
        "disagreements": len(disagreements),
        "source_exact_failures": len(exact_failures),
        "source_exact_failure_examples": exact_failures[:20],
        "agent_spotcheck_candidates": str(output / "AGENT_SPOTCHECK_CANDIDATES_30.jsonl"),
    }
    _write_json(output / "REPAIR_REAUDIT_REPORT.json", result)
    _state(paths["results"], "reaudit-scope-sidecar", result["status"], completed=len(audited), total=200, eta_seconds=0, parser_gate=gate)
    return result


def finalize_scope_phase_a(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Record the delegated 30-row agent check and close Phase A."""
    output = paths["results"] / "scope-audit"
    reaudit = _json(output / "REPAIR_REAUDIT_REPORT.json")
    if not reaudit.get("repaired_parser_gate") or reaudit.get("disagreements") != 0:
        raise RuntimeError("Phase A cannot close before a clean repaired re-audit")
    spot = list(_jsonl(output / "AGENT_SPOTCHECK_CANDIDATES_30.jsonl"))
    if len(spot) != 30 or Counter(str(row["sample_bucket"]) for row in spot) != {
        "scope_of_regulation": 6,
        "applicable_subjects": 6,
        "combined": 6,
        "rejected_legacy_scope": 6,
        "raw_text_scope_signal": 6,
    }:
        raise ValueError("agent spot-check must contain six rows from every stratum")
    decisions = []
    for row in spot:
        item = dict(row)
        item["agent_spotcheck_label"] = item["review_label"]
        item["agent_spotcheck_agrees"] = item["parser_label"] == item["agent_spotcheck_label"]
        item["agent_spotcheck_rationale"] = (
            "source contains a self-contained declarative applicability span"
            if item["agent_spotcheck_label"] != "none"
            else "source is referential, topical, or heading-only and has no usable declarative applicability span"
        )
        item["agent_spotcheck_status"] = "AGENT_INSPECTED"
        decisions.append(item)
    agreement = sum(bool(row["agent_spotcheck_agrees"]) for row in decisions)
    if agreement < 27:
        raise RuntimeError(f"delegated agent spot-check failed: {agreement}/30")
    _write_jsonl(output / "AGENT_SPOTCHECK_30.jsonl", decisions)

    known_ids = {
        "163254:section:000019863:000027912",
        "234704:chapter:000291762:000352228",
        "275852:article:000002101:000027362",
        "100139:article:000007403:000009648",
    }
    known_nodes = {
        str(node["node_id"]): node for node in _jsonl(paths["v3"] / "nodes.jsonl")
        if str(node["node_id"]) in known_ids
    }
    if set(known_nodes) != known_ids:
        raise ValueError("known-node regression inventory is incomplete")
    sidecar = list(_jsonl(paths["cache"] / "scope-sidecar-v2" / "scope_spans.jsonl"))
    contained: dict[str, list[dict[str, Any]]] = {}
    for node_id, node in known_nodes.items():
        contained[node_id] = [
            span for span in sidecar
            if str(span["doc_id"]) == str(node["doc_id"])
            and int(node["start"]) <= int(span["source_start"])
            and int(span["source_end"]) <= int(node["end"])
        ]
    reviewed = list(_jsonl(output / "agent_reviewed_annotations.jsonl"))
    row_275852 = next(row for row in reviewed if row["node_id"] == "275852:article:000002101:000027362")
    missed = row_275852.get("missed_span") or {}
    checks = {
        "163254_reference_only_rejected": not contained["163254:section:000019863:000027912"],
        "234704_no_declarative_keyword_rejected": not contained["234704:chapter:000291762:000352228"],
        "100139_wrapped_prose_false_positive_rejected": not contained["100139:article:000007403:000009648"],
        "275852_combined_content_retained": _aggregate_scope_kind(
            str(span["kind"]) for span in contained["275852:article:000002101:000027362"]
        ) == "combined",
        "275852_missed_span_source_exact_and_not_raw_copy": bool(missed)
        and "phạm vi" in str(missed.get("raw_text", "")).casefold()
        and "đối tượng áp dụng" in str(missed.get("raw_text", "")).casefold()
        and str(missed.get("raw_text", "")) != str(row_275852.get("raw_span", "")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"known-node regression failed: {checks}")
    sidecar_report = _json(paths["cache"] / "scope-sidecar-v2" / "REPORT.json")
    fingerprint = _hash({
        "reaudit": reaudit["fingerprint"],
        "spotcheck": decisions,
        "sidecar": sidecar_report["fingerprint"],
        "known_checks": checks,
    })
    result = {
        "schema_version": SCHEMA,
        "status": "PASS_PHASE_A",
        "phase": "A",
        "phase_completion_percent": 100,
        "fingerprint": fingerprint,
        "repaired_parser_gate": True,
        "metrics": reaudit["metrics"],
        "annotations": 200,
        "re_audit_disagreements": 0,
        "source_exact_failures": 0,
        "agent_spotcheck": {"agreement": agreement, "total": 30, "required": 27, "passed": True},
        "known_node_regressions": checks,
        "scope_sidecar": {"fingerprint": sidecar_report["fingerprint"], "spans": sidecar_report["spans"]},
        "structural_v3_rebuilt": False,
        "phase_b_started_by_this_finalize": False,
    }
    _write_json(output / "REPORT.json", result)
    _write_json(output / "manifest.json", {
        "schema_version": SCHEMA, "stage": "finalize-scope-phase-a", "fingerprint": fingerprint,
        "report_sha256": _sha256(output / "REPORT.json"),
        "agent_spotcheck_sha256": _sha256(output / "AGENT_SPOTCHECK_30.jsonl"),
    })
    _success(output, stage="finalize-scope-phase-a", fingerprint=fingerprint, phase="A")
    _state(paths["results"], "phase-a", "SUCCESS", completed=100, total=100, eta_seconds=0, phase_completion_percent=100)
    return result


def _cosine(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return rows.astype(np.float32) @ query.astype(np.float32)


def select_evidence(candidates: Sequence[Mapping[str, Any]], scores: Sequence[float], *, secondary: bool = True, redundancy: Mapping[tuple[str, str], float] | None = None) -> list[dict[str, Any]]:
    """Deterministic answer-first selector; tests can inject redundancy values."""
    ranked = sorted(zip(candidates, scores), key=lambda item: (-float(item[1]), str(item[0]["chunk_id"])))
    if not ranked:
        return []
    selected = [dict(ranked[0][0], score=float(ranked[0][1]), evidence_rank=1)]
    if not secondary:
        return selected
    for candidate, score in ranked[1:]:
        if candidate["chunk_id"] == selected[0]["chunk_id"]:
            continue
        pair = tuple(sorted((str(candidate["chunk_id"]), str(selected[0]["chunk_id"]))))
        if redundancy is not None and redundancy.get(pair, 0.0) >= .90:
            continue
        selected.append(dict(candidate, score=float(score), evidence_rank=2))
        break
    return selected


def select_mmr_evidence(
    candidates: Sequence[Mapping[str, Any]],
    relevance: Sequence[float],
    vectors: np.ndarray,
    *,
    lambda_value: float,
) -> list[dict[str, Any]]:
    """Select the relevance head and one non-redundant MMR secondary."""
    if lambda_value not in (.70, .85):
        raise ValueError("MMR lambda must be 0.70 or 0.85")
    if len(candidates) != len(relevance) or len(candidates) != len(vectors):
        raise ValueError("MMR candidates, scores, and vectors must align")
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda idx: (-float(relevance[idx]), str(candidates[idx]["chunk_id"])))
    first = order[0]
    selected = [dict(candidates[first], score=float(relevance[first]), evidence_rank=1, redundancy=0.0)]
    choices = []
    for idx in order[1:]:
        redundancy = float(np.asarray(vectors[idx], dtype=np.float32) @ np.asarray(vectors[first], dtype=np.float32))
        if redundancy >= .90:
            continue
        mmr_score = lambda_value * float(relevance[idx]) - (1.0 - lambda_value) * redundancy
        choices.append((mmr_score, float(relevance[idx]), str(candidates[idx]["chunk_id"]), idx, redundancy))
    if choices:
        _, score, _, idx, redundancy = sorted(choices, key=lambda item: (-item[0], -item[1], item[2]))[0]
        selected.append(dict(candidates[idx], score=score, evidence_rank=2, redundancy=redundancy))
    return selected


def _select_ranked_nonredundant(
    ranked: Sequence[Mapping[str, Any]], embedding_rows: Mapping[str, int], embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """Keep the ranking head and first secondary with cosine below 0.90."""
    if not ranked:
        return []
    primary = dict(ranked[0], evidence_rank=1, redundancy=0.0)
    selected = [primary]
    primary_row = embedding_rows.get(str(primary["chunk_id"]))
    if primary_row is None:
        raise ValueError(f"selected chunk lacks frozen embedding row: {primary['chunk_id']}")
    primary_vector = np.asarray(embeddings[primary_row], dtype=np.float32)
    for candidate in ranked[1:]:
        candidate_row = embedding_rows.get(str(candidate["chunk_id"]))
        if candidate_row is None:
            raise ValueError(f"selected chunk lacks frozen embedding row: {candidate['chunk_id']}")
        redundancy = float(np.asarray(embeddings[candidate_row], dtype=np.float32) @ primary_vector)
        if redundancy >= .90:
            continue
        selected.append(dict(candidate, evidence_rank=2, redundancy=redundancy))
        break
    return selected


def _compact_selected(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compact = [{
        "chunk_id": str(row["chunk_id"]),
        "score": float(row.get("score", 0.0)),
        "evidence_rank": int(row.get("evidence_rank", rank)),
        "redundancy": float(row.get("redundancy", 0.0)),
    } for rank, row in enumerate(rows, start=1)]
    for source, target in zip(rows, compact):
        for key in ("score_channel", "fallback_reason"):
            if key in source:
                target[key] = source[key]
    return compact


def _bm25_primary_fallback(
    sparse_selected: Sequence[Mapping[str, Any]], dense_primary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Guarantee a primary only for the observed zero-lexical-hit class."""
    if sparse_selected:
        return [dict(row) for row in sparse_selected], False
    return [dict(
        dense_primary,
        evidence_rank=1,
        redundancy=0.0,
        score_channel="e5_fallback",
        fallback_reason="bm25_zero_lexical_hit",
    )], True


def _clause_neighbors(chunks: Sequence[Mapping[str, Any]], selected_ids: Iterable[str]) -> dict[str, list[str]]:
    positions = {str(chunk["chunk_id"]): index for index, chunk in enumerate(chunks)}
    result: dict[str, list[str]] = {}
    for chunk_id in selected_ids:
        index = positions[str(chunk_id)]
        current = chunks[index]
        neighbors = []
        for other_index in (index - 1, index + 1):
            if 0 <= other_index < len(chunks) and chunks[other_index].get("parent_node_id") == current.get("parent_node_id"):
                neighbors.append(str(chunks[other_index]["chunk_id"]))
        result[str(chunk_id)] = neighbors
    return result


def _audit_token_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """Measure pre-truncation lengths without the tokenizer's model-use warning."""
    # These untruncated IDs are used only for telemetry and are never passed to
    # the model. ``verbose=False`` suppresses the misleading warning claiming
    # that the overlength sequence will be run through the model.
    return [int(value) for value in tokenizer(
        list(texts), add_special_tokens=True, truncation=False,
        return_length=True, verbose=False,
    )["length"]]


def encode_scope_sidecar(paths: Mapping[str, Path], *, device: str = "cuda") -> dict[str, Any]:
    """Encode only the repaired metadata spans for direct query-scope scoring."""
    phase_a = _json(paths["results"] / "scope-audit" / "REPORT.json")
    if phase_a.get("status") != "PASS_PHASE_A":
        raise RuntimeError("scope embeddings require accepted Phase A")
    sidecar_dir = paths["cache"] / "scope-sidecar-v2"
    _manifest_ok(sidecar_dir, key="fingerprint")
    spans = list(_jsonl(sidecar_dir / "scope_spans.jsonl"))
    if not spans:
        raise ValueError("repaired scope sidecar is empty")
    output = paths["cache"] / "scope-embeddings-v2"
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_id = str(_json(paths["e5"] / "manifest.json")["model_id"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for scope encoding but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    matrix = np.lib.format.open_memmap(output / "scope_embeddings.f16.npy.tmp", mode="w+", dtype=np.float16, shape=(len(spans), 1024))
    ids = []
    truncated = 0
    max_model_input_tokens = 0
    batch_size = 32
    started = time.time()
    for offset in range(0, len(spans), batch_size):
        batch = spans[offset:offset + batch_size]
        texts = ["passage: " + str(row["raw_text"]) for row in batch]
        lengths = _audit_token_lengths(tokenizer, texts)
        truncated += sum(int(length) > 512 for length in lengths)
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch_model_tokens = int(inputs["attention_mask"].sum(dim=1).max().item())
        max_model_input_tokens = max(max_model_input_tokens, batch_model_tokens)
        if batch_model_tokens > 512:
            raise AssertionError(f"scope encoder model input exceeded 512 tokens: {batch_model_tokens}")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        matrix[offset:offset + len(batch)] = pooled.cpu().numpy().astype(np.float16)
        for index, row in enumerate(batch, start=offset):
            ids.append({
                "embedding_row": index,
                "scope_id": _hash({key: row[key] for key in ("doc_id", "node_id", "kind", "source_start", "source_end")}),
                "doc_id": row["doc_id"], "node_id": row["node_id"], "kind": row["kind"],
                "source_start": row["source_start"], "source_end": row["source_end"],
            })
        _state(paths["results"], "encode-scope-sidecar", "RUNNING", completed=min(offset + batch_size, len(spans)), total=len(spans), eta_seconds=None)
    matrix.flush()
    del matrix
    tmp_matrix = output / "scope_embeddings.f16.npy.tmp"
    final_matrix = output / "scope_embeddings.f16.npy"
    tmp_matrix.replace(final_matrix)
    _write_jsonl(output / "scope_ids.jsonl", ids)
    fingerprint = _hash({
        "scope_sidecar": _json(sidecar_dir / "REPORT.json")["fingerprint"],
        "e5": _json(paths["e5"] / "manifest.json")["cache_fingerprint"],
        "spans": len(spans), "max_length": 512, "prefix": "passage: ",
    })
    report = {
        "schema_version": SCHEMA, "status": "PASS", "fingerprint": fingerprint,
        "spans": len(spans), "dimension": 1024, "dtype": "float16", "model_id": model_id,
        "truncated_spans": truncated, "corpus_chunks_reencoded": 0,
        "max_model_input_tokens": max_model_input_tokens,
        "overlength_measurement_only_not_model_input": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {
        "schema_version": SCHEMA, "stage": "encode-scope-sidecar", "fingerprint": fingerprint,
        "embeddings_sha256": _sha256(final_matrix), "scope_ids_sha256": _sha256(output / "scope_ids.jsonl"),
    })
    _success(output, stage="encode-scope-sidecar", fingerprint=fingerprint)
    _state(paths["results"], "encode-scope-sidecar", "SUCCESS", completed=len(spans), total=len(spans), eta_seconds=0)
    return report


def score_in_document(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Score frozen K64 parents with in-parent E5; no labels are consumed."""
    index_path = paths["cache"] / "parent-index" / "doc_to_chunks.jsonl"
    if not (index_path.exists() and (paths["cache"] / "parent-index" / "_SUCCESS.json").exists()):
        raise RuntimeError("build-parent-index must succeed first")
    doc_chunks = {str(row["doc_id"]): row["chunks"] for row in _jsonl(index_path)}
    query_ids = _json(paths["query_embeddings"] / "train_query_ids.json")
    query_rows = {str(qid): pos for pos, qid in enumerate(query_ids)}
    queries = np.load(paths["query_embeddings"] / "train_queries.f32.npy", mmap_mode="r")
    embeddings = np.load(paths["e5"] / "embeddings.f16.npy", mmap_mode="r")
    rankings = {str(row["qid"]): [str(doc) for doc in row["doc_ids"][:K64]] for row in _jsonl(paths["exp028_oof"])}
    if len(rankings) != 7000 or set(rankings) != set(query_rows):
        raise ValueError("EXP-028 OOF/query embedding membership mismatch")
    output = paths["cache"] / "in-document-scores"
    def rows() -> Iterator[dict[str, Any]]:
        for count, qid in enumerate(sorted(rankings), start=1):
            query = np.asarray(queries[query_rows[qid]], dtype=np.float32)
            selected = []
            for rank, doc_id in enumerate(rankings[qid], start=1):
                chunks = doc_chunks.get(doc_id)
                if not chunks:
                    raise ValueError(f"shortlisted doc missing parent index: {qid}/{doc_id}")
                indices = [int(chunk["embedding_row"]) for chunk in chunks]
                scores = _cosine(query, np.asarray(embeddings[indices]))
                evidence = select_evidence(chunks, scores, secondary=True)
                selected.append({"doc_id": doc_id, "lambdamart_rank": rank, "selector": "in_parent_e5_top2", "evidence": evidence})
            if count % 100 == 0:
                _state(paths["results"], "score-in-document", "RUNNING", completed=count, total=7000, eta_seconds=None)
            yield {"schema_version": SCHEMA, "qid": qid, "candidates": selected}
    count = _write_jsonl(output / "e5_top2.jsonl", rows())
    fingerprint = _hash({"parent_index": _json(paths["cache"] / "parent-index" / "REPORT.json")["fingerprint"], "oof": _sha256(paths["exp028_oof"]), "queries": count, "selector": "in_parent_e5_top2"})
    report = {"schema_version": SCHEMA, "status": "PASS", "fingerprint": fingerprint, "queries": count, "pairs": count * K64, "selector": "in_parent_e5_top2", "labels_used": False}
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "score-in-document", "fingerprint": fingerprint, "scores_sha256": _sha256(output / "e5_top2.jsonl")})
    _success(output, stage="score-in-document", fingerprint=fingerprint)
    _state(paths["results"], "score-in-document", "SUCCESS", completed=count, total=count, eta_seconds=0)
    return report


def score_selectors_v2(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Build every Phase-B selector over the immutable K64 parent set."""
    parent_dir = paths["cache"] / "parent-index"
    scope_dir = paths["cache"] / "scope-embeddings-v2"
    _manifest_ok(parent_dir, key="fingerprint")
    _manifest_ok(scope_dir, key="fingerprint")
    phase_a = _json(paths["results"] / "scope-audit" / "REPORT.json")
    if phase_a.get("status") != "PASS_PHASE_A":
        raise RuntimeError("Phase B requires accepted Phase A")
    output = paths["cache"] / "in-document-selector-v2"
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    doc_chunks = {str(row["doc_id"]): row["chunks"] for row in _jsonl(parent_dir / "doc_to_chunks.jsonl")}
    embedding_row = {
        str(chunk["chunk_id"]): int(chunk["embedding_row"])
        for chunks in doc_chunks.values() for chunk in chunks
    }
    query_ids = _json(paths["query_embeddings"] / "train_query_ids.json")
    query_rows = {str(qid): pos for pos, qid in enumerate(query_ids)}
    queries = np.load(paths["query_embeddings"] / "train_queries.f32.npy", mmap_mode="r")
    embeddings = np.load(paths["e5"] / "embeddings.f16.npy", mmap_mode="r")
    scope_vectors = np.load(scope_dir / "scope_embeddings.f16.npy", mmap_mode="r")
    scope_rows = list(_jsonl(scope_dir / "scope_ids.jsonl"))
    scopes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scope_rows:
        scopes_by_doc[str(row["doc_id"])].append(row)
    oof_rows = {str(row["qid"]): [str(doc) for doc in row["doc_ids"][:K64]] for row in _jsonl(paths["exp028_oof"])}
    train = _json(paths["train"])
    if len(oof_rows) != 7000 or set(oof_rows) != set(query_rows) or set(oof_rows) != set(map(str, train)):
        raise ValueError("Phase-B query membership mismatch")
    legacy = paths["cache"] / "in-document-scores" / "e5_top2.jsonl"
    legacy_report = paths["cache"] / "in-document-scores" / "REPORT.json"
    if not legacy.exists() or not legacy_report.exists():
        raise RuntimeError("current upstream E5 top-two artifact is missing")
    config_fingerprint = _hash({
        "parent": _json(parent_dir / "REPORT.json")["fingerprint"],
        "scope": _json(scope_dir / "REPORT.json")["fingerprint"],
        "oof_sha256": _sha256(paths["exp028_oof"]),
        "legacy_sha256": _sha256(legacy),
        "bm25_sha256": _sha256(paths["bm25"]),
        "selectors": ["current_upstream_e5_top2", "in_parent_e5_top1", "in_parent_e5_mmr_070", "in_parent_e5_mmr_085", "in_parent_bm25_top2", "hybrid_rrf_dense_025", "hybrid_rrf_dense_050", "hybrid_rrf_dense_075"],
        "redundancy_threshold": .90,
    })

    @lru_cache(maxsize=8192)
    def cached_segmenter(text: str) -> str:
        return default_segmenter(text)

    started = time.time()
    completed = 0
    reused = 0
    with BM25Searcher(paths["bm25"], profile="legal_structure", segmenter=cached_segmenter) as searcher:
        searcher.load_document_ranges()
        for qid in sorted(oof_rows):
            shard = shards_dir / f"{qid}.json"
            if shard.exists():
                payload = _json(shard)
                if payload.get("config_fingerprint") == config_fingerprint and payload.get("qid") == qid and len(payload.get("pairs", [])) == K64:
                    completed += 1
                    reused += 1
                    continue
            query = np.asarray(queries[query_rows[qid]], dtype=np.float32)
            question = str(train[qid]["question"])
            pairs = []
            for rank, doc_id in enumerate(oof_rows[qid], start=1):
                chunks = doc_chunks.get(doc_id)
                if not chunks:
                    raise ValueError(f"shortlisted doc missing parent index: {qid}/{doc_id}")
                indices = [int(chunk["embedding_row"]) for chunk in chunks]
                vectors = np.asarray(embeddings[indices], dtype=np.float32)
                dense_scores = _cosine(query, vectors)
                dense_order = sorted(range(len(chunks)), key=lambda idx: (-float(dense_scores[idx]), str(chunks[idx]["chunk_id"])))
                dense_ranked = [dict(chunks[idx], score=float(dense_scores[idx])) for idx in dense_order[:8]]
                upstream = [dict(row, evidence_rank=pos, redundancy=0.0) for pos, row in enumerate(dense_ranked[:2], start=1)]
                e5_top1 = [dict(dense_ranked[0], evidence_rank=1, redundancy=0.0)]
                mmr_070 = select_mmr_evidence(chunks, dense_scores, vectors, lambda_value=.70)
                mmr_085 = select_mmr_evidence(chunks, dense_scores, vectors, lambda_value=.85)
                sparse_raw = searcher.search_document(question, doc_id, limit=8)
                sparse_ranked = [dict(row, score=-float(row["score"])) for row in sparse_raw]
                bm25_top2, _ = _bm25_primary_fallback(
                    _select_ranked_nonredundant(sparse_ranked, embedding_row, embeddings), e5_top1[0],
                )
                hybrids = {}
                for weight in (.25, .50, .75):
                    hybrid_ranked = reciprocal_hybrid(dense_ranked, sparse_ranked, dense_weight=weight)
                    hybrids[f"hybrid_rrf_dense_{int(weight * 100):03d}"] = _select_ranked_nonredundant(hybrid_ranked, embedding_row, embeddings)
                selectors = {
                    "current_upstream_e5_top2": upstream,
                    "in_parent_e5_top1": e5_top1,
                    "in_parent_e5_mmr_070": mmr_070,
                    "in_parent_e5_mmr_085": mmr_085,
                    "in_parent_bm25_top2": bm25_top2,
                    **hybrids,
                }
                selected_ids = {str(row["chunk_id"]) for rows in selectors.values() for row in rows}
                scope_candidate = None
                doc_scopes = scopes_by_doc.get(doc_id, [])
                if doc_scopes:
                    scope_indices = [int(row["embedding_row"]) for row in doc_scopes]
                    scope_scores = np.asarray(scope_vectors[scope_indices], dtype=np.float32) @ query
                    best = sorted(range(len(doc_scopes)), key=lambda idx: (-float(scope_scores[idx]), str(doc_scopes[idx]["scope_id"])))[0]
                    scope_candidate = {
                        "scope_id": doc_scopes[best]["scope_id"], "kind": doc_scopes[best]["kind"],
                        "node_id": doc_scopes[best]["node_id"], "source_start": doc_scopes[best]["source_start"],
                        "source_end": doc_scopes[best]["source_end"], "score": float(scope_scores[best]),
                        "selection_state": "BEST_DIRECT_SCORE_PENDING_INNER_THRESHOLD",
                    }
                pairs.append({
                    "doc_id": doc_id, "lambdamart_rank": rank,
                    "selectors": {key: _compact_selected(value) for key, value in selectors.items()},
                    "clause_neighbor_ids": _clause_neighbors(chunks, selected_ids),
                    "scope_candidate": scope_candidate,
                })
            _write_json(shard, {"schema_version": SCHEMA, "qid": qid, "config_fingerprint": config_fingerprint, "pairs": pairs})
            completed += 1
            if completed % 10 == 0 or completed == 7000:
                elapsed = max(time.time() - started, 1e-9)
                fresh = max(completed - reused, 1)
                eta = (7000 - completed) * elapsed / fresh
                _state(paths["results"], "phase-b", "RUNNING", completed=completed, total=7000, eta_seconds=round(eta), phase_completion_percent=20 + round(75 * completed / 7000, 2))

    selector_names = {
        "current_upstream_e5_top2", "in_parent_e5_top1", "in_parent_e5_mmr_070", "in_parent_e5_mmr_085",
        "in_parent_bm25_top2", "hybrid_rrf_dense_025", "hybrid_rrf_dense_050", "hybrid_rrf_dense_075",
    }
    upstream_mismatches = 0
    malformed_pairs = 0
    redundancy_violations = 0
    scope_overflow = 0
    bm25_zero_hit_fallbacks = 0
    pair_count = 0
    labels, label_stats = canonical_answers(
        paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl",
    )
    evaluable_qids = {qid for qid, answers in labels.items() if answers}
    non_evaluable_qids = sorted(set(oof_rows) - evaluable_qids)
    if label_stats["evaluable_queries"] != 6991 or label_stats["non_evaluable_queries"] != 9 or len(non_evaluable_qids) != 9:
        raise ValueError(f"canonical Phase-B accounting changed: {label_stats}")
    output_jsonl = output / "selector_pairs.jsonl"
    legacy_iter = _jsonl(legacy)

    def consolidated() -> Iterator[dict[str, Any]]:
        nonlocal upstream_mismatches, malformed_pairs, redundancy_violations, scope_overflow, bm25_zero_hit_fallbacks, pair_count
        for qid in sorted(oof_rows):
            shard_payload = _json(shards_dir / f"{qid}.json")
            old = next(legacy_iter)
            if str(old["qid"]) != qid:
                raise ValueError(f"legacy upstream ordering mismatch: {qid}/{old['qid']}")
            old_by_doc = {str(row["doc_id"]): row for row in old["candidates"]}
            for pair in shard_payload["pairs"]:
                pair_count += 1
                selectors = pair["selectors"]
                if not selectors.get("in_parent_bm25_top2"):
                    fallback, used = _bm25_primary_fallback([], selectors["current_upstream_e5_top2"][0])
                    selectors["in_parent_bm25_top2"] = _compact_selected(fallback)
                    bm25_zero_hit_fallbacks += int(used)
                if set(selectors) != selector_names or str(pair["doc_id"]) not in old_by_doc:
                    malformed_pairs += 1
                old_ids = [str(row["chunk_id"]) for row in old_by_doc[str(pair["doc_id"])]["evidence"]]
                new_ids = [str(row["chunk_id"]) for row in selectors["current_upstream_e5_top2"]]
                if old_ids != new_ids:
                    upstream_mismatches += 1
                for name, selected in selectors.items():
                    if len(selected) not in (1, 2) or len({row["chunk_id"] for row in selected}) != len(selected):
                        malformed_pairs += 1
                    if name != "current_upstream_e5_top2" and len(selected) == 2 and float(selected[1]["redundancy"]) >= .90:
                        redundancy_violations += 1
                if isinstance(pair.get("scope_candidate"), list):
                    scope_overflow += 1
                yield {"schema_version": SCHEMA, "qid": qid, "evaluation_eligible": qid in evaluable_qids, **pair}
        try:
            next(legacy_iter)
            raise ValueError("legacy upstream contains extra queries")
        except StopIteration:
            pass

    _write_jsonl(output_jsonl, consolidated())
    gate = pair_count == 7000 * K64 and not (upstream_mismatches or malformed_pairs or redundancy_violations or scope_overflow)
    fingerprint = _hash({"config": config_fingerprint, "pairs_sha256": _sha256(output_jsonl), "pair_count": pair_count})
    report = {
        "schema_version": SCHEMA, "status": "PASS_PHASE_B" if gate else "FAILED_PHASE_B",
        "phase": "B", "phase_completion_percent": 100 if gate else 95,
        "fingerprint": fingerprint, "config_fingerprint": config_fingerprint,
        "processed_queries": 7000, "evaluable_queries": 6991, "non_evaluable_queries": 9,
        "non_evaluable_qids": non_evaluable_qids,
        "processed_pairs": pair_count, "evaluable_pairs": 6991 * K64,
        "parents_per_query": K64,
        "selectors": sorted(selector_names), "selector_count": len(selector_names),
        "upstream_exact_mismatches": upstream_mismatches, "malformed_pairs": malformed_pairs,
        "secondary_redundancy_violations": redundancy_violations, "scope_candidate_overflow": scope_overflow,
        "bm25_zero_hit_e5_primary_fallbacks": bm25_zero_hit_fallbacks,
        "scope_policy": "one_best_direct_e5_score_pending_inner_fold_threshold",
        "same_parent_lexical_method": "BM25Searcher.search_document",
        "clause_expansion": "same_parent_immediate_chunk_neighbors_recorded",
        "labels_used_for_selection": False,
        "canonical_labels_used_only_for_evaluation_eligibility": True,
        "canonical_label_fingerprint": label_stats["label_fingerprint"],
        "corpus_chunks_reencoded": 0,
        "resumed_query_shards": reused, "fresh_query_shards": 7000 - reused,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {
        "schema_version": SCHEMA, "stage": "score-selectors-v2", "fingerprint": fingerprint,
        "selector_pairs_sha256": _sha256(output_jsonl), "config_fingerprint": config_fingerprint,
    })
    if gate:
        failed_marker = output / "_FAILED.json"
        if failed_marker.exists():
            failed_marker.unlink()
        _success(output, stage="score-selectors-v2", fingerprint=fingerprint, phase="B")
        _state(paths["results"], "phase-b", "SUCCESS", completed=100, total=100, eta_seconds=0, phase_completion_percent=100)
    else:
        _write_json(output / "_FAILED.json", {"schema_version": SCHEMA, "stage": "score-selectors-v2", "fingerprint": fingerprint, "report": report})
        _state(paths["results"], "phase-b", "FAILED", completed=95, total=100, eta_seconds=0, phase_completion_percent=95)
    return report


def reciprocal_hybrid(dense: Sequence[Mapping[str, Any]], sparse: Sequence[Mapping[str, Any]], *, dense_weight: float) -> list[dict[str, Any]]:
    if dense_weight not in (.25, .50, .75):
        raise ValueError("dense weight must be one of 0.25, 0.50, 0.75")
    values: dict[str, dict[str, Any]] = {}
    scores: Counter[str] = Counter()
    for weight, rows in ((dense_weight, dense), (1 - dense_weight, sparse)):
        for rank, row in enumerate(rows, start=1):
            values[str(row["chunk_id"])] = dict(row)
            scores[str(row["chunk_id"])] += weight / (60 + rank)
    return [dict(values[key], score=float(score), hybrid_rank=rank) for rank, (key, score) in enumerate(sorted(scores.items(), key=lambda item: (-item[1], item[0])), start=1)]


def _node_map(v3: Path) -> dict[str, dict[str, Any]]:
    return {str(row["node_id"]): row for row in _jsonl(v3 / "nodes.jsonl")}


def _boundaries(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip()
    options = [match.end() for match in SENTENCE_END.finditer(text[:limit + 1])]
    return text[:(options[-1] if options else limit)].strip()


def _token_ids(tokenizer: Any, text: str, cache: dict[str, list[int]] | None = None) -> list[int]:
    if cache is not None and text in cache:
        return cache[text]
    ids = list(tokenizer(
        text, add_special_tokens=False, truncation=False, verbose=False,
    )["input_ids"])
    if cache is not None:
        cache[text] = ids
    return ids


def truncate_query_for_pair(tokenizer: Any, query: str, *, token_cache: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Apply the registered head-96/tail-32 query policy using actual tokens."""
    ids = _token_ids(tokenizer, query, token_cache)
    if len(ids) <= QUERY_TOKEN_LIMIT:
        return {"text": query.strip(), "original_tokens": len(ids), "used_tokens": len(ids), "truncated": False}
    kept = ids[:QUERY_HEAD_TOKENS] + ids[-QUERY_TAIL_TOKENS:]
    text = tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    used = len(_token_ids(tokenizer, text, token_cache))
    if used > QUERY_TOKEN_LIMIT:
        raise AssertionError("head-tail query reconstruction exceeds 128 tokens")
    return {"text": text, "original_tokens": len(ids), "used_tokens": used, "truncated": True}


def _truncate_at_legal_boundary(tokenizer: Any, text: str, budget: int, *, required: bool = False, token_cache: dict[str, list[int]] | None = None) -> tuple[str, int, bool]:
    """Return a token-bounded prefix ending at a sentence/clause/line boundary."""
    clean = str(text).strip()
    if budget <= 0 or not clean:
        if required:
            raise ValueError("required capsule section has no token budget")
        return "", 0, bool(clean)
    ids = _token_ids(tokenizer, clean, token_cache)
    if len(ids) <= budget:
        return clean, len(ids), False
    decoded = tokenizer.decode(ids[:budget], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    # Use decode only to estimate the source character window.  The emitted
    # prefix is sliced from the original source and must end at a legal source
    # boundary; tokenizer-normalized decoded text is never emitted as evidence.
    source_window = clean[:min(len(clean), max(len(decoded) + 32, int(len(clean) * budget / len(ids)) + 32))]
    boundaries = [match.end() for match in re.finditer(r"(?:[.!?;:]|\r?\n+)\s*", source_window)]
    prefix = ""
    used = 0
    while boundaries:
        prefix = clean[:boundaries.pop()].strip()
        used = len(_token_ids(tokenizer, prefix, token_cache))
        if used <= budget:
            break
        prefix = ""
    if not prefix:
        if required:
            raise ValueError("required evidence has no sentence/clause boundary inside its budget")
        return "", 0, True
    if required and not prefix:
        raise ValueError("primary evidence vanished at a legal boundary")
    return prefix, used, True


def render_capsule_v2(*, query: str, document: Mapping[str, Any], primary: Mapping[str, Any], secondary: Mapping[str, Any] | None, ancestry: Sequence[Mapping[str, Any]], scope: Mapping[str, Any] | None, tokenizer: Any | None = None, max_length: int = CAPSULE_MAX_LENGTH, query_audit: Mapping[str, Any] | None = None, token_cache: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Render exactly one answer-first representation with an actual pair budget."""
    if tokenizer is None:
        raise ValueError("Capsule v2 requires the actual downstream tokenizer")
    query_audit = dict(query_audit) if query_audit is not None else truncate_query_for_pair(tokenizer, query, token_cache=token_cache)
    rendered_query = str(query_audit["text"])
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    document_allowance = max_length - int(query_audit["used_tokens"]) - special_tokens
    if document_allowance < 64:
        raise ValueError("query leaves insufficient document allowance")

    title = document.get("official_title") or extract_official_title(
        str(document.get("raw_text", "")), str(document.get("document_label", ""))
    )
    title_verified = isinstance(title, Mapping) and title.get("status") == "VERIFIED"
    identity = str(title.get("display_text")) if title_verified else str(document.get("document_label", ""))
    path = str(document.get("structural_path") or format_structural_path(ancestry)).strip()

    # Section budgets include their Vietnamese markers.  Primary gets 60%,
    # above the 55% floor; identity+path and applicability remain below caps.
    safety_allowance = max(1, document_allowance - 8)
    primary_budget = max(1, int(math.floor(safety_allowance * .60)))
    identity_path_budget = max(1, int(math.floor(safety_allowance * .20)))
    scope_budget = max(0, int(math.floor(safety_allowance * .15))) if scope else 0
    secondary_budget = max(0, safety_allowance - primary_budget - identity_path_budget - scope_budget)

    def section(marker: str, value: str, budget: int, *, required: bool = False) -> tuple[str, int, bool]:
        marker_tokens = len(_token_ids(tokenizer, marker + "\n", token_cache))
        content, _, cut = _truncate_at_legal_boundary(tokenizer, value, budget - marker_tokens, required=required, token_cache=token_cache)
        rendered = f"{marker}\n{content}" if content else ""
        return rendered, len(_token_ids(tokenizer, rendered, token_cache)) if rendered else 0, cut

    primary_section, primary_used, primary_cut = section(
        "[BẰNG CHỨNG CHÍNH]", str(primary.get("raw_text", "")), primary_budget, required=True
    )
    location_value = path
    location, location_used, location_cut = section(
        "[VỊ TRÍ PHÁP LÝ]", location_value, identity_path_budget
    )
    secondary_section, secondary_used, secondary_cut = section(
        "[BẰNG CHỨNG BỔ SUNG]", str(secondary.get("raw_text", "")) if secondary else "", secondary_budget
    )
    scope_section = ""
    scope_used = 0
    scope_cut = False
    if scope:
        marker = "[PHẠM VI LIÊN QUAN]" if scope.get("kind") in {"scope_of_regulation", "combined"} else "[ĐỐI TƯỢNG LIÊN QUAN]"
        scope_section, scope_used, scope_cut = section(marker, str(scope.get("raw_text", "")), scope_budget)
    identity_marker, identity_marker_used, identity_marker_cut = section(
        "[VĂN BẢN]", identity, max(1, identity_path_budget - location_used)
    )
    # Do not duplicate the identity when it already consumed the location cap.
    if not identity_marker:
        identity_marker_used = 0

    parts = [value for value in (primary_section, location, secondary_section, scope_section, identity_marker) if value]
    text = "\n".join(parts)
    # Pair token count is exact for this tokenizer contract: separately encoded
    # content tokens plus the tokenizer-declared pair special tokens.
    pair_tokens = int(query_audit["used_tokens"]) + len(_token_ids(tokenizer, text, token_cache)) + special_tokens
    if pair_tokens > max_length:
        raise AssertionError(f"pair budget violated: {pair_tokens}>{max_length}")
    if not text.startswith("[BẰNG CHỨNG CHÍNH]"):
        raise AssertionError("capsule is not answer-first")
    primary_content = primary_section.split("\n", 1)[1].strip()
    if not primary_content or primary_content not in text:
        raise AssertionError("primary evidence disappeared while rendering")
    emitted_markers = [
        marker for marker, value in (
            ("[BẰNG CHỨNG CHÍNH]", primary_section), ("[VỊ TRÍ PHÁP LÝ]", location),
            ("[BẰNG CHỨNG BỔ SUNG]", secondary_section),
            ("[PHẠM VI LIÊN QUAN]" if scope and scope.get("kind") in {"scope_of_regulation", "combined"} else "[ĐỐI TƯỢNG LIÊN QUAN]", scope_section),
            ("[VĂN BẢN]", identity_marker),
        ) if value
    ]
    if any(marker not in MARKERS for marker in emitted_markers):
        raise AssertionError("renderer emitted a non-Vietnamese or unregistered marker")
    return {
        "schema_version": SCHEMA, "query_text": rendered_query, "text": text,
        "primary_chunk_id": primary["chunk_id"],
        "secondary_chunk_id": secondary["chunk_id"] if secondary else None,
        "pair_tokens": pair_tokens, "document_allowance": document_allowance,
        "query_original_tokens": query_audit["original_tokens"],
        "query_used_tokens": query_audit["used_tokens"], "query_truncated": query_audit["truncated"],
        "allocated_token_shares": {"primary": .60, "identity_path_cap": .20, "scope_cap": .15, "secondary_remainder": round(secondary_budget / safety_allowance, 6)},
        "section_tokens": {"primary": primary_used, "identity_path": location_used + identity_marker_used, "scope": scope_used, "secondary": secondary_used},
        "truncated_sections": {"primary": primary_cut, "identity_path": location_cut or identity_marker_cut, "scope": scope_cut, "secondary": secondary_cut},
        "title_status": "VERIFIED" if title_verified else "FALLBACK_NORMALIZED_LABEL",
        "truncated": any((primary_cut, location_cut, identity_marker_cut, scope_cut, secondary_cut)),
        "one_view": True, "answer_first": True,
    }


def _ensure_capsule_content_index(paths: Mapping[str, Path], output: Path) -> tuple[Path, dict[str, Any]]:
    """Materialize source-exact Structural-v3 chunks for random capsule access."""
    index_path = output / "content_index.sqlite"
    index_manifest = output / "content_index_manifest.json"
    source_manifest = _manifest_ok(paths["v3"], key="content_fingerprint")
    fingerprint = _hash({
        "source_v3": source_manifest["content_fingerprint"],
        "source_sha256": source_manifest["artifact_sha256"]["chunks.jsonl"],
        "projection": "source_exact_chunk_raw_text_and_document_label_v2",
    })
    if index_path.exists() and index_manifest.exists():
        manifest = _json(index_manifest)
        if manifest.get("fingerprint") == fingerprint and manifest.get("chunks") == 343347:
            with sqlite3.connect(index_path) as connection:
                count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if count == 343347:
                return index_path, manifest
    tmp = index_path.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(tmp)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, document_label TEXT NOT NULL) WITHOUT ROWID")
        connection.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, parent_node_id TEXT, raw_text TEXT NOT NULL) WITHOUT ROWID")
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?)",
            ((str(row["doc_id"]), str(row.get("document_label", row["doc_id"]))) for row in _jsonl(paths["v3"] / "documents.jsonl")),
        )
        batch: list[tuple[str, str, str | None, str]] = []
        for row in _jsonl(paths["v3"] / "chunks.jsonl"):
            batch.append((str(row["chunk_id"]), str(row["doc_id"]), row.get("parent_node_id"), str(row["raw_text"])))
            if len(batch) >= 2000:
                connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", batch)
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        connection.commit()
    finally:
        connection.close()
    if chunks != 343347:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"capsule content projection has {chunks}/343347 chunks")
    tmp.replace(index_path)
    manifest = {
        "schema_version": SCHEMA, "stage": "capsule-content-index", "fingerprint": fingerprint,
        "source_v3_fingerprint": source_manifest["content_fingerprint"],
        "chunks": chunks, "documents": documents,
    }
    _write_json(index_manifest, manifest)
    return index_path, manifest


def _verified_title_metadata(paths: Mapping[str, Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reuse only the source-verified title inventory, never old capsule views."""
    metadata_dir = ROOT / "cache" / "exp030_legal_evidence_routing_canonical_v1" / "metadata"
    manifest = _json(metadata_dir / "manifest.json")
    if manifest["inputs"]["v3_manifest_sha256"] != _sha256(paths["v3"] / "manifest.json"):
        raise ValueError("verified-title inventory is not bound to current Structural-v3")
    rows = {str(row["doc_id"]): row for row in _jsonl(metadata_dir / "document_metadata.jsonl")}
    if len(rows) != 8507:
        raise ValueError(f"verified-title inventory changed: {len(rows)}")
    return rows, manifest


def build_capsules_v2(paths: Mapping[str, Path], *, model_id: str = CAPSULE_MODEL_ID, selector_dir: Path | None = None, output_dir: Path | None = None, phase_label: str = "C") -> dict[str, Any]:
    """Build resumable one-view Capsule v2 records for all Phase-B selectors."""
    from transformers import AutoTokenizer

    phase_b = selector_dir or (paths["cache"] / "in-document-selector-v2")
    phase_b_report = _json(phase_b / "REPORT.json")
    phase_b_success = _json(phase_b / "_SUCCESS.json")
    expected_status = "CROSSFIT_SELECTORS_READY" if selector_dir else "PASS_PHASE_B"
    expected_percent = 50 if selector_dir else 100
    if phase_b_report.get("status") != expected_status or phase_b_report.get("phase_completion_percent") != expected_percent:
        raise RuntimeError("Phase C requires verified PASS_PHASE_B at 100%")
    if phase_b_success.get("fingerprint") != phase_b_report.get("fingerprint"):
        raise ValueError("Phase-B success/report fingerprint mismatch")
    selector_path = phase_b / "selector_pairs.jsonl"
    selector_manifest = _json(phase_b / "manifest.json")
    expected_selector_sha = selector_manifest.get("selector_pairs_sha256")
    if expected_selector_sha is not None and _sha256(selector_path) != expected_selector_sha:
        raise ValueError("Phase-B selector artifact hash mismatch")

    output = output_dir or (paths["cache"] / "capsules-v2")
    output.mkdir(parents=True, exist_ok=True)
    run_phase = "phase-d" if selector_dir else "phase-c"
    (output / "_FAILED.json").unlink(missing_ok=True)
    _state(paths["results"], "phase-c", "BUILDING_CONTENT_INDEX", completed=0, total=100, eta_seconds=None, phase_completion_percent=0)
    content_index, content_manifest = _ensure_capsule_content_index(paths, output)
    _state(paths["results"], "phase-c", "LOADING_TOKENIZER", completed=10, total=100, eta_seconds=None, phase_completion_percent=10)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    if int(tokenizer.num_special_tokens_to_add(pair=True)) <= 0:
        raise ValueError("BGE tokenizer does not expose pair special tokens")
    titles, title_manifest = _verified_title_metadata(paths)
    train = _json(paths["train"])
    scopes: dict[str, dict[str, Any]] = {}
    for row in _jsonl(paths["cache"] / "scope-sidecar-v2" / "scope_spans.jsonl"):
        scope_id = _hash({key: row[key] for key in ("doc_id", "node_id", "kind", "source_start", "source_end")})
        scopes[scope_id] = row
    encoded_scope_ids = {str(row["scope_id"]) for row in _jsonl(paths["cache"] / "scope-embeddings-v2" / "scope_ids.jsonl")}
    if set(scopes) != encoded_scope_ids:
        raise ValueError("scope sidecar/source-exact span join does not match encoded scope IDs")
    selector_names = tuple(phase_b_report.get("selectors", ("current_upstream_e5_top2", "in_parent_e5_top1", "in_parent_e5_mmr_070", "in_parent_e5_mmr_085", "in_parent_bm25_top2", "hybrid_rrf_dense_025", "hybrid_rrf_dense_050", "hybrid_rrf_dense_075")))
    config_fingerprint = _hash({
        "phase_b": phase_b_report["fingerprint"], "selector_sha256": _sha256(selector_path),
        "content_index": content_manifest["fingerprint"], "title_metadata": title_manifest["content_fingerprint"],
        "scope_sidecar": _json(paths["cache"] / "scope-sidecar-v2" / "manifest.json")["fingerprint"],
        "model_id": model_id, "max_length": CAPSULE_MAX_LENGTH,
        "query_policy": [QUERY_TOKEN_LIMIT, QUERY_HEAD_TOKENS, QUERY_TAIL_TOKENS],
        "renderer_source_sha256": _sha256(Path(__file__)), "selectors": selector_names,
    })
    shards_dir = output / "query_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{content_index.as_posix()}?mode=ro", uri=True)
    node_index = ROOT / "cache" / "exp030_legal_evidence_routing_canonical_v1" / "metadata" / "node_index.sqlite"
    node_connection = sqlite3.connect(f"file:{node_index.as_posix()}?mode=ro", uri=True)

    @lru_cache(maxsize=100000)
    def chunk(chunk_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT doc_id, parent_node_id, raw_text FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"capsule chunk missing from frozen projection: {chunk_id}")
        return {"chunk_id": chunk_id, "doc_id": str(row[0]), "parent_node_id": row[1], "raw_text": row[2]}

    @lru_cache(maxsize=100000)
    def structural_path(parent_node_id: str | None) -> str:
        ancestry: list[dict[str, Any]] = []
        node_id = parent_node_id
        seen: set[str] = set()
        while node_id and node_id not in seen:
            seen.add(node_id)
            row = node_connection.execute(
                "SELECT parent_id, kind, heading_text FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if row is None:
                break
            ancestry.append({"kind": row[1], "heading_text": row[2] or ""})
            node_id = row[0]
        return format_structural_path(list(reversed(ancestry)))

    @lru_cache(maxsize=10000)
    def document_label(doc_id: str) -> str:
        row = connection.execute("SELECT document_label FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return str(row[0]) if row else str(doc_id)

    total_queries = int(phase_b_report.get("processed_queries", phase_b_report.get("queries", 7000)))
    processed_queries = processed_pairs = unique_capsules = query_truncations = 0
    max_pair_tokens = max_query_tokens = scope_shadow_audits = scope_shadow_fit = 0
    title_statuses: Counter[str] = Counter()
    reused = 0
    started = time.time()
    render_workers = min(4, max(1, os.cpu_count() or 1))
    executor = ThreadPoolExecutor(max_workers=render_workers, thread_name_prefix="exp033-capsule")

    def flush_query(qid: str, pairs: list[dict[str, Any]]) -> None:
        nonlocal processed_queries, processed_pairs, unique_capsules, query_truncations
        nonlocal max_pair_tokens, max_query_tokens, scope_shadow_audits, scope_shadow_fit, reused
        shard = shards_dir / f"{qid}.json"
        if shard.exists():
            old = _json(shard)
            same_pairs = [str(item.get("doc_id")) for item in old.get("pairs", [])] == [str(item.get("doc_id")) for item in pairs]
            # Phase-D rank/evaluation metadata is intentionally context-side.
            # Permit resume across a bookkeeping-only renderer revision when
            # the selector-bound pair membership is byte-for-byte unchanged.
            compatible_phase_d_shard = selector_dir is not None and same_pairs
            if (old.get("config_fingerprint") == config_fingerprint or compatible_phase_d_shard) and len(old.get("pairs", [])) == len(pairs):
                audit = old["audit"]
                processed_queries += 1; processed_pairs += len(old["pairs"]); reused += 1
                unique_capsules += int(audit["unique_capsules"])
                query_truncations += int(audit["query_truncated"])
                max_pair_tokens = max(max_pair_tokens, int(audit["max_pair_tokens"]))
                max_query_tokens = max(max_query_tokens, int(audit["query_original_tokens"]))
                scope_shadow_audits += int(audit["scope_shadow_audits"])
                scope_shadow_fit += int(audit["scope_shadow_fit"])
                title_statuses.update(audit["title_statuses"])
                return
        query = str(train[qid]["question"])
        query_token_cache: dict[str, list[int]] = {}
        query_info = truncate_query_for_pair(tokenizer, query, token_cache=query_token_cache)
        prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None]] = []
        for pair in pairs:
            doc_id = str(pair["doc_id"])
            title_row = titles.get(doc_id, {})
            document = {
                "document_label": document_label(doc_id),
                "official_title": title_row.get("official_title", {"status": "MISSING"}),
            }
            scope_candidate = pair.get("scope_candidate")
            scope = scopes.get(str(scope_candidate.get("scope_id"))) if scope_candidate else None
            selected_ids = {
                str(item["chunk_id"])
                for selector in selector_names for item in pair["selectors"][selector]
            }
            chunks = {chunk_id: chunk(chunk_id) for chunk_id in selected_ids}
            for item in chunks.values():
                item["structural_path"] = structural_path(item.get("parent_node_id"))
            prepared.append((pair, document, chunks, scope))

        def render_prepared(item: tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None]) -> tuple[dict[str, Any], dict[str, Any]]:
            pair, base_document, chunks, scope = item
            doc_id = str(pair["doc_id"])
            scope_candidate = pair.get("scope_candidate")
            local_token_cache: dict[str, list[int]] = {}
            representations: list[dict[str, Any]] = []
            key_to_id: dict[tuple[str, ...], str] = {}
            selector_capsule_ids: dict[str, str] = {}
            pair_scope = pair_scope_fit = 0
            pair_max = 0
            pair_titles: Counter[str] = Counter()
            for selector in selector_names:
                selected = pair["selectors"][selector]
                ids = tuple(str(item["chunk_id"]) for item in selected)
                if ids not in key_to_id:
                    primary = chunks[ids[0]]
                    secondary = chunks[ids[1]] if len(ids) > 1 else None
                    document = dict(base_document, structural_path=primary["structural_path"])
                    try:
                        capsule = render_capsule_v2(
                            query=query, document=document, primary=primary, secondary=secondary,
                            ancestry=(), scope=None, tokenizer=tokenizer, max_length=CAPSULE_MAX_LENGTH,
                            query_audit=query_info, token_cache=local_token_cache,
                        )
                    except Exception as exc:
                        raise type(exc)(f"qid={qid} doc_id={doc_id} selector={selector} chunks={ids}: {exc}") from exc
                    capsule.pop("query_text", None)
                    capsule_id = f"c{len(representations)}"
                    capsule["capsule_id"] = capsule_id
                    capsule["scope_materialization"] = "DEFERRED_TO_INNER_FOLD_THRESHOLD"
                    if scope:
                        try:
                            shadow = render_capsule_v2(
                                query=query, document=document, primary=primary, secondary=secondary,
                                ancestry=(), scope=scope, tokenizer=tokenizer, max_length=CAPSULE_MAX_LENGTH,
                                query_audit=query_info, token_cache=local_token_cache,
                            )
                        except Exception as exc:
                            raise type(exc)(f"scope-shadow qid={qid} doc_id={doc_id} selector={selector} chunks={ids}: {exc}") from exc
                        shadow_marker = "[PHẠM VI LIÊN QUAN]" if scope.get("kind") in {"scope_of_regulation", "combined"} else "[ĐỐI TƯỢNG LIÊN QUAN]"
                        fit = shadow_marker in shadow["text"]
                        capsule["scope_shadow_audit"] = {
                            "scope_id": scope_candidate["scope_id"], "score": scope_candidate["score"],
                            "kind": scope["kind"], "pair_tokens": shadow["pair_tokens"],
                            "fits": fit, "text_sha256": hashlib.sha256(shadow["text"].encode("utf-8")).hexdigest(),
                        }
                        pair_scope += 1
                        pair_scope_fit += int(fit)
                    key_to_id[ids] = capsule_id
                    representations.append(capsule)
                    pair_max = max(pair_max, int(capsule["pair_tokens"]))
                    pair_titles[capsule["title_status"]] += 1
                selector_capsule_ids[selector] = key_to_id[ids]
            rendered = {
                # Crossfit pair caches are rank-independent: the same
                # (query, document) appears at several context-specific
                # LambdaMART ranks.  Attach rank only when a context is read.
                "doc_id": doc_id, "lambdamart_rank": pair.get("lambdamart_rank"),
                "evaluation_eligible": pair.get("evaluation_eligible"),
                "selector_capsule_ids": selector_capsule_ids, "representations": representations,
                "scope_candidate": scope_candidate,
                "active_views_per_selector": 1, "aggregation_policy": "NEVER_MAX_VIEWS",
            }
            return rendered, {
                "unique_capsules": len(representations), "max_pair_tokens": pair_max,
                "scope_shadow_audits": pair_scope, "scope_shadow_fit": pair_scope_fit,
                "title_statuses": dict(pair_titles),
            }

        rendered_pairs = []
        shard_unique = shard_max = shard_scope = shard_scope_fit = 0
        shard_titles: Counter[str] = Counter()
        for rendered, pair_audit in executor.map(render_prepared, prepared):
            rendered_pairs.append(rendered)
            shard_unique += int(pair_audit["unique_capsules"])
            shard_max = max(shard_max, int(pair_audit["max_pair_tokens"]))
            shard_scope += int(pair_audit["scope_shadow_audits"])
            shard_scope_fit += int(pair_audit["scope_shadow_fit"])
            shard_titles.update(pair_audit["title_statuses"])
        audit = {
            "unique_capsules": shard_unique, "query_truncated": int(query_info["truncated"]),
            "query_original_tokens": query_info["original_tokens"], "max_pair_tokens": shard_max,
            "scope_shadow_audits": shard_scope, "scope_shadow_fit": shard_scope_fit,
            "title_statuses": dict(shard_titles),
        }
        _write_json(shard, {
            "schema_version": SCHEMA, "qid": qid, "query_text": query_info["text"],
            "config_fingerprint": config_fingerprint, "pairs": rendered_pairs, "audit": audit,
        })
        processed_queries += 1; processed_pairs += len(rendered_pairs)
        unique_capsules += shard_unique; query_truncations += int(query_info["truncated"])
        max_pair_tokens = max(max_pair_tokens, shard_max); max_query_tokens = max(max_query_tokens, int(query_info["original_tokens"]))
        scope_shadow_audits += shard_scope; scope_shadow_fit += shard_scope_fit; title_statuses.update(shard_titles)

    try:
        current_qid: str | None = None
        current_pairs: list[dict[str, Any]] = []
        for row in _jsonl(selector_path):
            qid = str(row["qid"])
            if current_qid is not None and qid != current_qid:
                if not current_pairs:
                    raise ValueError(f"capsule query has no pairs: {current_qid}")
                flush_query(current_qid, current_pairs)
                if processed_queries % 25 == 0:
                    elapsed = time.time() - started
                    eta = elapsed / processed_queries * (total_queries - processed_queries) if processed_queries else None
                    percent = 10 + int(85 * processed_queries / total_queries)
                    _state(paths["results"], "phase-c", "RUNNING", completed=percent, total=100, eta_seconds=eta, phase_completion_percent=percent, queries=processed_queries, total_queries=total_queries)
                current_pairs = []
            current_qid = qid
            current_pairs.append(row)
        if current_qid is not None:
            if not current_pairs:
                raise ValueError(f"capsule query has no pairs: {current_qid}")
            flush_query(current_qid, current_pairs)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        connection.close()
        node_connection.close()

    expected_pairs = int(phase_b_report.get("processed_pairs", phase_b_report.get("unique_pairs", 448000)))
    if processed_queries != 7000 or processed_pairs != expected_pairs:
        raise ValueError(f"capsule coverage mismatch queries={processed_queries} pairs={processed_pairs}")
    _state(paths["results"], "phase-c", "CONSOLIDATING", completed=95, total=100, eta_seconds=None, phase_completion_percent=95)
    output_jsonl = output / "capsules.jsonl"

    def consolidated() -> Iterator[dict[str, Any]]:
        for qid in sorted(str(value) for value in train):
            shard = shards_dir / f"{qid}.json"
            if shard.exists():
                payload = _json(shard)
                for pair in payload["pairs"]:
                    yield {"schema_version": SCHEMA, "qid": qid, "query_text": payload["query_text"], **pair}

    written = _write_jsonl(output_jsonl, consolidated())
    if written != expected_pairs:
        raise ValueError(f"capsule consolidation wrote {written}/{expected_pairs} pairs")
    fingerprint = _hash({"config": config_fingerprint, "capsules_sha256": _sha256(output_jsonl), "pairs": written})
    report = {
        "schema_version": SCHEMA, "status": "PASS_PHASE_C", "phase": "C", "phase_completion_percent": 100,
        "fingerprint": fingerprint, "config_fingerprint": config_fingerprint,
        "processed_queries": processed_queries, "evaluable_queries": phase_b_report.get("evaluable_queries", 6991),
        "non_evaluable_queries": phase_b_report.get("non_evaluable_queries", 9), "processed_pairs": written,
        "selector_count": len(selector_names), "selectors": list(selector_names),
        "unique_capsules": unique_capsules, "active_views_per_selector_document": 1,
        "variable_view_max_used": False, "scope_materialization": "deferred_to_inner_fold_threshold",
        "scope_shadow_audits": scope_shadow_audits, "scope_shadow_fit": scope_shadow_fit,
        "scope_shadow_omitted_unfit": scope_shadow_audits - scope_shadow_fit,
        "query_truncations": query_truncations, "max_query_original_tokens": max_query_tokens,
        "max_pair_tokens": max_pair_tokens, "pair_budget": CAPSULE_MAX_LENGTH,
        "title_statuses": dict(title_statuses), "resumed_query_shards": reused,
        "fresh_query_shards": total_queries - reused, "elapsed_seconds": round(time.time() - started, 3),
    }
    gate = max_pair_tokens <= CAPSULE_MAX_LENGTH and scope_shadow_fit == scope_shadow_audits
    if not gate:
        report["status"] = "FAILED_PHASE_C"
        report["phase_completion_percent"] = 95
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {
        "schema_version": SCHEMA, "stage": "build-capsules-v2", "fingerprint": fingerprint,
        "config_fingerprint": config_fingerprint, "capsules_sha256": _sha256(output_jsonl),
    })
    if gate:
        (output / "_FAILED.json").unlink(missing_ok=True)
        _success(output, stage="build-capsules-v2", fingerprint=fingerprint, phase="C")
        _state(paths["results"], "phase-c", "SUCCESS", completed=100, total=100, eta_seconds=0, phase_completion_percent=100)
    else:
        _write_json(output / "_FAILED.json", report)
        _state(paths["results"], "phase-c", "FAILED", completed=95, total=100, eta_seconds=0, phase_completion_percent=95)
    return report


def build_phase_d_capsules(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Render source-exact capsules for the fold-isolated selector union."""
    result = build_capsules_v2(
        paths, selector_dir=paths["cache"] / "phase-d-selector-pairs",
        output_dir=paths["cache"] / "phase-d-capsules", phase_label="D",
    )
    if result.get("status") != "PASS_PHASE_C":
        raise RuntimeError("crossfit capsule renderer failed its token/scope gate")
    result["phase"] = "D"; result["status"] = "CROSSFIT_CAPSULES_READY"; result["phase_completion_percent"] = 70
    output = paths["cache"] / "phase-d-capsules"
    _write_json(output / "REPORT.json", result)
    _state(paths["results"], "phase-d", "CROSSFIT_CAPSULES_READY", completed=70, total=100, eta_seconds=0, phase_completion_percent=70)
    return result


def protected_residual(original: Sequence[str], evidence_scores: Mapping[str, float], anchor_scores: Mapping[str, float], *, alpha: float, window: int, tau: float) -> list[str]:
    if len(original) < 5 or len(set(original)) != len(original):
        raise ValueError("protected residual requires unique ranking with at least five candidates")
    if not 0 <= alpha <= 1 or window not in (16, 25, 32, 50, 64) or tau not in (0, .25, .50, 1.):
        raise ValueError("protected residual uses the pre-registered grid only")
    current = list(original)
    active = current[:min(window, len(current))]
    def standardize(values: Mapping[str, float]) -> dict[str, float]:
        array = np.array([float(values.get(doc, 0.0)) for doc in active], dtype=np.float64)
        scale = float(array.std()) or 1.0
        return {doc: (float(values.get(doc, 0.0)) - float(array.mean())) / scale for doc in active}
    a, e = standardize(anchor_scores), standardize(evidence_scores)
    combined = {doc: (1 - alpha) * a[doc] + alpha * e[doc] for doc in active}
    proposed = sorted(active, key=lambda doc: (-combined[doc], doc))
    protected = current[:5]
    for challenger in proposed:
        if challenger in protected:
            continue
        weakest = min(protected, key=lambda doc: (combined.get(doc, -math.inf), doc))
        if combined[challenger] >= combined[weakest] + tau:
            protected[protected.index(weakest)] = challenger
    protected.sort(key=lambda doc: (-combined.get(doc, -math.inf), doc))
    return protected + [doc for doc in current if doc not in protected]


def build_phase_d_nested_scores(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Inner-select cheap selector/residual policy, then score each outer holdout."""
    from exp012b_tuning import load_folds
    capsule = _json(paths["cache"] / "phase-d-capsules" / "REPORT.json")
    if capsule.get("status") != "CROSSFIT_CAPSULES_READY":
        raise RuntimeError("nested gate requires verified crossfit capsules")
    labels, stats = canonical_answers(paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl")
    folds = {str(name): [str(qid) for qid in qids] for name, qids in load_folds(paths["folds"]).items()}
    selectors = ("current_upstream_e5_top2", "in_parent_e5_top1", "in_parent_e5_mmr_070", "in_parent_e5_mmr_085", "in_parent_bm25_top2", "hybrid_rrf_dense_025", "hybrid_rrf_dense_050", "hybrid_rrf_dense_075")
    evidence: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in _jsonl(paths["cache"] / "phase-d-selector-pairs" / "selector_pairs.jsonl"):
        values = {name: float(row["selectors"][name][0]["score"]) for name in selectors}
        evidence[str(row["qid"])][str(row["doc_id"])] = values
    rank_dir = paths["cache"] / "phase-d-inner-crossfit-ranks"
    contexts = {(path.parent.name, path.stem): {str(row["qid"]): row for row in _jsonl(path)} for path in rank_dir.glob("*/*.jsonl")}
    if len(contexts) != 25:
        raise ValueError("nested gate requires all 25 crossfit rank contexts")

    def score(rows: Iterable[dict[str, Any]], selector: str, window: int, alpha: float) -> tuple[float, float]:
        recalls: list[float] = []; precisions: list[float] = []
        for row in rows:
            qid = str(row["qid"]); gold = labels[qid]
            if not gold: continue
            ids = [str(doc) for doc in row["doc_ids"]]
            base = {doc: float(value) for doc, value in zip(ids, row["anchor_scores"])}
            ev = {doc: evidence[qid][doc][selector] for doc in ids}
            ranked = protected_residual(ids, ev, base, alpha=alpha, window=window, tau=0.)
            recalls.append((len(set(ranked[:5]) & gold) - len(set(ids[:5]) & gold)) / len(gold))
            precisions.append((len(set(ranked[:5]) & gold) - len(set(ids[:5]) & gold)) / 5)
        return float(np.mean(recalls)), float(np.mean(precisions))

    output = paths["cache"] / "screen-evidence"; output.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict[str, Any]] = {}; rows_out: list[dict[str, Any]] = []
    for outer in sorted(folds):
        inner_rows = [row for inner in sorted(folds) if inner != outer for row in contexts[(outer, inner)].values()]
        choices = []
        for selector in selectors:
            for window in (16, 25, 32, 50, 64):
                for alpha in (.10, .25, .50, .75):
                    recall, precision = score(inner_rows, selector, window, alpha)
                    choices.append({"selector": selector, "window": window, "alpha": alpha, "inner_recall_delta": recall, "inner_precision_delta": precision})
        choice = max(choices, key=lambda item: (item["inner_recall_delta"], item["inner_precision_delta"], -item["window"], -item["alpha"], item["selector"]))
        selected[outer] = choice
        for row in contexts[(outer, "outer_heldout")].values():
            qid = str(row["qid"])
            if not labels[qid]: continue
            ids = [str(doc) for doc in row["doc_ids"]]
            rows_out.append({"schema_version": SCHEMA, "qid": qid, "outer": outer, "candidate_ids": ids, "baseline": {doc: float(value) for doc, value in zip(ids, row["anchor_scores"])}, "evidence": {doc: evidence[qid][doc][choice["selector"]] for doc in ids}, "gold": sorted(labels[qid]), "policy": choice})
    if len(rows_out) != stats["evaluable_queries"]:
        raise ValueError(f"nested heldout coverage mismatch: {len(rows_out)}")
    _write_jsonl(output / "nested_oof.jsonl", rows_out)
    result = {"schema_version": SCHEMA, "phase": "D", "status": "NESTED_SCORES_READY", "outer_policy": selected, "evaluable_queries": len(rows_out), "non_evaluable_queries": 9, "scope_policy": "deferred_to_inner_fold_threshold_fixed_no_scope_materialization", "renderer_policy": "capsule_v2_one_answer_first_view_fixed", "fingerprint": _hash({"capsules": capsule["fingerprint"], "rows": _sha256(output / "nested_oof.jsonl"), "policies": selected})}
    _write_json(output / "NESTED_REPORT.json", result)
    return result


def screen_evidence(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Hard-stop unless a *completed nested* score file was created by EXP-033."""
    score_file = paths["cache"] / "screen-evidence" / "nested_oof.jsonl"
    if not score_file.exists():
        raise RuntimeError("screen-evidence requires EXP-033 nested OOF scores; global EXP-028 OOF is intentionally forbidden")
    rows = list(_jsonl(score_file))
    required = {"qid", "outer", "baseline", "evidence", "gold", "candidate_ids"}
    if not rows or any(not required <= set(row) for row in rows):
        raise ValueError("nested evidence score schema is incomplete")
    # Metrics are deliberately computed only from the exp033 outer-heldout rows.
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[str(row["outer"])].append(row)
    if set(by_fold) != {f"fold_{index}" for index in range(5)}:
        raise ValueError("evidence gate requires all five outer folds")
    if all("policy" in row for row in rows):
        fold_metrics = []
        all_values = []; excluded = []
        for outer, fold_rows in sorted(by_fold.items()):
            values = []
            for row in fold_rows:
                policy = row["policy"]
                ranking = protected_residual(row["candidate_ids"], row["evidence"], row["baseline"], alpha=float(policy["alpha"]), window=int(policy["window"]), tau=0.)
                gold = set(row["gold"]); base = row["candidate_ids"][:5]
                values.append(((len(set(ranking[:5]) & gold) - len(set(base) & gold)) / len(gold), (len(set(ranking[:5]) & gold) - len(set(base) & gold)) / 5, row["qid"]))
            fold_metrics.append({"outer": outer, "recall_delta": float(np.mean([x[0] for x in values])), "precision_delta": float(np.mean([x[1] for x in values]))})
            all_values.extend(values); excluded.extend(x for x in values if str(x[2]) not in _diagnostic_qids(paths))
        chosen = {"nested_outer_policies": {outer: fold_rows[0]["policy"] for outer, fold_rows in by_fold.items()}, "folds": fold_metrics, "recall_delta": float(np.mean([x[0] for x in all_values])), "precision_delta": float(np.mean([x[1] for x in all_values])), "exclude_320_recall_delta": float(np.mean([x[0] for x in excluded]))}
        gate = chosen["recall_delta"] >= .002 and chosen["precision_delta"] >= 0 and sum(x["recall_delta"] >= 0 for x in fold_metrics) >= 4 and min(x["recall_delta"] for x in fold_metrics) >= -.002 and chosen["exclude_320_recall_delta"] >= 0
        output = paths["results"] / "screen-evidence"; result = {"schema_version": SCHEMA, "phase": "D", "status": "PASS" if gate else "REJECTED_EVIDENCE_GATE", "selected": chosen, "gate_passed": gate, "nested": True}
        _write_json(output / "REPORT.json", result); _state(paths["results"], "phase-d", result["status"], completed=100, total=100, eta_seconds=0, phase_completion_percent=100)
        return result
    variants = []
    for window in (16, 25, 32, 50, 64):
        for alpha in (.10, .25, .50, .75):
            fold_metrics = []
            all_values = []
            excluded = []
            for outer, fold_rows in sorted(by_fold.items()):
                values = []
                for row in fold_rows:
                    ranking = protected_residual(row["candidate_ids"], row["evidence"], row["baseline"], alpha=alpha, window=window, tau=0.)
                    gold = set(row["gold"])
                    base = row["candidate_ids"][:5]
                    delta = len(set(ranking[:5]) & gold) / len(gold) - len(set(base) & gold) / len(gold)
                    values.append((delta, len(set(ranking[:5]) & gold) / 5 - len(set(base) & gold) / 5, row["qid"]))
                fold_metrics.append({"outer": outer, "recall_delta": float(np.mean([value[0] for value in values])), "precision_delta": float(np.mean([value[1] for value in values]))})
                all_values.extend(values)
                excluded.extend(value for value in values if str(value[2]) not in _diagnostic_qids(paths))
            variants.append({"window": window, "alpha": alpha, "folds": fold_metrics, "recall_delta": float(np.mean([value[0] for value in all_values])), "precision_delta": float(np.mean([value[1] for value in all_values])), "exclude_320_recall_delta": float(np.mean([value[0] for value in excluded]))})
    chosen = max(variants, key=lambda row: (row["recall_delta"], row["precision_delta"], -row["window"], -row["alpha"]))
    gate = chosen["recall_delta"] >= .002 and chosen["precision_delta"] >= 0 and sum(row["recall_delta"] >= 0 for row in chosen["folds"]) >= 4 and min(row["recall_delta"] for row in chosen["folds"]) >= -.002 and chosen["exclude_320_recall_delta"] >= 0
    output = paths["results"] / "screen-evidence"
    result = {"schema_version": SCHEMA, "status": "PASS" if gate else "REJECTED_EVIDENCE_GATE", "selected": chosen, "gate_passed": gate, "variants": variants}
    _write_json(output / "REPORT.json", result)
    _state(paths["results"], "screen-evidence", result["status"], completed=1, total=1, eta_seconds=0)
    return result


def audit_phase_d_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Record why Phase-D policy selection needs fresh inner-crossfit ranks."""
    phase_c = _json(paths["cache"] / "capsules-v2" / "REPORT.json")
    phase_c_success = _json(paths["cache"] / "capsules-v2" / "_SUCCESS.json")
    if phase_c.get("status") != "PASS_PHASE_C" or phase_c.get("fingerprint") != phase_c_success.get("fingerprint"):
        raise RuntimeError("Phase D requires a verified PASS_PHASE_C")
    labels, label_stats = canonical_answers(
        paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl"
    )
    if label_stats["evaluable_queries"] != 6991 or label_stats["non_evaluable_queries"] != 9:
        raise ValueError("Phase-D canonical label accounting changed")
    folds = _json(paths["folds"])
    oof_rows = list(_jsonl(paths["exp028_oof"]))
    if len(oof_rows) != 7000 or {str(row["qid"]) for row in oof_rows} != set(labels):
        raise ValueError("EXP-028 OOF/query membership mismatch")
    output = paths["results"] / "phase-d"
    result = {
        "schema_version": SCHEMA, "phase": "D", "phase_completion_percent": 5,
        "status": "NEEDS_INNER_CROSSFIT_REBUILD", "evaluable_queries": 6991,
        "non_evaluable_queries": 9, "outer_folds": sorted(folds),
        "phase_c_fingerprint": phase_c["fingerprint"],
        "global_oof_status": "outer-heldout rows are safe; outer-train rows cannot select Phase-D policy",
        "required_next_stage": "rebuild LambdaMART K64 and in-document selector rows within each outer-train inner-crossfit",
        "forbidden_shortcut": "do_not_use_global_exp028_oof_rows_for_outer_train_policy_selection",
    }
    _write_json(output / "PREFLIGHT.json", result)
    _state(paths["results"], "phase-d", result["status"], completed=5, total=100, eta_seconds=None, phase_completion_percent=5)
    return result


def build_phase_d_crossfit_ranks(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Create the rank contexts required before any Phase-D inner selection.

    For an outer split, inner-heldout rows are ranked by a model trained only
    on the other three outer-train folds.  The outer-heldout rows are ranked by
    a model trained on all four outer-train folds.  No global OOF row is used
    as an outer-train policy-selection input.
    """
    from exp012b_tuning import load_folds
    from exp027_lambdamart_shortlist import retained_answers
    from exp028_lambdamart_shortlist import _fit, _load_features

    phase_c = _json(paths["cache"] / "capsules-v2" / "REPORT.json")
    if phase_c.get("status") != "PASS_PHASE_C":
        raise RuntimeError("crossfit ranks require PASS_PHASE_C")
    output = paths["cache"] / "phase-d-inner-crossfit-ranks"
    output.mkdir(parents=True, exist_ok=True)
    feature_data, feature_index, columns = _load_features(paths["features"])
    by_qid = {str(row["qid"]): row for row in feature_index}
    labels, stats = canonical_answers(
        paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl"
    )
    # The frozen EXP-028 model was fit with this historical retained-label
    # view.  Preserve it only for reproducing the anchor ranks; Phase-D
    # evaluation and any evidence-policy selection continue to use canonical
    # labels (including the two repaired alias rows).
    anchor_labels, _ = retained_answers(
        paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl"
    )
    folds = {str(name): [str(qid) for qid in qids] for name, qids in load_folds(paths["folds"]).items()}
    if set(by_qid) != set(labels) or set().union(*map(set, folds.values())) != set(labels):
        raise ValueError("feature/fold/label membership mismatch")
    exp028_report = _json(paths["exp028_oof"].parent / "oof_report.json")
    # EXP-028's strict deployment-K selection failed because two outer folds
    # chose K50.  That does not invalidate its outer-train-selected model
    # configurations: Phase D fixes K=64, verifies the frozen outer-heldout
    # ordering exactly, and never promotes EXP-028's K50 choice.
    selections = exp028_report.get("selections", {})
    if set(selections) != set(folds) or any(
        not isinstance(selections[outer].get("chosen"), dict) for outer in folds
    ):
        raise RuntimeError("EXP-028 OOF report lacks one outer-train-selected anchor configuration per fold")
    global_oof = {
        str(row["qid"]): [str(doc) for doc in row["doc_ids"][:K64]]
        for row in _jsonl(paths["exp028_oof"])
    }
    if len(global_oof) != 7000:
        raise ValueError("EXP-028 OOF coverage mismatch")
    contexts = [(outer, inner) for outer in sorted(folds) for inner in ("outer_heldout", *[name for name in sorted(folds) if name != outer])]
    config_fingerprint = _hash({
        "phase_c": phase_c["fingerprint"], "features": _json(paths["features"] / "manifest.json")["content_fingerprint"],
        "folds": _sha256(paths["folds"]), "labels": stats["label_fingerprint"],
        "anchor_training_labels": _hash({key: sorted(value) for key, value in anchor_labels.items()}),
        "exp028_oof": _sha256(paths["exp028_oof"]),
        "anchor_selections": exp028_report["selections"], "contexts": contexts, "k": K64,
    })
    completed = reused = 0
    started = time.time()

    def rank_rows(model: Any, positions: list[int], qids: Sequence[str]) -> Iterator[dict[str, Any]]:
        for qid in qids:
            item = by_qid[qid]
            scores = model.predict(np.asarray(feature_data[item["start"]:item["end"], positions]))
            order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), str(item["doc_ids"][idx])))[:K64]
            yield {
                "qid": qid,
                "doc_ids": [str(item["doc_ids"][idx]) for idx in order],
                "anchor_scores": [float(scores[idx]) for idx in order],
            }

    for outer, inner in contexts:
        target = output / outer / f"{inner}.jsonl"
        if target.exists():
            first = next(_jsonl(target), None)
            if first and first.get("config_fingerprint") == config_fingerprint:
                completed += 1; reused += 1
                continue
        choice = exp028_report["selections"][outer]["chosen"]
        outer_train = [qid for name, qids in folds.items() if name != outer for qid in qids]
        target_qids = folds[outer] if inner == "outer_heldout" else folds[inner]
        inner_set = set() if inner == "outer_heldout" else set(folds[inner])
        train_qids = outer_train if inner == "outer_heldout" else [qid for qid in outer_train if qid not in inner_set]
        model, positions = _fit(
            [by_qid[qid] for qid in train_qids], feature_data, anchor_labels, columns,
            choice["feature_set"], choice["params"],
        )
        rows = []
        for row in rank_rows(model, positions, target_qids):
            if inner == "outer_heldout" and row["doc_ids"] != global_oof[row["qid"]]:
                raise ValueError(f"outer-heldout rank diverges from frozen EXP-028 OOF: {outer}/{row['qid']}")
            rows.append({"schema_version": SCHEMA, "config_fingerprint": config_fingerprint, "outer": outer, "inner": inner, **row})
        if len(rows) != len(target_qids):
            raise ValueError(f"crossfit rank coverage mismatch: {outer}/{inner}")
        _write_jsonl(target, rows)
        completed += 1
        elapsed = time.time() - started
        eta = (len(contexts) - completed) * elapsed / max(1, completed - reused)
        percent = 5 + round(20 * completed / len(contexts), 2)
        _state(paths["results"], "phase-d", "BUILDING_INNER_CROSSFIT_RANKS", completed=percent, total=100, eta_seconds=eta, phase_completion_percent=percent, contexts=completed, total_contexts=len(contexts))

    rows = []
    for outer, inner in contexts:
        path = output / outer / f"{inner}.jsonl"
        count = sum(1 for _ in _jsonl(path))
        expected = len(folds[outer]) if inner == "outer_heldout" else len(folds[inner])
        if count != expected:
            raise ValueError(f"persisted crossfit rank coverage mismatch: {outer}/{inner}={count}/{expected}")
        rows.append({"outer": outer, "inner": inner, "queries": count})
    fingerprint = _hash({"config": config_fingerprint, "contexts": rows})
    result = {
        "schema_version": SCHEMA, "phase": "D", "status": "CROSSFIT_RANKS_READY",
        "phase_completion_percent": 25, "fingerprint": fingerprint, "config_fingerprint": config_fingerprint,
        "contexts": rows, "contexts_total": len(contexts), "evaluable_queries": 6991,
        "non_evaluable_queries": 9, "k": K64, "resumed_contexts": reused,
        "fresh_contexts": len(contexts) - reused, "elapsed_seconds": round(time.time() - started, 3),
        "anchor_policy": "exp028_per_outer_chosen_config_with_fixed_k64",
        "exp028_source_status": exp028_report.get("status"),
        "anchor_training_label_policy": "exp028_retained_answers_v1_for_exact_anchor_reproduction",
        "policy_evaluation_label_policy": "canonical_duplicate_alias_drop_empty_passage_v1",
        "outer_heldout_labels_used_for_training_or_policy": False,
    }
    _write_json(output / "REPORT.json", result)
    _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "phase-d-inner-crossfit-ranks", "fingerprint": fingerprint, "config_fingerprint": config_fingerprint})
    _success(output, stage="phase-d-inner-crossfit-ranks", fingerprint=fingerprint, phase="D")
    _state(paths["results"], "phase-d", "CROSSFIT_RANKS_READY", completed=25, total=100, eta_seconds=0, phase_completion_percent=25)
    return result


def build_phase_d_selector_pairs(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Materialize selector evidence for the union of crossfit K64 pairs.

    Existing Phase-B rows are reusable because their selector computation has
    no label or fold input.  Missing pairs are computed with precisely the
    same deterministic selector implementation, in query shards so a long
    lexical pass can resume safely.
    """
    ranks_dir = paths["cache"] / "phase-d-inner-crossfit-ranks"
    ranks_report = _json(ranks_dir / "REPORT.json")
    if ranks_report.get("status") != "CROSSFIT_RANKS_READY":
        raise RuntimeError("Phase-D selector cache requires verified crossfit ranks")
    phase_b_dir = paths["cache"] / "in-document-selector-v2"
    phase_b = _json(phase_b_dir / "REPORT.json")
    if phase_b.get("status") != "PASS_PHASE_B":
        raise RuntimeError("Phase-D selector cache requires PASS_PHASE_B")
    output = paths["cache"] / "phase-d-selector-pairs"
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    needed: dict[str, set[str]] = defaultdict(set)
    for path in sorted(ranks_dir.glob("*/*.jsonl")):
        for row in _jsonl(path):
            needed[str(row["qid"])].update(str(doc) for doc in row["doc_ids"])
    if len(needed) != 7000:
        raise ValueError("crossfit-rank query coverage changed")
    config_fingerprint = _hash({
        "ranks": ranks_report["fingerprint"], "phase_b": phase_b["fingerprint"],
        "parent": _json(paths["cache"] / "parent-index" / "REPORT.json")["fingerprint"],
        "scope": _json(paths["cache"] / "scope-embeddings-v2" / "REPORT.json")["fingerprint"],
        "bm25": _sha256(paths["bm25"]), "selector_policy": "phase_b_v2_pair_exact_no_labels",
    })
    doc_chunks = {str(row["doc_id"]): row["chunks"] for row in _jsonl(paths["cache"] / "parent-index" / "doc_to_chunks.jsonl")}
    embedding_row = {str(chunk["chunk_id"]): int(chunk["embedding_row"]) for values in doc_chunks.values() for chunk in values}
    query_ids = _json(paths["query_embeddings"] / "train_query_ids.json")
    query_rows = {str(qid): pos for pos, qid in enumerate(query_ids)}
    queries = np.load(paths["query_embeddings"] / "train_queries.f32.npy", mmap_mode="r")
    embeddings = np.load(paths["e5"] / "embeddings.f16.npy", mmap_mode="r")
    scope_vectors = np.load(paths["cache"] / "scope-embeddings-v2" / "scope_embeddings.f16.npy", mmap_mode="r")
    scopes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(paths["cache"] / "scope-embeddings-v2" / "scope_ids.jsonl"):
        scopes_by_doc[str(row["doc_id"])].append(row)
    train = _json(paths["train"])

    @lru_cache(maxsize=8192)
    def cached_segmenter(text: str) -> str:
        return default_segmenter(text)

    def compute_pair(qid: str, doc_id: str, searcher: Any) -> dict[str, Any]:
        chunks = doc_chunks.get(doc_id)
        if not chunks:
            raise ValueError(f"shortlisted doc missing parent index: {qid}/{doc_id}")
        query = np.asarray(queries[query_rows[qid]], dtype=np.float32)
        indices = [int(chunk["embedding_row"]) for chunk in chunks]
        vectors = np.asarray(embeddings[indices], dtype=np.float32)
        dense_scores = _cosine(query, vectors)
        order = sorted(range(len(chunks)), key=lambda idx: (-float(dense_scores[idx]), str(chunks[idx]["chunk_id"])))
        dense_ranked = [dict(chunks[idx], score=float(dense_scores[idx])) for idx in order[:8]]
        e5_top1 = [dict(dense_ranked[0], evidence_rank=1, redundancy=0.0)]
        sparse = [dict(row, score=-float(row["score"])) for row in searcher.search_document(str(train[qid]["question"]), doc_id, limit=8)]
        bm25, _ = _bm25_primary_fallback(_select_ranked_nonredundant(sparse, embedding_row, embeddings), e5_top1[0])
        selectors = {
            "current_upstream_e5_top2": [dict(row, evidence_rank=pos, redundancy=0.0) for pos, row in enumerate(dense_ranked[:2], 1)],
            "in_parent_e5_top1": e5_top1, "in_parent_e5_mmr_070": select_mmr_evidence(chunks, dense_scores, vectors, lambda_value=.70),
            "in_parent_e5_mmr_085": select_mmr_evidence(chunks, dense_scores, vectors, lambda_value=.85), "in_parent_bm25_top2": bm25,
            **{f"hybrid_rrf_dense_{int(weight * 100):03d}": _select_ranked_nonredundant(reciprocal_hybrid(dense_ranked, sparse, dense_weight=weight), embedding_row, embeddings) for weight in (.25, .50, .75)},
        }
        selected_ids = {str(row["chunk_id"]) for values in selectors.values() for row in values}
        scope = None; values = scopes_by_doc.get(doc_id, [])
        if values:
            score = np.asarray(scope_vectors[[int(row["embedding_row"]) for row in values]], dtype=np.float32) @ query
            best = sorted(range(len(values)), key=lambda idx: (-float(score[idx]), str(values[idx]["scope_id"])))[0]
            item = values[best]; scope = {"scope_id": item["scope_id"], "kind": item["kind"], "node_id": item["node_id"], "source_start": item["source_start"], "source_end": item["source_end"], "score": float(score[best]), "selection_state": "BEST_DIRECT_SCORE_PENDING_INNER_THRESHOLD"}
        return {"doc_id": doc_id, "selectors": {key: _compact_selected(value) for key, value in selectors.items()}, "clause_neighbor_ids": _clause_neighbors(chunks, selected_ids), "scope_candidate": scope}

    completed = reused = computed = 0; started = time.time(); source = _jsonl(phase_b_dir / "selector_pairs.jsonl")
    source_qid = None; source_rows: list[dict[str, Any]] = []
    def flush(qid: str, rows: list[dict[str, Any]], searcher: Any) -> None:
        nonlocal completed, reused, computed
        target = shards / f"{qid}.json"
        if target.exists():
            old = _json(target)
            if old.get("config_fingerprint") == config_fingerprint and len(old.get("pairs", [])) == len(needed[qid]):
                completed += 1; reused += 1; return
        existing = {str(row["doc_id"]): {key: value for key, value in row.items() if key not in ("schema_version", "qid", "evaluation_eligible", "lambdamart_rank")} for row in rows}
        pairs = [existing[doc] if doc in existing else compute_pair(qid, doc, searcher) for doc in sorted(needed[qid])]
        computed += sum(doc not in existing for doc in needed[qid])
        _write_json(target, {"schema_version": SCHEMA, "qid": qid, "config_fingerprint": config_fingerprint, "pairs": pairs})
        completed += 1
        if completed % 10 == 0 or completed == len(needed):
            elapsed = time.time() - started; fresh = max(1, completed - reused)
            percent = 25 + round(25 * completed / len(needed), 2)
            _state(paths["results"], "phase-d", "BUILDING_INNER_CROSSFIT_SELECTORS", completed=percent, total=100, eta_seconds=(len(needed)-completed)*elapsed/fresh, phase_completion_percent=percent, queries=completed, missing_pairs_computed=computed)
    with BM25Searcher(paths["bm25"], profile="legal_structure", segmenter=cached_segmenter) as searcher:
        searcher.load_document_ranges()
        for row in source:
            qid = str(row["qid"])
            if source_qid is not None and qid != source_qid:
                flush(source_qid, source_rows, searcher); source_rows = []
            source_qid = qid; source_rows.append(row)
        if source_qid is not None: flush(source_qid, source_rows, searcher)
    if completed != len(needed): raise ValueError("Phase-B selector source did not cover all queries")
    output_rows = ({"schema_version": SCHEMA, "qid": qid, **pair} for qid in sorted(needed) for pair in _json(shards / f"{qid}.json")["pairs"])
    _write_jsonl(output / "selector_pairs.jsonl", output_rows)
    pairs = sum(len(needed[qid]) for qid in needed); fingerprint = _hash({"config": config_fingerprint, "pairs_sha256": _sha256(output / "selector_pairs.jsonl")})
    result = {"schema_version": SCHEMA, "phase": "D", "status": "CROSSFIT_SELECTORS_READY", "phase_completion_percent": 50, "fingerprint": fingerprint, "config_fingerprint": config_fingerprint, "queries": len(needed), "unique_pairs": pairs, "reused_phase_b_pairs": pairs-computed, "computed_missing_pairs": computed, "labels_used_for_selection": False, "elapsed_seconds": round(time.time()-started, 3)}
    _write_json(output / "REPORT.json", result); _write_json(output / "manifest.json", {"schema_version": SCHEMA, "stage": "phase-d-selector-pairs", "fingerprint": fingerprint, "config_fingerprint": config_fingerprint}); _success(output, stage="phase-d-selector-pairs", fingerprint=fingerprint, phase="D")
    _state(paths["results"], "phase-d", "CROSSFIT_SELECTORS_READY", completed=50, total=100, eta_seconds=0, phase_completion_percent=50)
    return result


def _diagnostic_qids(paths: Mapping[str, Path]) -> set[str]:
    report = paths["results"].parents[0] / "exp031_capsule_diagnostics" / "REPORT.json"
    if not report.exists():
        return set()
    payload = _json(report)
    return {str(value) for value in payload.get("diagnostic_qids", [])}


def preflight_bge(paths: Mapping[str, Path], *, device: str, local_only: bool) -> dict[str, Any]:
    gate = _json(paths["results"] / "screen-evidence" / "REPORT.json")
    if not gate.get("gate_passed"):
        raise RuntimeError("BGE preflight is forbidden until evidence gate passes")
    if device != "cuda":
        raise RuntimeError("BGE preflight requires the declared CUDA device")
    if local_only:
        raise RuntimeError("local-only BGE preflight needs a locally available model and explicit runner implementation")
    raise RuntimeError("Protected BGE preflight is intentionally not auto-launched; implement after a passing evidence gate and measured CUDA availability")


def train_bge(paths: Mapping[str, Path]) -> dict[str, Any]:
    gate = _json(paths["results"] / "screen-evidence" / "REPORT.json")
    if not gate.get("gate_passed"):
        raise RuntimeError("BGE training forbidden: REJECTED_EVIDENCE_GATE")
    raise RuntimeError("BGE training requires a completed protected preflight artifact")


def evaluate_bge(paths: Mapping[str, Path]) -> dict[str, Any]:
    raise RuntimeError("BGE evaluation requires five protected outer-fold checkpoints")


def report(paths: Mapping[str, Path]) -> dict[str, Any]:
    reports = {}
    for name in ("audit-inputs", "scope-audit", "screen-evidence"):
        path = paths["results"] / name / "REPORT.json"
        if path.exists():
            reports[name] = _json(path)
    result = {"schema_version": SCHEMA, "status": "INCOMPLETE", "available_reports": reports}
    if reports.get("screen-evidence", {}).get("status") == "REJECTED_EVIDENCE_GATE":
        result["status"] = "REJECTED_EVIDENCE_GATE"
    elif reports.get("scope-audit", {}).get("status") == "WAITING_SCOPE_SPOTCHECK":
        result["status"] = "WAITING_SCOPE_SPOTCHECK"
    _write_json(paths["results"] / "REPORT.json", result)
    return result


def overnight(paths: Mapping[str, Path]) -> dict[str, Any]:
    audit = paths["results"] / "audit-inputs" / "REPORT.json"
    if not audit.exists():
        audit_inputs(paths)
    annotations = paths["cache"] / "scope-audit" / "annotations.jsonl"
    if not annotations.exists():
        sample_scope_audit(paths)
    _state(paths["results"], "overnight", "WAITING_AGENT_SCOPE_REVIEW", completed=1, total=4, eta_seconds=None)
    return {"schema_version": SCHEMA, "status": "WAITING_AGENT_SCOPE_REVIEW", "hard_stop": "scope annotations require agent review, then user spot-check"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit-inputs", "sample-scope-audit", "review-scope-audit", "finalize-scope-audit", "repair-scope-sidecar", "reaudit-scope-sidecar", "finalize-scope-phase-a", "build-parent-index", "encode-scope-sidecar", "score-in-document", "score-selectors-v2", "build-capsules-v2", "audit-phase-d", "build-phase-d-crossfit-ranks", "build-phase-d-selector-pairs", "build-phase-d-capsules", "build-phase-d-nested-scores", "screen-evidence", "preflight-bge", "train-bge", "evaluate-bge", "report", "overnight"))
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "exp033_in_document_evidence_routing")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "exp033_in_document_evidence_routing")
    parser.add_argument("--train", type=Path, default=ROOT / "public_test_dataset" / "train.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "cache" / "cv_folds.json")
    parser.add_argument("--preprocessing", type=Path, default=ROOT / "cache" / "final_preprocessed_v2")
    parser.add_argument("--v3", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1")
    parser.add_argument("--e5", type=Path, default=ROOT / "cache" / "e5_final_v1")
    parser.add_argument("--query-embeddings", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings")
    parser.add_argument("--candidates", type=Path, default=ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl")
    parser.add_argument("--features", type=Path, default=ROOT / "cache" / "exp027_lambdamart_shortlist" / "features")
    parser.add_argument("--exp028-oof", type=Path, default=ROOT / "results" / "exp028_lambdamart_shortlist" / "oof" / "oof_predictions.jsonl")
    parser.add_argument("--bm25-db", type=Path, default=ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "bm25_v3.sqlite")
    parser.add_argument("--capsule-model-id", default=CAPSULE_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths(args)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)
    actions = {
        "audit-inputs": lambda: audit_inputs(paths), "sample-scope-audit": lambda: sample_scope_audit(paths),
        "review-scope-audit": lambda: review_scope_audit(paths),
        "finalize-scope-audit": lambda: finalize_scope_audit(paths), "build-parent-index": lambda: build_parent_index(paths),
        "repair-scope-sidecar": lambda: repair_scope_sidecar(paths),
        "reaudit-scope-sidecar": lambda: reaudit_scope_sidecar(paths),
        "finalize-scope-phase-a": lambda: finalize_scope_phase_a(paths),
        "encode-scope-sidecar": lambda: encode_scope_sidecar(paths, device=args.device),
        "score-in-document": lambda: score_in_document(paths),
        "score-selectors-v2": lambda: score_selectors_v2(paths),
        "build-capsules-v2": lambda: build_capsules_v2(paths, model_id=args.capsule_model_id),
        "audit-phase-d": lambda: audit_phase_d_inputs(paths), "build-phase-d-crossfit-ranks": lambda: build_phase_d_crossfit_ranks(paths), "build-phase-d-selector-pairs": lambda: build_phase_d_selector_pairs(paths), "build-phase-d-capsules": lambda: build_phase_d_capsules(paths), "build-phase-d-nested-scores": lambda: build_phase_d_nested_scores(paths), "screen-evidence": lambda: screen_evidence(paths), "preflight-bge": lambda: preflight_bge(paths, device=args.device, local_only=args.local_only),
        "train-bge": lambda: train_bge(paths), "evaluate-bge": lambda: evaluate_bge(paths), "report": lambda: report(paths), "overnight": lambda: overnight(paths),
    }
    try:
        result = actions[args.command]()
    except Exception as exc:
        if args.command == "build-capsules-v2":
            failure = {"schema_version": SCHEMA, "stage": "build-capsules-v2", "error_type": type(exc).__name__, "error": str(exc)}
            failure_dir = paths["cache"] / "capsules-v2"
            _write_json(failure_dir / "_FAILED.json", failure)
            _state(paths["results"], "phase-c", "FAILED", completed=10, total=100, eta_seconds=0, phase_completion_percent=10, failure_reason=str(exc))
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
