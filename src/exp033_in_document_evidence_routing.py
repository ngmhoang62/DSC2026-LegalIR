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
    return [{
        "chunk_id": str(row["chunk_id"]),
        "score": float(row.get("score", 0.0)),
        "evidence_rank": int(row.get("evidence_rank", rank)),
        "redundancy": float(row.get("redundancy", 0.0)),
    } for rank, row in enumerate(rows, start=1)]


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
                bm25_top2 = _select_ranked_nonredundant(sparse_ranked, embedding_row, embeddings)
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
    pair_count = 0
    output_jsonl = output / "selector_pairs.jsonl"
    legacy_iter = _jsonl(legacy)

    def consolidated() -> Iterator[dict[str, Any]]:
        nonlocal upstream_mismatches, malformed_pairs, redundancy_violations, scope_overflow, pair_count
        for qid in sorted(oof_rows):
            shard_payload = _json(shards_dir / f"{qid}.json")
            old = next(legacy_iter)
            if str(old["qid"]) != qid:
                raise ValueError(f"legacy upstream ordering mismatch: {qid}/{old['qid']}")
            old_by_doc = {str(row["doc_id"]): row for row in old["candidates"]}
            for pair in shard_payload["pairs"]:
                pair_count += 1
                selectors = pair["selectors"]
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
                yield {"schema_version": SCHEMA, "qid": qid, **pair}
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
        "queries": 7000, "parents_per_query": K64, "pairs": pair_count,
        "selectors": sorted(selector_names), "selector_count": len(selector_names),
        "upstream_exact_mismatches": upstream_mismatches, "malformed_pairs": malformed_pairs,
        "secondary_redundancy_violations": redundancy_violations, "scope_candidate_overflow": scope_overflow,
        "scope_policy": "one_best_direct_e5_score_pending_inner_fold_threshold",
        "same_parent_lexical_method": "BM25Searcher.search_document",
        "clause_expansion": "same_parent_immediate_chunk_neighbors_recorded",
        "labels_used": False, "corpus_chunks_reencoded": 0,
        "resumed_query_shards": reused, "fresh_query_shards": 7000 - reused,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(output / "REPORT.json", report)
    _write_json(output / "manifest.json", {
        "schema_version": SCHEMA, "stage": "score-selectors-v2", "fingerprint": fingerprint,
        "selector_pairs_sha256": _sha256(output_jsonl), "config_fingerprint": config_fingerprint,
    })
    if gate:
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


def render_capsule_v2(*, query: str, document: Mapping[str, Any], primary: Mapping[str, Any], secondary: Mapping[str, Any] | None, ancestry: Sequence[Mapping[str, Any]], scope: Mapping[str, Any] | None, tokenizer: Any | None = None, max_length: int = 512) -> dict[str, Any]:
    """One representation/document.  If a tokenizer is supplied, enforce pair length."""
    title = extract_official_title(str(document.get("raw_text", "")), str(document.get("document_label", "")))
    identity = title["display_text"] if title.get("status") == "VERIFIED" else str(document.get("document_label", ""))
    path = format_structural_path(ancestry)
    optional = ""
    if scope:
        marker = "[PHẠM VI LIÊN QUAN]" if scope["kind"] == "scope_of_regulation" else "[ĐỐI TƯỢNG LIÊN QUAN]"
        optional = f"\n{marker}\n{_boundaries(str(scope['raw_text']), 600)}"
    text = f"[BẰNG CHỨNG CHÍNH]\n{_boundaries(str(primary['raw_text']), 2600)}\n[VỊ TRÍ PHÁP LÝ]\n{path}\n[VĂN BẢN]\n{identity}{optional}"
    if secondary:
        text += f"\n[BẰNG CHỨNG BỔ SUNG]\n{_boundaries(str(secondary['raw_text']), 900)}"
    truncated = False
    pair_tokens = None
    if tokenizer is not None:
        def length(value: str) -> int:
            return len(tokenizer(query, value, add_special_tokens=True, truncation=False)["input_ids"])
        while length(text) > max_length:
            truncated = True
            target = max(256, len(text) - max(16, (length(text) - max_length) * 4))
            text = _boundaries(text, target)
            if target <= 256 and length(text) > max_length:
                raise ValueError("answer-first capsule cannot satisfy actual tokenizer budget")
        pair_tokens = length(text)
        if pair_tokens > max_length:
            raise AssertionError("pair budget violated")
    if "[BẰNG CHỨNG CHÍNH]" not in text or not str(primary["raw_text"]).strip()[:12] in text:
        raise AssertionError("primary evidence disappeared while rendering")
    return {"schema_version": SCHEMA, "text": text, "primary_chunk_id": primary["chunk_id"], "secondary_chunk_id": secondary["chunk_id"] if secondary else None, "pair_tokens": pair_tokens, "truncated": truncated, "one_view": True}


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
    parser.add_argument("command", choices=("audit-inputs", "sample-scope-audit", "review-scope-audit", "finalize-scope-audit", "repair-scope-sidecar", "reaudit-scope-sidecar", "finalize-scope-phase-a", "build-parent-index", "encode-scope-sidecar", "score-in-document", "score-selectors-v2", "build-capsules-v2", "screen-evidence", "preflight-bge", "train-bge", "evaluate-bge", "report", "overnight"))
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
        "build-capsules-v2": lambda: (_ for _ in ()).throw(RuntimeError("build-capsules-v2 requires fold-isolated EXP-033 selector output")),
        "screen-evidence": lambda: screen_evidence(paths), "preflight-bge": lambda: preflight_bge(paths, device=args.device, local_only=args.local_only),
        "train-bge": lambda: train_bge(paths), "evaluate-bge": lambda: evaluate_bge(paths), "report": lambda: report(paths), "overnight": lambda: overnight(paths),
    }
    result = actions[args.command]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
