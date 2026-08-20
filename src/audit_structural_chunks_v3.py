"""Audit and quality gate for LegalIR structural chunk cache v3."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from structural_chunker_v3 import (
    DEFAULT_TOKENIZER,
    HuggingFaceTokenizerAdapter,
    canonical_json,
    sha256_file,
)


AUDIT_SCHEMA_VERSION = "legalir.structural_audit.v1"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def grouped_by_doc(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for doc_id, group in itertools.groupby(read_jsonl(path), key=lambda item: item["doc_id"]):
        yield str(doc_id), list(group)


def _source_documents(contexts_dir: Path) -> tuple[dict[str, Path], list[str]]:
    paths: dict[str, Path] = {}
    duplicate_ids: list[str] = []
    for path in sorted(contexts_dir.glob("context_*.json"), key=lambda item: item.name):
        with path.open("r", encoding="utf-8") as handle:
            doc_id = str(json.load(handle)["id"])
        if doc_id in paths:
            duplicate_ids.append(doc_id)
        paths[doc_id] = path
    return paths, duplicate_ids


def _old_cache_summary(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 131072))
        tail = handle.read().decode("utf-8", errors="ignore")
    documents = re.search(r'"total_documents"\s*:\s*(\d+)', tail)
    chunks = re.search(r'"total_chunks"\s*:\s*(\d+)', tail)
    if not documents or not chunks:
        return None
    return {"documents": int(documents.group(1)), "chunks": int(chunks.group(1))}


def _non_whitespace(text: str) -> bool:
    return bool(text and not text.isspace())


def _coverage_gaps(passage: str, chunks: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    intervals = sorted(
        (int(chunk["start"]), int(chunk["end"]))
        for chunk in chunks
        if not chunk.get("synthetic")
        and chunk.get("start") is not None
        and chunk.get("end") is not None
    )
    gaps: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end in intervals:
        if start > cursor and _non_whitespace(passage[cursor:start]):
            gaps.append((cursor, start, passage[cursor:start][:160]))
        cursor = max(cursor, end)
    if cursor < len(passage) and _non_whitespace(passage[cursor:]):
        gaps.append((cursor, len(passage), passage[cursor:][:160]))
    return gaps


def _unexpected_overlaps(chunks: list[dict[str, Any]]) -> list[tuple[str, str]]:
    source_chunks = sorted(
        [
            chunk
            for chunk in chunks
            if not chunk.get("synthetic") and chunk.get("start") is not None
        ],
        key=lambda chunk: (int(chunk["start"]), int(chunk["end"])),
    )
    overlaps: list[tuple[str, str]] = []
    previous: dict[str, Any] | None = None
    previous_end = -1
    for chunk in source_chunks:
        start = int(chunk["start"])
        end = int(chunk["end"])
        if previous is not None and start < previous_end:
            reasons = {previous.get("split_reason"), chunk.get("split_reason")}
            if "oversized_leaf_window" not in reasons:
                overlaps.append((previous["chunk_id"], chunk["chunk_id"]))
        if end > previous_end:
            previous = chunk
            previous_end = end
    return overlaps


def _add_error(
    errors: list[dict[str, Any]], code: str, *, doc_id: str | None = None, detail: Any = None
) -> None:
    record: dict[str, Any] = {"code": code}
    if doc_id is not None:
        record["doc_id"] = doc_id
    if detail is not None:
        record["detail"] = detail
    errors.append(record)


def _artifact_integrity(cache_dir: Path, manifest: dict[str, Any], errors: list[dict[str, Any]]) -> None:
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = cache_dir / name
        if not path.exists():
            _add_error(errors, "missing_artifact", detail=name)
        else:
            actual = sha256_file(path)
            if actual != expected:
                _add_error(
                    errors,
                    "artifact_hash_mismatch",
                    detail={"artifact": name, "expected": expected, "actual": actual},
                )


def audit_cache(
    cache_dir: Path,
    contexts_dir: Path,
    train_path: Path,
    output_dir: Path,
    tokenizer: Any,
    *,
    expected_documents: int = 8532,
    compare_cache: Path | None = None,
    old_v1_path: Path | None = None,
    old_v2_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    hard_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    _artifact_integrity(cache_dir, manifest, hard_errors)

    source_paths, duplicate_source_ids = _source_documents(contexts_dir)
    for doc_id in duplicate_source_ids:
        _add_error(hard_errors, "duplicate_source_doc_id", doc_id=doc_id)
    if len(source_paths) != expected_documents:
        _add_error(
            hard_errors,
            "unexpected_source_document_count",
            detail={"expected": expected_documents, "actual": len(source_paths)},
        )

    documents = list(read_jsonl(cache_dir / "documents.jsonl"))
    document_ids = [str(document["doc_id"]) for document in documents]
    if len(document_ids) != len(set(document_ids)):
        _add_error(hard_errors, "duplicate_document_record")
    if set(document_ids) != set(source_paths):
        _add_error(
            hard_errors,
            "document_set_mismatch",
            detail={
                "missing": sorted(set(source_paths) - set(document_ids))[:50],
                "extra": sorted(set(document_ids) - set(source_paths))[:50],
            },
        )
    if len(documents) != expected_documents:
        _add_error(
            hard_errors,
            "unexpected_cache_document_count",
            detail={"expected": expected_documents, "actual": len(documents)},
        )

    with (cache_dir / "doc_to_chunk_ids.json").open("r", encoding="utf-8") as handle:
        doc_index = json.load(handle)
    if set(doc_index) != set(document_ids):
        _add_error(hard_errors, "doc_index_set_mismatch")

    node_groups = iter(grouped_by_doc(cache_dir / "nodes.jsonl"))
    chunk_groups = iter(grouped_by_doc(cache_dir / "chunks.jsonl"))
    next_nodes = next(node_groups, None)
    next_chunks = next(chunk_groups, None)
    seen_node_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    represented_docs: set[str] = set()
    parse_modes: Counter[str] = Counter()
    split_reasons: Counter[str] = Counter()
    token_counts: list[int] = []
    chunks_per_doc: dict[str, int] = {}
    article_labels_per_doc: dict[str, Counter[str]] = {}
    anomaly_samples: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()

    for doc_number, document in enumerate(documents, start=1):
        doc_id = str(document["doc_id"])
        parse_modes[document["parse_mode"]] += 1
        if document["parse_mode"] == "fallback":
            warning_counts["fallback_document"] += 1

        nodes: list[dict[str, Any]] = []
        if next_nodes is not None and next_nodes[0] == doc_id:
            nodes = next_nodes[1]
            next_nodes = next(node_groups, None)
        chunks: list[dict[str, Any]] = []
        if next_chunks is not None and next_chunks[0] == doc_id:
            chunks = next_chunks[1]
            next_chunks = next(chunk_groups, None)
        chunks_per_doc[doc_id] = len(chunks)
        if chunks:
            represented_docs.add(doc_id)
        else:
            _add_error(hard_errors, "document_without_chunks", doc_id=doc_id)

        source_path = source_paths.get(doc_id)
        if source_path is None:
            continue
        with source_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        passage = str(source.get("passage") or "")
        if hashlib.sha256(passage.encode("utf-8")).hexdigest() != document["source_sha256"]:
            _add_error(hard_errors, "source_sha256_mismatch", doc_id=doc_id)

        node_map = {node["node_id"]: node for node in nodes}
        if len(node_map) != len(nodes):
            _add_error(hard_errors, "duplicate_node_id", doc_id=doc_id)
        for node_id in node_map:
            if node_id in seen_node_ids:
                _add_error(hard_errors, "duplicate_node_id_global", doc_id=doc_id, detail=node_id)
            seen_node_ids.add(node_id)

        labels = Counter(
            str(node["label"]).casefold() for node in nodes if node["kind"] == "article"
        )
        article_labels_per_doc[doc_id] = labels
        repeated_labels = {label: count for label, count in labels.items() if count > 1}
        if repeated_labels:
            warning_counts["repeated_article_label"] += 1
            if len(anomaly_samples) < 50:
                anomaly_samples.append(
                    {"doc_id": doc_id, "reason": "repeated_article_label", "detail": repeated_labels}
                )

        for node in nodes:
            if not str(node.get("raw_text") or "").strip():
                _add_error(hard_errors, "empty_node", doc_id=doc_id, detail=node["node_id"])
            if node.get("synthetic"):
                if node.get("start") is not None or node.get("end") is not None:
                    _add_error(hard_errors, "synthetic_node_has_offsets", doc_id=doc_id)
                continue
            start, end = node.get("start"), node.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(passage)):
                _add_error(hard_errors, "invalid_node_span", doc_id=doc_id, detail=node["node_id"])
                continue
            if passage[start:end] != node["raw_text"]:
                _add_error(hard_errors, "node_round_trip_mismatch", doc_id=doc_id, detail=node["node_id"])
            parent_id = node.get("parent_id")
            if parent_id:
                parent = node_map.get(parent_id)
                if parent is None:
                    _add_error(hard_errors, "broken_parent_reference", doc_id=doc_id, detail=node["node_id"])
                elif not parent.get("synthetic"):
                    if not (int(parent["start"]) <= start and int(parent["end"]) >= end):
                        _add_error(hard_errors, "child_outside_parent", doc_id=doc_id, detail=node["node_id"])

        chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        if len(chunk_map) != len(chunks):
            _add_error(hard_errors, "duplicate_chunk_id", doc_id=doc_id)
        retrieval_texts = [str(chunk.get("retrieval_text") or "") for chunk in chunks]
        if hasattr(tokenizer, "token_lengths"):
            actual_token_lengths = tokenizer.token_lengths(retrieval_texts)
        else:
            actual_token_lengths = [tokenizer.token_length(text) for text in retrieval_texts]
        for chunk, actual_tokens in zip(chunks, actual_token_lengths):
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_chunk_ids:
                _add_error(hard_errors, "duplicate_chunk_id_global", doc_id=doc_id, detail=chunk_id)
            seen_chunk_ids.add(chunk_id)
            if not str(chunk.get("raw_text") or "").strip() or not str(chunk.get("retrieval_text") or "").strip():
                _add_error(hard_errors, "empty_chunk", doc_id=doc_id, detail=chunk_id)
            if chunk.get("parent_node_id") not in node_map:
                _add_error(hard_errors, "broken_chunk_parent", doc_id=doc_id, detail=chunk_id)
            token_counts.append(actual_tokens)
            split_reasons[chunk["split_reason"]] += 1
            if actual_tokens != int(chunk["token_count"]):
                _add_error(hard_errors, "stored_token_count_mismatch", doc_id=doc_id, detail=chunk_id)
            if actual_tokens > int(manifest["max_passage_tokens"]):
                _add_error(hard_errors, "retrieval_text_over_token_budget", doc_id=doc_id, detail=chunk_id)
            if chunk.get("split_reason") == "oversized_leaf_window":
                warning_counts["oversized_leaf_window"] += 1
            if chunk.get("synthetic"):
                continue
            start, end = chunk.get("start"), chunk.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(passage)):
                _add_error(hard_errors, "invalid_chunk_span", doc_id=doc_id, detail=chunk_id)
            elif passage[start:end] != chunk["raw_text"]:
                _add_error(hard_errors, "chunk_round_trip_mismatch", doc_id=doc_id, detail=chunk_id)

        expected_ids = doc_index.get(doc_id, [])
        if expected_ids != [chunk["chunk_id"] for chunk in chunks]:
            _add_error(hard_errors, "doc_index_order_mismatch", doc_id=doc_id)
        for scope_id in document.get("scope_node_ids", []):
            if scope_id not in node_map:
                _add_error(hard_errors, "broken_scope_reference", doc_id=doc_id, detail=scope_id)

        if passage.strip():
            gaps = _coverage_gaps(passage, chunks)
            if gaps:
                _add_error(hard_errors, "lost_non_whitespace_span", doc_id=doc_id, detail=gaps[:5])
            overlaps = _unexpected_overlaps(chunks)
            if overlaps:
                _add_error(hard_errors, "unexpected_chunk_overlap", doc_id=doc_id, detail=overlaps[:5])

        if doc_number % 500 == 0 or doc_number == len(documents):
            print(f"Audited {doc_number}/{len(documents)} documents", flush=True)

    if next_nodes is not None or next_chunks is not None:
        _add_error(hard_errors, "orphan_jsonl_document_group")

    with train_path.open("r", encoding="utf-8") as handle:
        train = json.load(handle)
    truth_docs = {str(doc_id) for item in train.values() for doc_id in item["answer"]}
    missing_truth = sorted(truth_docs - represented_docs)
    if missing_truth:
        _add_error(hard_errors, "ground_truth_documents_not_represented", detail=missing_truth[:100])

    if compare_cache is not None:
        with (compare_cache / "manifest.json").open("r", encoding="utf-8") as handle:
            other_manifest = json.load(handle)
        if manifest["content_fingerprint"] != other_manifest.get("content_fingerprint"):
            _add_error(
                hard_errors,
                "non_deterministic_build",
                detail={
                    "current": manifest["content_fingerprint"],
                    "other": other_manifest.get("content_fingerprint"),
                },
            )

    # Limit verbose error material while preserving exact aggregate counts.
    error_counts = Counter(error["code"] for error in hard_errors)
    sorted_tokens = sorted(token_counts)

    def percentile(percent: int) -> int:
        if not sorted_tokens:
            return 0
        return sorted_tokens[round((len(sorted_tokens) - 1) * percent / 100)]

    largest_docs = sorted(chunks_per_doc.items(), key=lambda item: (-item[1], item[0]))[:50]
    for doc_id, count in largest_docs[: min(10, len(largest_docs))]:
        anomaly_samples.append({"doc_id": doc_id, "reason": "high_chunk_count", "detail": count})
    status = "PASS" if not hard_errors else "FAIL"
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": status,
        "chunk_schema_version": manifest.get("schema_version"),
        "content_fingerprint": manifest.get("content_fingerprint"),
        "hard_error_counts": dict(sorted(error_counts.items())),
        "hard_error_samples": hard_errors[:200],
        "warning_counts": dict(sorted(warning_counts.items())),
        "counts": {
            "source_documents": len(source_paths),
            "cache_documents": len(documents),
            "represented_documents": len(represented_docs),
            "nodes": len(seen_node_ids),
            "chunks": len(seen_chunk_ids),
            "ground_truth_documents": len(truth_docs),
            "ground_truth_documents_represented": len(truth_docs - set(missing_truth)),
            "parse_modes": dict(sorted(parse_modes.items())),
            "split_reasons": dict(sorted(split_reasons.items())),
        },
        "token_percentiles": {
            str(percent): percentile(percent)
            for percent in (0, 25, 50, 75, 90, 95, 99, 100)
        },
        "largest_documents_by_chunk_count": [
            {"doc_id": doc_id, "chunks": count} for doc_id, count in largest_docs
        ],
        "legacy_cache_summary": {
            "v1": _old_cache_summary(old_v1_path) if old_v1_path else None,
            "v2": _old_cache_summary(old_v2_path) if old_v2_path else None,
        },
    }

    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (output_dir / "audit_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in anomaly_samples[:50]:
            handle.write(canonical_json(sample) + "\n")

    legacy = report["legacy_cache_summary"]
    markdown = [
        "# Structural Chunker v3 Audit",
        "",
        f"**Status:** `{status}`",
        "",
        "## Hard invariants",
        "",
        "| Check | Count |",
        "|---|---:|",
    ]
    if error_counts:
        markdown.extend(f"| `{code}` | {count} |" for code, count in sorted(error_counts.items()))
    else:
        markdown.append("| All hard invariants | 0 failures |")
    markdown.extend(
        [
            "",
            "## Corpus summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Source documents | {len(source_paths)} |",
            f"| Represented documents | {len(represented_docs)} |",
            f"| Nodes | {len(seen_node_ids)} |",
            f"| Chunks | {len(seen_chunk_ids)} |",
            f"| Structured documents | {parse_modes.get('structured', 0)} |",
            f"| Fallback documents | {parse_modes.get('fallback', 0)} |",
            f"| Oversized-leaf windows | {warning_counts.get('oversized_leaf_window', 0)} |",
            "",
            "## Token percentiles",
            "",
            "| P0 | P25 | P50 | P75 | P90 | P95 | P99 | P100 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| " + " | ".join(str(report["token_percentiles"][str(p)]) for p in (0,25,50,75,90,95,99,100)) + " |",
            "",
            "## Legacy comparison",
            "",
            "| Cache | Documents | Chunks |",
            "|---|---:|---:|",
            f"| v1 | {(legacy['v1'] or {}).get('documents', 'n/a')} | {(legacy['v1'] or {}).get('chunks', 'n/a')} |",
            f"| v2 | {(legacy['v2'] or {}).get('documents', 'n/a')} | {(legacy['v2'] or {}).get('chunks', 'n/a')} |",
            f"| v3 | {len(documents)} | {len(seen_chunk_ids)} |",
            "",
            "## Top anomalies",
            "",
        ]
    )
    markdown.extend(
        f"- `{sample['doc_id']}` — {sample['reason']}: {sample['detail']}"
        for sample in anomaly_samples[:50]
    )
    (output_dir / "audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--contexts-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, default=8532)
    parser.add_argument("--compare-cache", type=Path)
    parser.add_argument("--old-v1", type=Path)
    parser.add_argument("--old-v2", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    with (args.cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tokenizer = HuggingFaceTokenizerAdapter.load(
        args.tokenizer or manifest.get("tokenizer", DEFAULT_TOKENIZER),
        allow_download=args.allow_download,
    )
    report = audit_cache(
        args.cache_dir,
        args.contexts_dir,
        args.train_file,
        args.output_dir,
        tokenizer,
        expected_documents=args.expected_documents,
        compare_cache=args.compare_cache,
        old_v1_path=args.old_v1,
        old_v2_path=args.old_v2,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
