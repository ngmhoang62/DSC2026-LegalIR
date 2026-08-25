"""Read-only structural and query audit for the final LegalIR sparse branch."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file
from exp012b_bm25 import default_segmenter


SCHEMA = "legalir.exp021_sparse_audit.v1"
LEGAL_IDENTIFIER = re.compile(
    r"\b(?:điều|khoản|mục|chương)\s*\d+[\w.-]*|\b\d{1,4}/\d{2,4}/[\w-]+\b",
    flags=re.IGNORECASE,
)


def _percentiles(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {str(value): 0 for value in (0, 25, 50, 75, 90, 95, 99, 100)}
    return {
        str(value): ordered[round((len(ordered) - 1) * value / 100)]
        for value in (0, 25, 50, 75, 90, 95, 99, 100)
    }


def audit_sparse_inputs(
    *, v3_dir: Path, processed_contexts: Path, preprocessing_manifest: Path,
    train_path: Path, output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads((v3_dir / "manifest.json").read_text(encoding="utf-8"))
    if not (v3_dir / "_SUCCESS.json").exists():
        raise RuntimeError("structural corpus is incomplete")
    if manifest.get("source_manifest_sha256") != sha256_file(preprocessing_manifest):
        raise RuntimeError("structural corpus is not bound to this preprocessing manifest")
    processed = json.loads(preprocessing_manifest.read_text(encoding="utf-8"))
    docs = list(read_jsonl(v3_dir / "documents.jsonl"))
    per_doc_chunks: Counter[str] = Counter()
    chunk_count = 0
    for row in read_jsonl(v3_dir / "chunks.jsonl"):
        per_doc_chunks[str(row["doc_id"])] += 1
        chunk_count += 1
    if len(docs) != int(manifest["counts"]["documents"]) or chunk_count != int(manifest["counts"]["chunks"]):
        raise RuntimeError("structural artifact count mismatch")
    context_count = sum(1 for _ in processed_contexts.glob("context_*.json"))
    if context_count != int(processed["retained_context_count"]):
        raise RuntimeError("processed context count disagrees with manifest")
    provenance = {
        "supplied": context_count - int(processed["retained_derived_name_count"]),
        "derived_from_link_final_path_segment": int(processed["retained_derived_name_count"]),
    }
    empty_labels = 0
    for row in docs:
        if not str(row.get("document_label", "")).strip():
            empty_labels += 1
    train = json.loads(train_path.read_text(encoding="utf-8"))
    query_tokens: Counter[str] = Counter()
    identifier_queries = 0
    for row in train.values():
        query = str(row["question"])
        query_tokens.update(default_segmenter(query).split())
        identifier_queries += int(bool(LEGAL_IDENTIFIER.search(query)))
    report = {
        "schema_version": SCHEMA,
        "structural_fingerprint": manifest["content_fingerprint"],
        "preprocessing_fingerprint": processed["content_fingerprint"],
        "counts": {
            "documents": len(docs), "chunks": chunk_count, "queries": len(train),
            "empty_document_labels": empty_labels, "scope_documents": sum(bool(row.get("scope_node_ids")) for row in docs),
            "structured_documents": sum(row.get("parse_mode") == "structured" for row in docs),
            "fallback_documents": sum(row.get("parse_mode") == "fallback" for row in docs),
            "queries_with_legal_identifier": identifier_queries,
            "unique_segmented_query_tokens": len(query_tokens),
        },
        "name_provenance": provenance,
        "chunk_count_per_document_percentiles": _percentiles(per_doc_chunks.values()),
        "e5_retrieval_text_token_percentiles": manifest.get("token_percentiles", {}),
        "query_token_frequency_top50": query_tokens.most_common(50),
        "planned_baseline": {
            "unit": "structural_chunk", "field_mode": "passage_only",
            "segmenter": "underthesea", "retriever": "SQLite FTS5 BM25",
            "aggregation": "deferred; first measure chunk-level candidate coverage",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "sparse_input_audit.json", report)
    (output_dir / "_SUCCESS.json").write_text(
        canonical_json({"stage": "sparse-input-audit", "structural_fingerprint": manifest["content_fingerprint"]}) + "\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--processed-contexts", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_sparse_inputs(
        v3_dir=args.v3_dir,
        processed_contexts=args.processed_contexts,
        preprocessing_manifest=args.preprocessing_manifest,
        train_path=args.train_file,
        output_dir=args.output_dir,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
