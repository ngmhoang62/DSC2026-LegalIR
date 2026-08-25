"""Measure whether the final document-label field has usable lexical signal."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from exp012b_bm25 import default_segmenter
from exp012b_core import atomic_json, read_jsonl


def _has_vietnamese_diacritic(text: str) -> bool:
    normalized = unicodedata.normalize("NFD", text)
    return "đ" in text.casefold() or any(unicodedata.combining(char) for char in normalized)


def _fold_diacritics(text: str) -> str:
    return "".join(
        "d" if char.casefold() == "đ" else char
        for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char)
    )


def audit_title_lexicon(*, v3_dir: Path, train_path: Path, output_dir: Path) -> dict[str, Any]:
    documents = {str(row["doc_id"]): row for row in read_jsonl(v3_dir / "documents.jsonl")}
    train = json.loads(train_path.read_text(encoding="utf-8"))
    titles = {doc_id: str(row.get("document_label", "")) for doc_id, row in documents.items()}
    title_tokens = {doc_id: set(default_segmenter(value).split()) for doc_id, value in titles.items()}
    folded_title_tokens = {
        doc_id: set(_fold_diacritics(token) for token in tokens)
        for doc_id, tokens in title_tokens.items()
    }
    present_gold_pairs = 0
    missing_gold_pairs = 0
    any_overlap_pairs = 0
    overlap_sizes: list[int] = []
    queries_with_any_gold_overlap = 0
    folded_any_overlap_pairs = 0
    folded_queries_with_any_gold_overlap = 0
    folded_overlap_sizes: list[int] = []
    for record in train.values():
        query_tokens = set(default_segmenter(str(record["question"])).split())
        folded_query_tokens = set(_fold_diacritics(token) for token in query_tokens)
        overlaps = []
        folded_overlaps = []
        for doc_id in map(str, record["answer"]):
            if doc_id not in title_tokens:
                missing_gold_pairs += 1
                continue
            present_gold_pairs += 1
            overlap = query_tokens & title_tokens[doc_id]
            overlap_sizes.append(len(overlap))
            any_overlap_pairs += int(bool(overlap))
            overlaps.append(overlap)
            folded_overlap = folded_query_tokens & folded_title_tokens[doc_id]
            folded_any_overlap_pairs += int(bool(folded_overlap))
            folded_overlap_sizes.append(len(folded_overlap))
            folded_overlaps.append(folded_overlap)
        queries_with_any_gold_overlap += int(any(overlaps))
        folded_queries_with_any_gold_overlap += int(any(folded_overlaps))
    populated = [value for value in titles.values() if value.strip()]
    report = {
        "documents": len(documents),
        "document_labels": {
            "nonempty": len(populated),
            "with_vietnamese_diacritic": sum(_has_vietnamese_diacritic(value) for value in populated),
            "without_vietnamese_diacritic": sum(not _has_vietnamese_diacritic(value) for value in populated),
        },
        "queries": len(train),
        "queries_with_vietnamese_diacritic": sum(_has_vietnamese_diacritic(str(row["question"])) for row in train.values()),
        "gold_title_token_overlap": {
            "present_gold_pairs": present_gold_pairs,
            "missing_gold_pairs": missing_gold_pairs,
            "pairs_with_any_overlap": any_overlap_pairs,
            "pair_coverage": any_overlap_pairs / present_gold_pairs if present_gold_pairs else 0.0,
            "mean_overlap_tokens": sum(overlap_sizes) / len(overlap_sizes) if overlap_sizes else 0.0,
            "queries_with_any_gold_overlap": queries_with_any_gold_overlap,
            "query_coverage": queries_with_any_gold_overlap / len(train) if train else 0.0,
        },
        "gold_title_token_overlap_after_diacritic_folding": {
            "pairs_with_any_overlap": folded_any_overlap_pairs,
            "pair_coverage": folded_any_overlap_pairs / present_gold_pairs if present_gold_pairs else 0.0,
            "mean_overlap_tokens": sum(folded_overlap_sizes) / len(folded_overlap_sizes) if folded_overlap_sizes else 0.0,
            "queries_with_any_gold_overlap": folded_queries_with_any_gold_overlap,
            "query_coverage": folded_queries_with_any_gold_overlap / len(train) if train else 0.0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "title_lexicon_audit.json", report)
    return report
