"""Deterministic structural chunking for the LegalIR EXP-012b pipeline.

The parser deliberately recognises legal headings only at physical line starts.
This prevents references such as ``theo Điều 12`` from becoming fake articles.
All source-backed nodes and chunks retain exact character offsets into the
original passage so the companion audit can prove round-trip integrity.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

from transformers import AutoTokenizer


SCHEMA_VERSION = "legalir.structural_chunks.v3"
DEFAULT_TOKENIZER = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_PASSAGE_TOKENS = 384
DEFAULT_TOKEN_WINDOW = 352
DEFAULT_TOKEN_OVERLAP = 32
SOURCE_KINDS = ("chapter", "section", "article", "annex")


class TokenizerLike(Protocol):
    name_or_path: str

    def token_length(self, text: str) -> int: ...

    def token_lengths(self, texts: Sequence[str]) -> list[int]: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def windows(
        self, text: str, max_tokens: int, overlap: int
    ) -> list[tuple[int, int]]: ...


class HuggingFaceTokenizerAdapter:
    """Small deterministic wrapper around a fast Hugging Face tokenizer."""

    def __init__(self, tokenizer: Any, requested_name: str):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Structural chunking requires a fast tokenizer")
        self.tokenizer = tokenizer
        self.name_or_path = requested_name

    @classmethod
    def load(
        cls, model_name_or_path: str, *, allow_download: bool = False
    ) -> "HuggingFaceTokenizerAdapter":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=not allow_download,
            use_fast=True,
        )
        return cls(tokenizer, model_name_or_path)

    def _offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if int(end) > int(start)
        ]

    def token_length(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            padding=False,
            verbose=False,
        )
        return [len(row) for row in encoded["input_ids"]]

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        offsets = self._offsets(text)
        if len(offsets) <= max_tokens:
            return text.strip()
        return text[: offsets[max_tokens - 1][1]].strip()

    def windows(
        self, text: str, max_tokens: int, overlap: int
    ) -> list[tuple[int, int]]:
        if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
            raise ValueError("Require max_tokens > overlap >= 0")
        offsets = self._offsets(text)
        if not offsets:
            return []
        if len(offsets) <= max_tokens:
            return [(offsets[0][0], offsets[-1][1])]
        stride = max_tokens - overlap
        result: list[tuple[int, int]] = []
        token_start = 0
        while token_start < len(offsets):
            token_end = min(token_start + max_tokens, len(offsets))
            result.append((offsets[token_start][0], offsets[token_end - 1][1]))
            if token_end == len(offsets):
                break
            token_start += stride
        return result


@dataclass(frozen=True)
class Marker:
    kind: str
    start: int
    heading_end: int
    label: str
    heading_text: str


@dataclass
class Node:
    node_id: str
    doc_id: str
    kind: str
    label: str
    parent_id: str | None
    start: int | None
    end: int | None
    heading_text: str
    raw_text: str
    synthetic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "kind": self.kind,
            "label": self.label,
            "parent_id": self.parent_id,
            "start": self.start,
            "end": self.end,
            "heading_text": self.heading_text,
            "raw_text": self.raw_text,
            "synthetic": self.synthetic,
        }


_CHAPTER_RE = re.compile(
    r"(?mi)^[ \t]*Chương[ \t]+(?P<label>(?:[IVXLCDM]+|\d+))\b(?P<tail>[^\r\n]*)"
)
_SECTION_RE = re.compile(
    r"(?mi)^[ \t]*Mục[ \t]+(?P<label>\d+[a-zđ]?)(?P<tail>[^\r\n]*)"
)
_ARTICLE_RE = re.compile(
    r"(?mi)^[ \t]*Điều[ \t]+(?P<label>\d+[a-zđ]?)(?:[ \t]*[\.:])?"
    r"(?=[ \t\r\n]|$)(?P<tail>[^\r\n]*)"
)
_CLAUSE_RE = re.compile(r"(?m)^[ \t]*(?P<label>\d+)\.[ \t]+")
_POINT_RE = re.compile(r"(?mi)^[ \t]*(?P<label>[a-zđ])\)[ \t]+")
_ANNEX_RE = re.compile(
    r"(?mi)^[ \t]*(?P<label>PHỤ[ \t]+LỤC|DANH[ \t]+MỤC|MẪU)"
    r"(?P<tail>[^\r\n]{0,140})"
)
_PARAGRAPH_SEPARATOR_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")
_SCOPE_RE = re.compile(r"(?i)\b(phạm\s+vi|đối\s+tượng\s+áp\s+dụng)\b")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def stable_node_id(doc_id: str, kind: str, start: int, end: int) -> str:
    return f"{doc_id}:{kind}:{start:09d}:{end:09d}"


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _line_heading(match: re.Match[str], kind: str) -> Marker:
    label = _compact_whitespace(match.group("label"))
    heading = _compact_whitespace(match.group(0))
    return Marker(kind, match.start(), match.end(), label, heading)


def find_structural_markers(passage: str) -> list[Marker]:
    markers: list[Marker] = []
    markers.extend(_line_heading(match, "chapter") for match in _CHAPTER_RE.finditer(passage))
    markers.extend(_line_heading(match, "section") for match in _SECTION_RE.finditer(passage))
    markers.extend(_line_heading(match, "article") for match in _ARTICLE_RE.finditer(passage))
    for match in _ANNEX_RE.finditer(passage):
        line = _compact_whitespace(match.group(0))
        # Annex-like prose is common. Requiring an uppercase physical line keeps
        # this marker conservative while preserving real headers.
        if line == line.upper():
            markers.append(
                Marker("annex", match.start(), match.end(), line[:80], line)
            )
    priority = {"annex": 0, "chapter": 1, "section": 2, "article": 3}
    markers.sort(key=lambda marker: (marker.start, priority[marker.kind]))
    deduplicated: list[Marker] = []
    occupied: set[int] = set()
    for marker in markers:
        if marker.start not in occupied:
            deduplicated.append(marker)
            occupied.add(marker.start)
    return deduplicated


def _extend_heading(passage: str, marker: Marker, boundary: int) -> str:
    """Attach at most one short following title line to Chương/Mục headings."""
    if marker.kind not in {"chapter", "section", "annex"}:
        return marker.heading_text
    tail = passage[marker.heading_end : min(boundary, marker.heading_end + 300)]
    lines = [_compact_whitespace(line) for line in tail.splitlines() if line.strip()]
    if lines and len(lines[0]) <= 160:
        return _compact_whitespace(f"{marker.heading_text} {lines[0]}")
    return marker.heading_text


def build_structural_nodes(doc_id: str, passage: str) -> tuple[list[Node], list[Node]]:
    markers = find_structural_markers(passage)
    root_id = stable_node_id(doc_id, "document", 0, len(passage))
    root = Node(root_id, doc_id, "document", doc_id, None, 0, len(passage), "", passage)
    nodes = [root]
    marker_nodes: list[Node] = []
    current_chapter: str | None = None
    current_section: str | None = None

    # Precompute the next boundary for each hierarchy level in one reverse
    # pass. The previous implementation searched the remaining marker list for
    # every node, which became quadratic on consolidated laws with hundreds of
    # headings.
    next_chapter_or_annex: list[int | None] = [None] * len(markers)
    next_section_boundary: list[int | None] = [None] * len(markers)
    next_annex: list[int | None] = [None] * len(markers)
    chapter_boundary: int | None = None
    section_boundary: int | None = None
    annex_boundary: int | None = None
    for index in range(len(markers) - 1, -1, -1):
        next_chapter_or_annex[index] = chapter_boundary
        next_section_boundary[index] = section_boundary
        next_annex[index] = annex_boundary
        kind = markers[index].kind
        if kind in {"chapter", "annex"}:
            chapter_boundary = index
        if kind in {"chapter", "section", "annex"}:
            section_boundary = index
        if kind == "annex":
            annex_boundary = index

    for index, marker in enumerate(markers):
        next_start = markers[index + 1].start if index + 1 < len(markers) else len(passage)
        if marker.kind == "chapter":
            boundary = next_chapter_or_annex[index]
            end = markers[boundary].start if boundary is not None else len(passage)
            parent_id = root_id
        elif marker.kind == "section":
            boundary = next_section_boundary[index]
            end = markers[boundary].start if boundary is not None else len(passage)
            parent_id = current_chapter or root_id
        elif marker.kind == "article":
            end = next_start
            parent_id = current_section or current_chapter or root_id
        else:
            boundary = next_annex[index]
            end = markers[boundary].start if boundary is not None else len(passage)
            parent_id = root_id

        node_id = stable_node_id(doc_id, marker.kind, marker.start, end)
        node = Node(
            node_id=node_id,
            doc_id=doc_id,
            kind=marker.kind,
            label=marker.label,
            parent_id=parent_id,
            start=marker.start,
            end=end,
            heading_text=_extend_heading(passage, marker, next_start),
            raw_text=passage[marker.start:end],
        )
        nodes.append(node)
        marker_nodes.append(node)
        if marker.kind == "chapter":
            current_chapter = node_id
            current_section = None
        elif marker.kind == "section":
            current_section = node_id
        elif marker.kind == "annex":
            current_chapter = None
            current_section = None

    # Add Clause and Point nodes inside each article. They are structural
    # children and later act as preferred packing boundaries.
    for article in [node for node in marker_nodes if node.kind == "article"]:
        assert article.start is not None and article.end is not None
        article_text = passage[article.start : article.end]
        clauses = list(_CLAUSE_RE.finditer(article_text))
        for clause_index, match in enumerate(clauses):
            start = article.start + match.start()
            end = (
                article.start + clauses[clause_index + 1].start()
                if clause_index + 1 < len(clauses)
                else article.end
            )
            clause_id = stable_node_id(doc_id, "clause", start, end)
            clause = Node(
                clause_id,
                doc_id,
                "clause",
                match.group("label"),
                article.node_id,
                start,
                end,
                _compact_whitespace(match.group(0)),
                passage[start:end],
            )
            nodes.append(clause)
            clause_text = passage[start:end]
            points = list(_POINT_RE.finditer(clause_text))
            for point_index, point_match in enumerate(points):
                point_start = start + point_match.start()
                point_end = (
                    start + points[point_index + 1].start()
                    if point_index + 1 < len(points)
                    else end
                )
                nodes.append(
                    Node(
                        stable_node_id(doc_id, "point", point_start, point_end),
                        doc_id,
                        "point",
                        point_match.group("label"),
                        clause_id,
                        point_start,
                        point_end,
                        _compact_whitespace(point_match.group(0)),
                        passage[point_start:point_end],
                    )
                )

    nodes.sort(
        key=lambda node: (
            -1 if node.start is None else node.start,
            -(len(passage) if node.end is None else node.end),
            node.kind,
        )
    )
    return nodes, marker_nodes


def _deepest_context_node(nodes: Sequence[Node], start: int, end: int) -> Node:
    candidates = [
        node
        for node in nodes
        if node.start is not None
        and node.end is not None
        and node.start <= start
        and node.end >= end
    ]
    return min(candidates, key=lambda node: (node.end - node.start, node.kind))


def _metadata_for_node(nodes_by_id: dict[str, Node], node: Node) -> dict[str, str]:
    result = {"chapter": "", "section": "", "article": ""}
    current: Node | None = node
    while current is not None:
        if current.kind in result and not result[current.kind]:
            result[current.kind] = current.heading_text
        current = nodes_by_id.get(current.parent_id or "")
    return result


def _render_prefix(
    tokenizer: TokenizerLike, document_label: str, metadata: dict[str, str]
) -> str:
    fields = [
        ("Văn bản", document_label, 48),
        ("Chương", metadata.get("chapter", ""), 32),
        ("Mục", metadata.get("section", ""), 24),
        ("Điều", metadata.get("article", ""), 64),
    ]
    lines = []
    for label, value, budget in fields:
        value = _compact_whitespace(value)
        if value:
            lines.append(f"[{label}] {tokenizer.truncate(value, budget)}")
    return "\n".join(lines)


def _paragraph_spans(text: str, absolute_start: int) -> list[tuple[int, int]]:
    # Split on blank-line separators in a single linear scan. A previous
    # non-greedy DOTALL expression exhibited pathological behaviour on the
    # corpus' largest table/annex documents.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in _PARAGRAPH_SEPARATOR_RE.finditer(text):
        raw_start, raw_end = cursor, separator.start()
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_start < raw_end:
            spans.append((absolute_start + raw_start, absolute_start + raw_end))
        cursor = separator.end()
    raw_start, raw_end = cursor, len(text)
    while raw_start < raw_end and text[raw_start].isspace():
        raw_start += 1
    while raw_end > raw_start and text[raw_end - 1].isspace():
        raw_end -= 1
    if raw_start < raw_end:
        spans.append((absolute_start + raw_start, absolute_start + raw_end))
    return spans


def _preferred_units(
    passage: str, region: Node, children_by_parent: dict[str, list[Node]]
) -> list[tuple[int, int]]:
    assert region.start is not None and region.end is not None
    if region.kind != "article":
        return _paragraph_spans(region.raw_text, region.start)
    clauses = sorted(
        [node for node in children_by_parent.get(region.node_id, []) if node.kind == "clause"],
        key=lambda node: int(node.start or 0),
    )
    if not clauses:
        return [(region.start, region.end)]
    units: list[tuple[int, int]] = []
    if region.start < int(clauses[0].start):
        units.append((region.start, int(clauses[0].start)))
    units.extend((int(node.start), int(node.end)) for node in clauses)
    return units


def _split_oversized_span(
    passage: str,
    span: tuple[int, int],
    prefix: str,
    tokenizer: TokenizerLike,
    max_tokens: int,
    token_window: int,
    token_overlap: int,
) -> list[tuple[int, int, str]]:
    start, end = span
    text = passage[start:end]
    paragraphs = _paragraph_spans(text, start)
    if len(paragraphs) > 1:
        results: list[tuple[int, int, str]] = []
        current: tuple[int, int] | None = None
        for paragraph in paragraphs:
            candidate = paragraph if current is None else (current[0], paragraph[1])
            rendered = f"{prefix}\n[Nội dung] {passage[candidate[0]:candidate[1]]}".strip()
            if tokenizer.token_length(rendered) <= max_tokens:
                current = candidate
                continue
            if current is not None:
                results.append((current[0], current[1], "paragraph_pack"))
            single_rendered = f"{prefix}\n[Nội dung] {passage[paragraph[0]:paragraph[1]]}".strip()
            if tokenizer.token_length(single_rendered) <= max_tokens:
                current = paragraph
            else:
                results.extend(
                    _split_oversized_span(
                        passage,
                        paragraph,
                        prefix,
                        tokenizer,
                        max_tokens,
                        token_window,
                        token_overlap,
                    )
                )
                current = None
        if current is not None:
            results.append((current[0], current[1], "paragraph_pack"))
        return results

    prefix_cost = tokenizer.token_length(f"{prefix}\n[Nội dung]")
    content_budget = max(1, min(token_window, max_tokens - prefix_cost))
    while True:
        overlap = min(token_overlap, max(0, content_budget - 1))
        windows = tokenizer.windows(text, content_budget, overlap)
        overflows = [
            tokenizer.token_length(
                f"{prefix}\n[Nội dung] {text[window_start:window_end]}".strip()
            )
            - max_tokens
            for window_start, window_end in windows
        ]
        maximum_overflow = max(overflows, default=0)
        if maximum_overflow <= 0:
            break
        # Subword tokenization at a string boundary is not additive: the
        # prefix and content can require one more token after concatenation.
        # Regenerate the complete sliding-window cover with a smaller content
        # budget instead of trimming a single chunk and leaving a source gap.
        content_budget -= maximum_overflow
        if content_budget <= 0:
            raise ValueError("Metadata prefix leaves no room for source content")
    return [
        (start + window_start, start + window_end, "oversized_leaf_window")
        for window_start, window_end in windows
    ]


def _pack_region(
    passage: str,
    region: Node,
    nodes_by_id: dict[str, Node],
    children_by_parent: dict[str, list[Node]],
    document_label: str,
    tokenizer: TokenizerLike,
    max_tokens: int,
    token_window: int,
    token_overlap: int,
) -> list[dict[str, Any]]:
    metadata = _metadata_for_node(nodes_by_id, region)
    prefix = _render_prefix(tokenizer, document_label, metadata)
    units = _preferred_units(passage, region, children_by_parent)
    packed: list[tuple[int, int, str]] = []
    current: tuple[int, int] | None = None
    for unit in units:
        candidate = unit if current is None else (current[0], unit[1])
        rendered = f"{prefix}\n[Nội dung] {passage[candidate[0]:candidate[1]]}".strip()
        if tokenizer.token_length(rendered) <= max_tokens:
            current = candidate
            continue
        if current is not None:
            packed.append((current[0], current[1], "structural_pack"))
        unit_rendered = f"{prefix}\n[Nội dung] {passage[unit[0]:unit[1]]}".strip()
        if tokenizer.token_length(unit_rendered) <= max_tokens:
            current = unit
        else:
            packed.extend(
                _split_oversized_span(
                    passage,
                    unit,
                    prefix,
                    tokenizer,
                    max_tokens,
                    token_window,
                    token_overlap,
                )
            )
            current = None
    if current is not None:
        packed.append((current[0], current[1], "structural_pack"))

    chunks: list[dict[str, Any]] = []
    for part_index, (start, end, split_reason) in enumerate(packed):
        raw_text = passage[start:end]
        if not raw_text.strip():
            continue
        retrieval_text = f"{prefix}\n[Nội dung] {raw_text}".strip()
        retrieval_token_count = tokenizer.token_length(retrieval_text)
        if retrieval_token_count > max_tokens:
            raise ValueError(
                f"Internal token budget failure for {region.node_id}: "
                f"{retrieval_token_count} > {max_tokens}"
            )
        chunks.append(
            {
                "schema_version": SCHEMA_VERSION,
                "chunk_id": f"{region.node_id}:p{part_index:03d}",
                "doc_id": region.doc_id,
                "parent_node_id": region.node_id,
                "source_kind": region.kind,
                "part_index": part_index,
                "start": start,
                "end": end,
                "raw_text": raw_text,
                "retrieval_text": retrieval_text,
                "token_count": retrieval_token_count,
                "split_reason": split_reason,
                "synthetic": False,
            }
        )
    return chunks


def _retrieval_regions(
    doc_id: str, passage: str, nodes: Sequence[Node], marker_nodes: Sequence[Node]
) -> list[Node]:
    # Articles are the retrieval regions. Annex nodes remain structural
    # boundaries: an article before an annex ends at the annex heading, while
    # annex-only content is emitted as a context gap. Including both an annex
    # and its nested articles here would create overlapping primary regions.
    primary = sorted(
        [node for node in marker_nodes if node.kind == "article"],
        key=lambda node: int(node.start or 0),
    )
    if not primary:
        root = next(node for node in nodes if node.kind == "document")
        return [
            Node(
                stable_node_id(doc_id, "fallback", 0, len(passage)),
                doc_id,
                "fallback",
                "fallback",
                root.node_id,
                0,
                len(passage),
                "",
                passage,
            )
        ]
    # Context gaps can only be parented by structural containers.  Clause and
    # point nodes may number in the thousands in table-heavy documents; asking
    # _deepest_context_node() to rescan them for every article gap turns an
    # otherwise linear parse into quadratic work without changing the answer.
    context_nodes = [
        node
        for node in nodes
        if node.kind in {"document", "chapter", "section", "annex"}
    ]
    regions: list[Node] = []
    cursor = 0
    for node in primary:
        assert node.start is not None and node.end is not None
        if cursor < node.start and passage[cursor:node.start].strip():
            context = _deepest_context_node(context_nodes, cursor, node.start)
            regions.append(
                Node(
                    stable_node_id(doc_id, "context", cursor, node.start),
                    doc_id,
                    "context",
                    "context",
                    context.node_id,
                    cursor,
                    node.start,
                    context.heading_text,
                    passage[cursor:node.start],
                )
            )
        regions.append(node)
        cursor = max(cursor, node.end)
    if cursor < len(passage) and passage[cursor:].strip():
        context = _deepest_context_node(context_nodes, cursor, len(passage))
        regions.append(
            Node(
                stable_node_id(doc_id, "context", cursor, len(passage)),
                doc_id,
                "context",
                "context",
                context.node_id,
                cursor,
                len(passage),
                context.heading_text,
                passage[cursor:],
            )
        )
    return sorted(regions, key=lambda node: int(node.start or 0))


def parse_document(
    document: dict[str, Any],
    tokenizer: TokenizerLike,
    *,
    max_tokens: int = DEFAULT_MAX_PASSAGE_TOKENS,
    token_window: int = DEFAULT_TOKEN_WINDOW,
    token_overlap: int = DEFAULT_TOKEN_OVERLAP,
    source_file: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    doc_id = str(document["id"])
    passage = str(document.get("passage") or "")
    name = str(document.get("name") or "").strip()
    link = str(document.get("link") or "").strip()
    document_label = name or Path(link).stem or doc_id
    source_sha256 = hashlib.sha256(passage.encode("utf-8")).hexdigest()

    if not passage.strip():
        synthetic_text = document_label
        node_id = f"{doc_id}:synthetic"
        node = Node(
            node_id,
            doc_id,
            "synthetic",
            "synthetic",
            None,
            None,
            None,
            synthetic_text,
            synthetic_text,
            True,
        )
        retrieval_text = f"[Văn bản] {tokenizer.truncate(document_label, max_tokens)}"
        chunk = {
            "schema_version": SCHEMA_VERSION,
            "chunk_id": f"{node_id}:p000",
            "doc_id": doc_id,
            "parent_node_id": node_id,
            "source_kind": "synthetic",
            "part_index": 0,
            "start": None,
            "end": None,
            "raw_text": synthetic_text,
            "retrieval_text": retrieval_text,
            "token_count": tokenizer.token_length(retrieval_text),
            "split_reason": "empty_passage_fallback",
            "synthetic": True,
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            "doc_id": doc_id,
            "name": name,
            "link": link,
            "document_label": document_label,
            "source_file": source_file,
            "source_sha256": source_sha256,
            "passage_length": len(passage),
            "parse_mode": "fallback",
            "scope_node_ids": [],
            "synthetic": True,
        }
        return record, [node.as_dict()], [chunk]

    nodes, marker_nodes = build_structural_nodes(doc_id, passage)
    regions = _retrieval_regions(doc_id, passage, nodes, marker_nodes)
    existing_ids = {node.node_id for node in nodes}
    for region in regions:
        if region.node_id not in existing_ids:
            nodes.append(region)
            existing_ids.add(region.node_id)
    chunks: list[dict[str, Any]] = []
    nodes_by_id = {node.node_id: node for node in nodes}
    children_by_parent: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        if node.parent_id:
            children_by_parent[node.parent_id].append(node)
    for region in regions:
        chunks.extend(
            _pack_region(
                passage,
                region,
                nodes_by_id,
                children_by_parent,
                document_label,
                tokenizer,
                max_tokens,
                token_window,
                token_overlap,
            )
        )

    articles = [node for node in marker_nodes if node.kind == "article"]
    scope_ids = [
        node.node_id
        for node in articles
        if _SCOPE_RE.search((node.heading_text + " " + node.raw_text[:400]))
    ]
    record = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "name": name,
        "link": link,
        "document_label": document_label,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "passage_length": len(passage),
        "parse_mode": "structured" if articles else "fallback",
        "scope_node_ids": scope_ids,
        "synthetic": False,
    }
    nodes.sort(
        key=lambda node: (
            -1 if node.start is None else node.start,
            0 if node.end is None else -node.end,
            node.kind,
            node.node_id,
        )
    )
    chunks.sort(
        key=lambda chunk: (
            -1 if chunk["start"] is None else chunk["start"],
            chunk["chunk_id"],
        )
    )
    return record, [node.as_dict() for node in nodes], chunks


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
            count += 1
    return count


def build_corpus(
    contexts_dir: Path,
    output_dir: Path,
    tokenizer: TokenizerLike,
    *,
    max_tokens: int = DEFAULT_MAX_PASSAGE_TOKENS,
    token_window: int = DEFAULT_TOKEN_WINDOW,
    token_overlap: int = DEFAULT_TOKEN_OVERLAP,
    workers: int = 1,
) -> dict[str, Any]:
    paths = sorted(contexts_dir.glob("context_*.json"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No context_*.json files found in {contexts_dir}")
    if token_window > max_tokens:
        raise ValueError("token_window cannot exceed max_tokens")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    worker_state = threading.local()

    def tokenizer_for_worker() -> TokenizerLike:
        if workers == 1 or not isinstance(tokenizer, HuggingFaceTokenizerAdapter):
            return tokenizer
        thread_tokenizer = getattr(worker_state, "tokenizer", None)
        if thread_tokenizer is None:
            # Fast-tokenizer instances can serialize concurrent calls through
            # internal state. A local-cache clone per worker restores genuine
            # document-level parallelism without loading model weights.
            thread_tokenizer = HuggingFaceTokenizerAdapter.load(
                tokenizer.name_or_path, allow_download=False
            )
            worker_state.tokenizer = thread_tokenizer
        return thread_tokenizer

    def parse_path(
        path: Path,
    ) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        record, nodes, chunks = parse_document(
            document,
            tokenizer_for_worker(),
            max_tokens=max_tokens,
            token_window=token_window,
            token_overlap=token_overlap,
            source_file=path.name,
        )
        return path, record, nodes, chunks

    def parsed_documents() -> Iterator[
        tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]
    ]:
        if workers == 1:
            for path in paths:
                yield parse_path(path)
            return
        # Submit small, ordered batches instead of all 8.5k documents. This
        # bounds memory while future.result() preserves byte-stable file order.
        batch_size = workers * 2
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch_start in range(0, len(paths), batch_size):
                batch = paths[batch_start : batch_start + batch_size]
                futures = [executor.submit(parse_path, path) for path in batch]
                for future in futures:
                    yield future.result()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent))
    )
    try:
        documents_path = temp_dir / "documents.jsonl"
        nodes_path = temp_dir / "nodes.jsonl"
        chunks_path = temp_dir / "chunks.jsonl"
        index_path = temp_dir / "doc_to_chunk_ids.json"

        doc_to_chunk_ids: dict[str, list[str]] = {}
        parse_modes: Counter[str] = Counter()
        split_reasons: Counter[str] = Counter()
        token_counts: list[int] = []
        document_count = 0
        node_count = 0
        chunk_count = 0

        # Stream large artifacts. The old JSON cache is hundreds of MiB; keeping
        # every node and chunk as Python dictionaries would multiply its memory
        # footprint without improving determinism.
        with (
            documents_path.open("w", encoding="utf-8", newline="\n") as document_handle,
            nodes_path.open("w", encoding="utf-8", newline="\n") as node_handle,
            chunks_path.open("w", encoding="utf-8", newline="\n") as chunk_handle,
        ):
            for number, (path, record, nodes, chunks) in enumerate(
                parsed_documents(), start=1
            ):
                if number == 1:
                    print(f"Parsing 1/{len(paths)} documents", flush=True)
                document_handle.write(canonical_json(record) + "\n")
                for node in nodes:
                    node_handle.write(canonical_json(node) + "\n")
                for chunk in chunks:
                    chunk_handle.write(canonical_json(chunk) + "\n")
                document_count += 1
                node_count += len(nodes)
                chunk_count += len(chunks)
                doc_to_chunk_ids[record["doc_id"]] = [
                    chunk["chunk_id"] for chunk in chunks
                ]
                parse_modes[record["parse_mode"]] += 1
                split_reasons.update(chunk["split_reason"] for chunk in chunks)
                token_counts.extend(int(chunk["token_count"]) for chunk in chunks)
                if number % 100 == 0 or number == len(paths):
                    print(f"Parsed {number}/{len(paths)} documents", flush=True)
        index_path.write_text(
            canonical_json(doc_to_chunk_ids) + "\n", encoding="utf-8", newline="\n"
        )

        artifact_hashes = {
            path.name: sha256_file(path)
            for path in (documents_path, nodes_path, chunks_path, index_path)
        }
        config = {
            "schema_version": SCHEMA_VERSION,
            "tokenizer": tokenizer.name_or_path,
            "max_passage_tokens": max_tokens,
            "token_window": token_window,
            "token_overlap": token_overlap,
            "heading_policy": "physical_line_start_only",
        }
        fingerprint_payload = {
            "source_fingerprint": corpus_fingerprint(paths),
            "config": config,
            "artifacts": artifact_hashes,
        }
        content_fingerprint = hashlib.sha256(
            canonical_json(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        token_counts_sorted = sorted(token_counts)

        def percentile(percent: float) -> int:
            if not token_counts_sorted:
                return 0
            index = round((len(token_counts_sorted) - 1) * percent / 100)
            return token_counts_sorted[index]

        manifest = {
            **config,
            "source_fingerprint": fingerprint_payload["source_fingerprint"],
            "artifact_sha256": artifact_hashes,
            "content_fingerprint": content_fingerprint,
            "counts": {
                "documents": document_count,
                "nodes": node_count,
                "chunks": chunk_count,
                "parse_modes": dict(sorted(parse_modes.items())),
                "split_reasons": dict(sorted(split_reasons.items())),
            },
            "token_percentiles": {
                str(percent): percentile(percent)
                for percent in (0, 25, 50, 75, 90, 95, 99, 100)
            },
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            output_dir.replace(backup)
            temp_dir.replace(output_dir)
            shutil.rmtree(backup)
        else:
            temp_dir.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-passage-tokens", type=int, default=DEFAULT_MAX_PASSAGE_TOKENS)
    parser.add_argument("--token-window", type=int, default=DEFAULT_TOKEN_WINDOW)
    parser.add_argument("--token-overlap", type=int, default=DEFAULT_TOKEN_OVERLAP)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tokenizer = HuggingFaceTokenizerAdapter.load(
        args.tokenizer, allow_download=args.allow_download
    )
    manifest = build_corpus(
        args.contexts_dir,
        args.output_dir,
        tokenizer,
        max_tokens=args.max_passage_tokens,
        token_window=args.token_window,
        token_overlap=args.token_overlap,
        workers=args.workers,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
