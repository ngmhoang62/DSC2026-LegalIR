"""EXP-030: provenance-safe legal evidence routing and gated reranking.

This experiment consumes the frozen EXP-022 candidates and the fold-isolated
K=64 cascades.  It never retrieves a new parent document.  Its first job is to
repair the representation handed to a reranker: verified document metadata,
the actual ancestry of every evidence chunk, typed applicability clauses and
model-tokenizer-bounded Vietnamese views.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

# Transformers may be imported transitively by retrieval modules below and
# freezes this value at import time.  Point it at a workspace-writable cache
# before importing any project dependency.
_HF_MODULES_CACHE = Path(__file__).resolve().parent.parent / "cache" / "exp030_legal_evidence_routing" / "hf_modules"
os.environ["HF_MODULES_CACHE"] = str(_HF_MODULES_CACHE)

import numpy as np

from exp012b_core import (
    artifact_manifest,
    atomic_json,
    canonical_json,
    load_answers,
    load_v3_manifest,
    read_jsonl,
    require_success,
    sha256_file,
    write_jsonl,
)
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "legalir.exp030_legal_evidence_routing.v1"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
PREFLIGHT_CONTRACT = "workspace_dynamic_module_cache_v1"
SEED = 2030
K = 64
COMMON_MAX_LENGTH = 512
JINA_RUNTIME_CONTEXT = 4096
# Canonicalizing a duplicate gold changes pre-ranker labels, samples, training
# groups and metrics.  No previous EXP-030 run is resume-compatible.
HOTFIX_COMPATIBLE_RUNS: dict[str, str] = {
    # Only remote-code cache initialization changed. Label-dependent cascade,
    # enriched capsules and already completed non-Jina score jobs are valid;
    # preflight has its own contract fingerprint and is forced to rerun.
    "7fafa285a5456667c176d7d06ea4d55830d18ea713b82ab91416a4a06f7f4cd0": "jina_dynamic_module_cache_v1",
}
EXPECTED_LABEL_ACCOUNTING = {
    "queries": 7000,
    "evaluable_queries": 6991,
    "non_evaluable_queries": 9,
    "canonicalized_duplicate_occurrences": 2,
    "dropped_empty_occurrences": 11,
}
SCREEN_PER_INNER = 64
VIETNAMESE_MARKERS = (
    "[VĂN BẢN]",
    "[TÊN CHÍNH THỨC]",
    "[TÊN CHUẨN HÓA]",
    "[PHẠM VI ĐIỀU CHỈNH]",
    "[ĐỐI TƯỢNG ÁP DỤNG]",
    "[VỊ TRÍ TRONG VĂN BẢN]",
    "[BẰNG CHỨNG TRẢ LỜI]",
    "[QUAN HỆ PHÁP LÝ]",
)

LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "epochs": 3,
    "effective_batch": 32,
    "learning_rate": 1e-4,
}

_ACCENT_RE = re.compile(
    r"[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    r"ÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]"
)
_TYPE_PATTERN = (
    r"BỘ\s+LUẬT|LUẬT|PHÁP\s+LỆNH|NGHỊ\s+QUYẾT|NGHỊ\s+ĐỊNH|"
    r"THÔNG\s+TƯ(?:\s+LIÊN\s+TỊCH)?|QUYẾT\s+ĐỊNH|CHỈ\s+THỊ|"
    r"QUY\s+CHUẨN|TIÊU\s+CHUẨN"
)
_TYPE_LINE_RE = re.compile(rf"(?mi)^[ \t]*(?P<type>{_TYPE_PATTERN})[ \t]*(?P<tail>[^\r\n]*)$")
_NUMBER_RE = re.compile(
    r"(?i)\b(?:số\s*[:.]?\s*)?(?P<number>\d+[A-Za-zĐđ]?(?:/\d{4})?/[A-ZĐ-]{2,})\b"
)
_TITLE_STOP_RE = re.compile(
    r"(?mi)^[ \t]*(?:Căn\s+cứ|Theo\s+đề\s+nghị|Điều\s+1\b|Chương\s+I\b)"
)
_TITLE_TRAILING_STOP_RE = re.compile(
    r"(?mi)^[ \t]*(?:CỘNG\s+HÒA\b|Độc\s+lập\b|Số\s*:\s*|QUỐC\s+HỘI\b|"
    r"CHÍNH\s+PHỦ\b|ỦY\s+BAN\s+THƯỜNG\s+VỤ\s+QUỐC\s+HỘI\b|"
    r"THỦ\s+TƯỚNG\s+CHÍNH\s+PHỦ\b|BỘ\s+TRƯỞNG\b|TỔNG\s+CỤC\b)"
)
_ARTICLE_PREFIX_RE = re.compile(r"(?is)^\s*Điều\s+\d+[a-zđ]?\s*[.:]?\s*")
_COMBINED_SCOPE_RE = re.compile(
    r"(?i)^(?:"
    r"phạm\s+vi(?:\s+điều\s+chỉnh)?\s+và\s+đối\s+tượng(?:\s+áp\s+dụng|\s+điều\s+chỉnh)?|"
    r"đối\s+tượng\s+và\s+phạm\s+vi\s+áp\s+dụng"
    r")\b"
)
_REGULATION_SCOPE_RE = re.compile(r"(?i)^phạm\s+vi\s+(?:điều\s+chỉnh|áp\s+dụng)\b")
_SUBJECT_SCOPE_RE = re.compile(r"(?i)^đối\s+tượng\s+áp\s+dụng\b")
_SCOPE_SUBHEADING_RE = re.compile(
    r"(?i)(?:^|\s)(?P<number>\d+)\.\s*(?P<title>đối\s+tượng\s+áp\s+dụng|phạm\s+vi\s+(?:điều\s+chỉnh|áp\s+dụng))\s*[:.]?"
)
_RAW_SCOPE_SUBHEADING_RE = re.compile(
    r"(?im)^[ \t]*(?P<number>\d+)\.[ \t]*"
    r"(?P<title>đối[ \t]+tượng[ \t]+áp[ \t]+dụng|phạm[ \t]+vi[ \t]+(?:điều[ \t]+chỉnh|áp[ \t]+dụng))[ \t]*[:.]?"
)
_RELATION_RE = re.compile(
    rf"(?i)\b(?P<relation>sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|viện\s+dẫn|"
    rf"theo\s+quy\s+định\s+tại)?\s*(?P<type>{_TYPE_PATTERN})\s+(?:số\s+)?"
    rf"(?P<number>\d+[A-Za-zĐđ]?(?:/\d{{4}})?/[A-ZĐ-]{{2,}})\b"
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    kind: str
    native_max_length: int
    trust_remote_code: bool = False
    lora_family: str | None = None


MODELS = (
    ModelSpec("vietnamese", "AITeamVN/Vietnamese_Reranker", "pair", 2304, lora_family="xlmr"),
    ModelSpec("bge_m3", "BAAI/bge-reranker-v2-m3", "pair", 512, lora_family="xlmr"),
    ModelSpec("gte", "Alibaba-NLP/gte-multilingual-reranker-base", "pair", 8192, True, "discover"),
    ModelSpec("mmarco", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "pair", 512, lora_family="minilm"),
    ModelSpec("qwen3", "Qwen/Qwen3-Reranker-0.6B", "qwen", 32768, lora_family="qwen"),
    ModelSpec("jina", "jinaai/jina-reranker-v3.5", "jina", 131072, True, None),
)
MODEL_BY_KEY = {spec.key: spec for spec in MODELS}
MODEL_CARD_METADATA: dict[str, dict[str, Any]] = {
    "vietnamese": {"url": "https://huggingface.co/AITeamVN/Vietnamese_Reranker"},
    "bge_m3": {"url": "https://huggingface.co/BAAI/bge-reranker-v2-m3"},
    "gte": {"url": "https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base"},
    "mmarco": {"url": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"},
    "qwen3": {"url": "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B"},
    "jina": {
        "url": "https://huggingface.co/jinaai/jina-reranker-v3.5",
        "license": "CC BY-NC 4.0",
        "citation_bibtex": (
            "@misc{nasika2026jinarerankerv35,\n"
            "      title={jina-reranker-v3.5: Hybrid-Attention Listwise Reranking with Self-Distillation for Domain-Robust Retrieval},\n"
            "      author={Christina Nasika and Feng Wang and Antonis Minas Krasakis and Han Xiao},\n"
            "      year={2026},\n"
            "      eprint={2607.18152},\n"
            "      archivePrefix={arXiv},\n"
            "      primaryClass={cs.CL},\n"
            "      url={https://arxiv.org/abs/2607.18152},\n"
            "}"
        ),
    },
}

CAPSULE_CONFIGS: dict[str, dict[str, Any]] = {
    "evidence_only": {"identity_variant": "none", "structure": False, "scope": False, "relations": False, "view_policy": "base"},
    "unaccented_base": {"identity_variant": "unaccented", "structure": True, "scope": False, "relations": False, "view_policy": "base"},
    "title_base": {"identity_variant": "title", "structure": True, "scope": False, "relations": False, "view_policy": "base"},
    "both_base": {"identity_variant": "both", "structure": True, "scope": False, "relations": False, "view_policy": "base"},
    "typed_scope": {"identity_variant": "both", "structure": True, "scope": True, "relations": False, "view_policy": "routed"},
    "multi_view": {"identity_variant": "both", "structure": True, "scope": True, "relations": True, "view_policy": "routed"},
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_answers(
    train: Path, exclusions: Path, impact: Path,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Return EXP-030 labels after canonical duplicate aliasing.

    Empty source documents have no retrievable content and are dropped.  Exact
    duplicate identities are instead mapped to the retained identity recorded
    by preprocessing.  The impact sidecar remains an integrity check; it is
    never treated as a source from which to invent labels.
    """
    original = load_answers(train)
    exclusion_rows = _json(exclusions)
    by_doc = {str(row["doc_id"]): row for row in exclusion_rows}
    if len(by_doc) != len(exclusion_rows):
        raise ValueError("duplicate document id in preprocessing exclusions")
    impacted_rows = list(read_jsonl(impact))
    impacted_by_qid = {str(row["query_id"]): row for row in impacted_rows}
    if len(impacted_by_qid) != len(impacted_rows):
        raise ValueError("duplicate query id in train label impact")

    answers: dict[str, set[str]] = {}
    observed_impacts: dict[str, set[str]] = {}
    duplicate_occurrences = 0
    empty_occurrences = 0
    for qid, gold_ids in original.items():
        canonical: set[str] = set()
        removed: set[str] = set()
        for doc_id in gold_ids:
            exclusion = by_doc.get(str(doc_id))
            if exclusion is None:
                canonical.add(str(doc_id))
                continue
            removed.add(str(doc_id))
            reasons = {str(reason) for reason in exclusion.get("reasons", [])}
            replacement = exclusion.get("duplicate_retained_id")
            if "exact_duplicate_raw_passage" in reasons:
                if not replacement:
                    raise ValueError(f"duplicate gold has no retained alias: {qid}/{doc_id}")
                replacement = str(replacement)
                if replacement in by_doc:
                    raise ValueError(f"duplicate gold aliases another excluded document: {qid}/{doc_id}->{replacement}")
                canonical.add(replacement)
                duplicate_occurrences += 1
            elif reasons == {"empty_passage"}:
                empty_occurrences += 1
            else:
                raise ValueError(f"unsupported gold exclusion policy: {qid}/{doc_id}/{sorted(reasons)}")
        answers[str(qid)] = canonical
        if removed:
            observed_impacts[str(qid)] = removed

    declared_impacts = {
        qid: {str(value) for value in row.get("intentionally_excluded_gold_ids", [])}
        for qid, row in impacted_by_qid.items()
    }
    if observed_impacts != declared_impacts:
        raise ValueError("canonical-gold impact sidecar mismatch")
    non_evaluable = sorted(qid for qid, gold in answers.items() if not gold)
    stats = {
        "policy": LABEL_POLICY,
        "queries": len(answers),
        "evaluable_queries": len(answers) - len(non_evaluable),
        "non_evaluable_queries": len(non_evaluable),
        "non_evaluable_qids": non_evaluable,
        "affected_queries": len(observed_impacts),
        "canonicalized_duplicate_occurrences": duplicate_occurrences,
        "dropped_empty_occurrences": empty_occurrences,
        "label_fingerprint": _hash({qid: sorted(gold) for qid, gold in sorted(answers.items())}),
    }
    return answers, stats


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def accent_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text))
    value = "".join(char for char in normalized if not unicodedata.combining(char))
    return value.replace("đ", "d").replace("Đ", "D").lower()


def has_vietnamese_accent(text: str) -> bool:
    return bool(_ACCENT_RE.search(str(text)))


def normalized_tokens(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(accent_fold(text)) if len(token) > 1}


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def extract_official_title(passage: str, label: str = "") -> dict[str, Any]:
    """Extract a conservative, source-exact accented title from the header.

    `raw_text` is always exactly `passage[start:end]`.  `display_text` only
    compacts source whitespace and is never generated or diacritized.
    """
    header = passage[: min(len(passage), 4000)]
    header_boundary = len(header)
    boundary = _TITLE_STOP_RE.search(header)
    if boundary:
        header_boundary = boundary.start()
    for match in _TYPE_LINE_RE.finditer(header):
        document_type = _compact(match.group("type"))
        tail = _compact(match.group("tail"))
        source_type = match.group("type")
        # References inside prose are the main false-positive source.  An
        # official instrument heading is uppercase and precedes every Căn cứ /
        # Điều 1 boundary in the header.
        if source_type != source_type.upper() or match.start() >= min(header_boundary, 2500):
            continue
        if tail and tail != tail.upper():
            continue
        if accent_fold(document_type) not in accent_fold(label):
            continue
        start = match.start("tail") if tail else match.end()
        stop = _TITLE_STOP_RE.search(header, pos=start)
        trailing = _TITLE_TRAILING_STOP_RE.search(header, pos=start)
        stop_positions = [item.start() for item in (stop, trailing) if item]
        end = min(stop_positions) if stop_positions else min(len(header), start + 800)
        start, end = _trim_span(passage, start, end)
        raw = passage[start:end]
        display = _compact(raw.strip(" -–—.:;\r\n\t"))
        if tail:
            display = _compact(tail + " " + _compact(passage[match.end():end]))
            start = match.start("tail")
            start, end = _trim_span(passage, start, end)
            raw = passage[start:end]
        letters = [char for char in display if char.isalpha()]
        uppercase_ratio = sum(char.isupper() for char in letters) / max(1, len(letters))
        if (
            len(normalized_tokens(display)) < 3
            or not has_vietnamese_accent(display)
            or len(display) > 800
            or uppercase_ratio < 0.70
        ):
            continue
        label_tokens = normalized_tokens(label)
        title_tokens = normalized_tokens(display)
        descriptive = {x for x in label_tokens if not x.isdigit() and len(x) > 2}
        overlap = len(descriptive & title_tokens) / max(1, min(len(descriptive), len(title_tokens)))
        if descriptive and overlap < 0.50:
            continue
        number_match = _NUMBER_RE.search(header[: max(end, 1800)])
        return {
            "status": "VERIFIED",
            "document_type": document_type,
            "display_text": display,
            "raw_text": raw,
            "start": start,
            "end": end,
            "number": number_match.group("number") if number_match else None,
            "rule": "accented_header_exact_span_v1",
            "label_title_overlap": overlap,
        }
    return {"status": "MISSING", "reason": "no_verified_accented_header_title"}


def classify_scope_node(node: Mapping[str, Any]) -> str | None:
    prefix = _compact(str(node.get("raw_text") or "")[:260])
    prefix = _ARTICLE_PREFIX_RE.sub("", prefix)
    heading = _compact(str(node.get("heading_text") or ""))
    heading = _ARTICLE_PREFIX_RE.sub("", heading)
    probe = heading if len(heading) >= 12 else prefix
    if _COMBINED_SCOPE_RE.match(probe) or _COMBINED_SCOPE_RE.match(prefix):
        return "combined"
    if _REGULATION_SCOPE_RE.match(probe) or _REGULATION_SCOPE_RE.match(prefix):
        return "scope_of_regulation"
    if _SUBJECT_SCOPE_RE.match(probe) or _SUBJECT_SCOPE_RE.match(prefix):
        return "applicable_subjects"
    subheading = _SCOPE_SUBHEADING_RE.search(prefix)
    if subheading:
        return "applicable_subjects" if accent_fold(subheading.group("title")).startswith("doi tuong") else "scope_of_regulation"
    return None


def scope_display_span(node: Mapping[str, Any], kind: str) -> dict[str, Any]:
    raw = str(node.get("raw_text") or "")
    # Match the original string directly.  Accent folding and whitespace
    # compaction are deliberately forbidden here because either can invalidate
    # offsets into the source document.
    subheading = _RAW_SCOPE_SUBHEADING_RE.search(raw[:1000])
    if subheading and not (
        _COMBINED_SCOPE_RE.match(_ARTICLE_PREFIX_RE.sub("", _compact(raw[:260])))
        or _REGULATION_SCOPE_RE.match(_ARTICLE_PREFIX_RE.sub("", _compact(raw[:260])))
        or _SUBJECT_SCOPE_RE.match(_ARTICLE_PREFIX_RE.sub("", _compact(raw[:260])))
    ):
        start = subheading.start()
        subsequent = re.search(r"(?m)^[ \t]*\d+\.[ \t]+", raw[subheading.end():])
        end = subheading.end() + subsequent.start() if subsequent else min(len(raw), start + 1600)
        start, end = _trim_span(raw, start, end)
        return {"raw_text": raw[start:end], "relative_start": start, "relative_end": end}
    end = min(len(raw), 1800)
    start, end = _trim_span(raw, 0, end)
    return {"raw_text": raw[start:end], "relative_start": start, "relative_end": end}


def extract_relations(text: str, *, limit: int = 12) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for match in _RELATION_RE.finditer(text):
        key = (_compact(match.group("type")), match.group("number"), match.start())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "relation": _compact(match.group("relation") or "viện dẫn"),
                "document_type": key[0],
                "number": key[1],
                "raw_text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
        if len(result) >= limit:
            break
    return result


class NodeIndex:
    """Disk-backed node lookup; avoids loading 1.14M nodes into RAM."""

    SCHEMA_VERSION = 2

    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def build(cls, nodes_path: Path, output: Path) -> "NodeIndex":
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(str(temporary))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE nodes (node_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, "
                "parent_id TEXT, kind TEXT, label TEXT, heading_text TEXT, raw_text TEXT, "
                "start INTEGER, end INTEGER) WITHOUT ROWID"
            )
            connection.execute(f"PRAGMA user_version={cls.SCHEMA_VERSION}")
            batch: list[tuple[Any, ...]] = []
            for number, row in enumerate(read_jsonl(nodes_path), 1):
                batch.append(
                    (
                        str(row["node_id"]), str(row["doc_id"]),
                        None if not row.get("parent_id") else str(row["parent_id"]),
                        str(row.get("kind") or ""), str(row.get("label") or ""),
                        str(row.get("heading_text") or ""), str(row.get("raw_text") or "")[:2200],
                        row.get("start"), row.get("end"),
                    )
                )
                if len(batch) >= 4096:
                    connection.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
                if number % 100_000 == 0:
                    print(f"[node-index] {number:,} nodes", flush=True)
            if batch:
                connection.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)", batch)
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, output)
        return cls(output)

    @classmethod
    def is_current(cls, path: Path) -> bool:
        if not path.exists():
            return False
        connection = sqlite3.connect(str(path))
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            return version == cls.SCHEMA_VERSION
        finally:
            connection.close()

    def get(self, node_id: str | None) -> dict[str, Any] | None:
        if not node_id:
            return None
        row = self.connection.execute("SELECT * FROM nodes WHERE node_id=?", (str(node_id),)).fetchone()
        return dict(row) if row else None

    def ancestry(self, node_id: str | None, *, max_depth: int = 8) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        current = str(node_id) if node_id else ""
        while current and current not in seen and len(result) < max_depth:
            seen.add(current)
            row = self.get(current)
            if not row:
                break
            result.append(row)
            current = str(row.get("parent_id") or "")
        result.reverse()
        return result

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "NodeIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_metadata_sidecar(*, v3_dir: Path, preprocessing_dir: Path, output_dir: Path) -> dict[str, Any]:
    v3 = load_v3_manifest(v3_dir)
    require_success(v3_dir)
    documents = {str(row["doc_id"]): row for row in read_jsonl(v3_dir / "documents.jsonl")}
    node_index_path = output_dir / "node_index.sqlite"
    if not NodeIndex.is_current(node_index_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        index = NodeIndex.build(v3_dir / "nodes.jsonl", node_index_path)
        index.close()
    typed_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scope_counts: Counter[str] = Counter()
    with NodeIndex(node_index_path) as nodes:
        for document in documents.values():
            for node_id in document.get("scope_node_ids", []):
                node = nodes.get(str(node_id))
                if not node:
                    continue
                kind = classify_scope_node(node)
                if kind:
                    span = scope_display_span(node, kind)
                    entry = {
                        "node_id": str(node_id), "kind": kind,
                        "heading_text": str(node.get("heading_text") or ""),
                        "raw_text": span["raw_text"],
                        "start": (int(node["start"]) + span["relative_start"]) if node.get("start") is not None else None,
                        "end": (int(node["start"]) + span["relative_end"]) if node.get("start") is not None else None,
                    }
                    typed_scope[str(document["doc_id"])].append(entry)
                    scope_counts[kind] += 1
                else:
                    scope_counts["rejected_legacy_scope"] += 1

    rows: list[dict[str, Any]] = []
    title_counts: Counter[str] = Counter()
    relation_count = 0
    for number, doc_id in enumerate(sorted(documents, key=lambda value: (len(value), value)), 1):
        document = documents[doc_id]
        context_path = preprocessing_dir / "contexts" / f"context_{doc_id}.json"
        raw = _json(context_path)
        passage = str(raw.get("passage") or "")
        label = str(document.get("document_label") or "")
        title = extract_official_title(passage, label)
        if title.get("status") == "VERIFIED":
            if passage[int(title["start"]):int(title["end"])] != title["raw_text"]:
                raise AssertionError(f"official title offset mismatch for document {doc_id}")
        title_counts[str(title["status"])] += 1
        relations = extract_relations(passage[:6000])
        relation_count += len(relations)
        rows.append(
            {
                "schema_version": SCHEMA,
                "doc_id": doc_id,
                "normalized_label": label,
                "official_title": title,
                "typed_scope": typed_scope.get(doc_id, []),
                "relations": relations,
                "parse_mode": document.get("parse_mode"),
            }
        )
        if number % 1000 == 0:
            print(f"[metadata] {number:,}/{len(documents):,} documents", flush=True)
    write_jsonl(output_dir / "document_metadata.jsonl", rows)
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "documents": len(rows),
        "accented_labels": sum(has_vietnamese_accent(row["normalized_label"]) for row in rows),
        "title_status": dict(title_counts),
        "verified_title_exact_spans": title_counts["VERIFIED"],
        "scope_counts": dict(scope_counts),
        "documents_with_typed_scope": sum(bool(row["typed_scope"]) for row in rows),
        "header_relations": relation_count,
        "v3_fingerprint": v3["content_fingerprint"],
    }
    atomic_json(output_dir / "metadata_audit.json", report)
    atomic_json(
        output_dir / "manifest.json",
        artifact_manifest(
            stage="exp030-build-metadata",
            inputs={
                "v3_manifest_sha256": sha256_file(v3_dir / "manifest.json"),
                "preprocessing_manifest_sha256": sha256_file(preprocessing_dir / "manifest.json"),
            },
            config={"title_rule": "accented_header_exact_span_v1", "scope_rule": "heading_prefix_v1"},
            files=[output_dir / "document_metadata.jsonl", output_dir / "metadata_audit.json", node_index_path],
        ),
    )
    atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "status": "PASS"})
    return report


class ChunkIndex:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def build(cls, chunks_path: Path, output: Path) -> "ChunkIndex":
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(str(temporary))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, "
                "parent_node_id TEXT, start INTEGER, end INTEGER, token_count INTEGER) WITHOUT ROWID"
            )
            batch: list[tuple[Any, ...]] = []
            for row in read_jsonl(chunks_path):
                batch.append(
                    (
                        str(row["chunk_id"]), str(row["doc_id"]),
                        None if not row.get("parent_node_id") else str(row["parent_node_id"]),
                        row.get("start"), row.get("end"), row.get("token_count"),
                    )
                )
                if len(batch) >= 4096:
                    connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", batch)
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, output)
        return cls(output)

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM chunks WHERE chunk_id=?", (str(chunk_id),)).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ChunkIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class CandidateSourceIndex:
    """Disk-backed lookup for the 1.05M frozen retrieval provenance rows."""

    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def build(
        cls, candidates_path: Path, output: Path, *, qids: set[str] | None = None,
    ) -> "CandidateSourceIndex":
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(str(temporary))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE provenance (qid TEXT NOT NULL, doc_id TEXT NOT NULL, sources_json TEXT NOT NULL, "
                "PRIMARY KEY (qid, doc_id)) WITHOUT ROWID"
            )
            batch: list[tuple[str, str, str]] = []
            for query_number, row in enumerate(read_jsonl(candidates_path), 1):
                qid = str(row["qid"])
                if qids is not None and qid not in qids:
                    continue
                for candidate in row.get("candidates", []):
                    batch.append((qid, str(candidate["doc_id"]), canonical_json(candidate.get("sources") or {})))
                    if len(batch) >= 4096:
                        connection.executemany("INSERT INTO provenance VALUES (?,?,?)", batch)
                        batch.clear()
                if query_number % 500 == 0:
                    print(f"[candidate-source-index] {query_number:,} queries", flush=True)
            if batch:
                connection.executemany("INSERT INTO provenance VALUES (?,?,?)", batch)
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, output)
        return cls(output)

    def for_query(self, qid: str) -> dict[str, dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT doc_id, sources_json FROM provenance WHERE qid=?", (str(qid),)
        ).fetchall()
        return {str(row["doc_id"]): json.loads(row["sources_json"]) for row in rows}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CandidateSourceIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def write_indexed_jsonl(path: Path, records: Iterable[dict[str, Any]], index_path: Path) -> int:
    """Atomically write JSONL plus a qid -> byte-range index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    json_fd, json_temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(json_fd)
    json_temporary = Path(json_temporary_name)
    index_temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    index_temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(str(index_temporary))
    count = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE offsets (qid TEXT PRIMARY KEY, byte_offset INTEGER NOT NULL, "
            "byte_length INTEGER NOT NULL, ordinal INTEGER NOT NULL) WITHOUT ROWID"
        )
        with json_temporary.open("wb") as handle:
            batch: list[tuple[str, int, int, int]] = []
            for ordinal, record in enumerate(records):
                qid = str(record["qid"])
                payload = (canonical_json(record) + "\n").encode("utf-8")
                offset = handle.tell()
                handle.write(payload)
                batch.append((qid, offset, len(payload), ordinal))
                count += 1
                if len(batch) >= 512:
                    connection.executemany("INSERT INTO offsets VALUES (?,?,?,?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO offsets VALUES (?,?,?,?)", batch)
            handle.flush()
            os.fsync(handle.fileno())
        connection.commit()
        connection.close()
        os.replace(json_temporary, path)
        os.replace(index_temporary, index_path)
        return count
    except Exception:
        connection.close()
        json_temporary.unlink(missing_ok=True)
        index_temporary.unlink(missing_ok=True)
        raise


def read_selected_jsonl(
    path: Path, qids: set[str] | Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Seek selected qids when an offset index exists; otherwise stream safely."""
    if qids is None:
        yield from read_jsonl(path)
        return
    requested = {str(qid) for qid in qids}
    requested_order = {str(qid): index for index, qid in enumerate(qids)} if not isinstance(qids, set) else None
    index_path = path.with_suffix(".index.sqlite")
    if not index_path.exists():
        for row in read_jsonl(path):
            if str(row["qid"]) in requested:
                yield row
        return
    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in requested)
        if not placeholders:
            return
        rows = connection.execute(
            f"SELECT qid, byte_offset, byte_length FROM offsets WHERE qid IN ({placeholders}) ORDER BY ordinal",
            tuple(sorted(requested)),
        ).fetchall()
        found = {str(row["qid"]) for row in rows}
        if found != requested:
            raise ValueError(f"capsule index missing qids: {sorted(requested - found)[:10]}")
        if requested_order is not None:
            rows.sort(key=lambda row: requested_order[str(row["qid"])])
        with path.open("rb") as handle:
            for row in rows:
                handle.seek(int(row["byte_offset"]))
                payload = handle.read(int(row["byte_length"]))
                yield json.loads(payload.decode("utf-8"))
    finally:
        connection.close()


_KIND_VI = {
    "document": "Văn bản",
    "chapter": "Chương",
    "section": "Mục",
    "article": "Điều",
    "clause": "Khoản",
    "point": "Điểm",
    "region": "Đoạn",
    "annex": "Phụ lục",
}


def format_structural_path(ancestry: Sequence[Mapping[str, Any]]) -> str:
    pieces: list[str] = []
    for node in ancestry:
        kind = str(node.get("kind") or "")
        if kind in {"document", "region", "synthetic"}:
            continue
        heading = _compact(str(node.get("heading_text") or ""))
        label = _compact(str(node.get("label") or ""))
        value = heading or _compact(f"{_KIND_VI.get(kind, kind)} {label}")
        if value and value not in pieces:
            pieces.append(value)
    return " > ".join(pieces)


def _metadata_by_doc(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["doc_id"]): row for row in read_jsonl(path)}


def enrich_capsules(
    *, capsules: Path, metadata_dir: Path, v3_dir: Path, output_dir: Path,
    frozen_candidates: Path | None = None, max_queries: int | None = None,
) -> dict[str, Any]:
    """Bind K=64 capsules to verified metadata and actual evidence ancestry."""
    require_success(metadata_dir)
    metadata = _metadata_by_doc(metadata_dir / "document_metadata.jsonl")
    node_index_path = metadata_dir / "node_index.sqlite"
    chunk_index_path = metadata_dir / "chunk_index.sqlite"
    source_index_path = output_dir / "candidate_sources.sqlite"
    capsule_index_path = output_dir / "capsules.index.sqlite"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not chunk_index_path.exists():
        index = ChunkIndex.build(v3_dir / "chunks.jsonl", chunk_index_path)
        index.close()
    limited_qids: set[str] | None = None
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        limited_qids = set()
        for row in read_jsonl(capsules):
            limited_qids.add(str(row["qid"]))
            if len(limited_qids) >= max_queries:
                break
    # This index may be query-limited for smoke tests, so it is rebuilt for
    # each materialization instead of risking reuse with a different scope.
    if frozen_candidates:
        index = CandidateSourceIndex.build(frozen_candidates, source_index_path, qids=limited_qids)
        index.close()
    counters: Counter[str] = Counter()
    source_context = CandidateSourceIndex(source_index_path) if frozen_candidates else None
    with NodeIndex(node_index_path) as nodes, ChunkIndex(chunk_index_path) as chunks:
        def enriched_rows() -> Iterator[dict[str, Any]]:
            for row in read_jsonl(capsules):
                if max_queries is not None and counters["queries"] >= max_queries:
                    break
                counters["queries"] += 1
                counters["pairs"] += len(row.get("candidates", []))
                if counters["queries"] % 100 == 0:
                    print(
                        f"[capsules] {counters['queries']:,} queries / {counters['pairs']:,} pairs",
                        flush=True,
                    )
                candidates: list[dict[str, Any]] = []
                if len(row.get("candidates", [])) != K:
                    raise ValueError(f"EXP-030 requires exactly K=64 candidates: {row.get('qid')}")
                frozen_docs = source_context.for_query(str(row["qid"])) if source_context else {}
                for candidate in row["candidates"]:
                    doc_id = str(candidate["doc_id"])
                    if doc_id not in metadata:
                        raise ValueError(f"missing metadata for document {doc_id}")
                    meta = metadata[doc_id]
                    evidence: list[dict[str, Any]] = []
                    for item in candidate.get("evidence", []):
                        chunk = chunks.get(str(item["chunk_id"]))
                        if not chunk or str(chunk["doc_id"]) != doc_id:
                            raise ValueError(f"chunk/document mismatch: {row['qid']}/{doc_id}/{item['chunk_id']}")
                        parent = str(chunk.get("parent_node_id") or item.get("parent_node_id") or "")
                        ancestry = nodes.ancestry(parent)
                        if parent and not ancestry:
                            raise ValueError(f"unresolved evidence parent node: {parent}")
                        path = format_structural_path(ancestry)
                        counters["evidence"] += 1
                        counters["evidence_with_parent"] += bool(parent)
                        counters["evidence_with_path"] += bool(path)
                        evidence.append(
                            {
                                **item,
                                "parent_node_id": parent or None,
                                "structural_path": path,
                                "ancestry_node_ids": [str(node["node_id"]) for node in ancestry],
                                "relations": extract_relations(str(item.get("raw_text") or "")),
                            }
                        )
                    sources = frozen_docs.get(doc_id, {})
                    source_names = sorted(str(key) for key, value in sources.items() if value)
                    counters["official_title"] += meta["official_title"].get("status") == "VERIFIED"
                    counters["typed_scope"] += bool(meta["typed_scope"])
                    candidates.append(
                        {
                            **candidate,
                            "document_label": meta["normalized_label"],
                            "official_title": meta["official_title"],
                            "typed_scope": meta["typed_scope"],
                            "header_relations": meta["relations"],
                            "retrieval_sources": source_names,
                            "retrieval_provenance": sources,
                            "evidence": evidence,
                        }
                    )
                if [str(x["doc_id"]) for x in candidates] != [str(x["doc_id"]) for x in row["candidates"]]:
                    raise AssertionError("capsule enrichment changed candidate membership/order")
                yield {**row, "schema_version": SCHEMA, "candidates": candidates}

        try:
            write_indexed_jsonl(output_dir / "capsules.jsonl", enriched_rows(), capsule_index_path)
        finally:
            if source_context:
                source_context.close()
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "queries": counters["queries"],
        "pairs": counters["pairs"],
        "candidate_membership_preserved": True,
        "evidence": counters["evidence"],
        "evidence_with_parent": counters["evidence_with_parent"],
        "evidence_with_structural_path": counters["evidence_with_path"],
        "documents_with_official_title": counters["official_title"],
        "documents_with_typed_scope": counters["typed_scope"],
    }
    atomic_json(output_dir / "capsule_audit.json", report)
    atomic_json(
        output_dir / "manifest.json",
        artifact_manifest(
            stage="exp030-enrich-capsules",
            inputs={
                "capsules_sha256": sha256_file(capsules),
                "metadata_manifest_sha256": sha256_file(metadata_dir / "manifest.json"),
                "v3_manifest_sha256": sha256_file(v3_dir / "manifest.json"),
                **({"frozen_candidates_sha256": sha256_file(frozen_candidates)} if frozen_candidates else {}),
            },
            config={
                "k": K, "markers": list(VIETNAMESE_MARKERS),
                "frozen_candidates": bool(frozen_candidates), "max_queries": max_queries,
            },
            files=[output_dir / "capsules.jsonl", capsule_index_path, output_dir / "capsule_audit.json"]
            + ([source_index_path] if frozen_candidates else []),
        ),
    )
    atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "status": "PASS"})
    return report


def query_signals(query: str) -> dict[str, bool]:
    folded = accent_fold(query)
    return {
        "explicit_subject": bool(re.search(r"\b(ai|nguoi nao|doi tuong nao|to chuc nao|ca nhan nao)\b", folded)),
        "explicit_scope": bool(re.search(r"\b(pham vi|truong hop nao|ap dung doi voi|linh vuc nao)\b", folded)),
        "explicit_article": bool(re.search(r"\b(dieu|khoan|diem)\s+\d+", folded)),
        "explicit_instrument": bool(_NUMBER_RE.search(query)),
    }


def lexical_view_relevance(query: str, text: str) -> float:
    query_tokens = normalized_tokens(query)
    text_tokens = normalized_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / math.sqrt(len(query_tokens) * len(text_tokens))


def _identity_lines(candidate: Mapping[str, Any], variant: str) -> list[str]:
    lines = ["[VĂN BẢN]"]
    title = candidate.get("official_title") or {}
    if variant in {"title", "both"} and title.get("status") == "VERIFIED":
        lines.append(f"[TÊN CHÍNH THỨC] {title['display_text']}")
    if variant in {"unaccented", "both"}:
        lines.append(f"[TÊN CHUẨN HÓA] {candidate.get('document_label', '')}")
    return lines


def _scope_text(candidate: Mapping[str, Any], kind: str) -> str:
    values = []
    for row in candidate.get("typed_scope", []):
        if row.get("kind") in {kind, "combined"}:
            values.append(_compact(str(row.get("raw_text") or "")))
    return "\n".join(dict.fromkeys(values))


def _relation_text(candidate: Mapping[str, Any]) -> str:
    values: list[str] = []
    for row in list(candidate.get("header_relations", [])) + [
        relation for evidence in candidate.get("evidence", []) for relation in evidence.get("relations", [])
    ]:
        raw = _compact(str(row.get("raw_text") or ""))
        if raw and raw not in values:
            values.append(raw)
    return "; ".join(values[:12])


def build_candidate_views(
    query: str, candidate: Mapping[str, Any], *, identity_variant: str = "both",
    include_structure: bool = True, include_scope: bool = True, include_relations: bool = True,
) -> list[dict[str, Any]]:
    if identity_variant not in {"none", "unaccented", "title", "both"}:
        raise ValueError(f"unknown identity variant: {identity_variant}")
    identity = _identity_lines(candidate, identity_variant)
    evidence = list(candidate.get("evidence", []))
    primary = _compact(str(evidence[0].get("raw_text") or "")) if evidence else ""
    secondary = _compact(str(evidence[1].get("raw_text") or "")) if len(evidence) > 1 else ""
    paths = [str(item.get("structural_path") or "") for item in evidence if item.get("structural_path")]
    base_lines = list(identity)
    if include_structure and paths:
        base_lines.append("[VỊ TRÍ TRONG VĂN BẢN] " + " | ".join(dict.fromkeys(paths)))
    base_lines.append("[BẰNG CHỨNG TRẢ LỜI] " + primary)
    if secondary:
        base_lines.append(secondary)
    views = [{"kind": "base", "text": "\n".join(base_lines), "routing_score": 1.0}]

    scope = _scope_text(candidate, "scope_of_regulation")
    subjects = _scope_text(candidate, "applicable_subjects")
    if include_scope and (scope or subjects):
        lines = list(identity)
        if scope:
            lines.append("[PHẠM VI ĐIỀU CHỈNH] " + scope)
        if subjects:
            lines.append("[ĐỐI TƯỢNG ÁP DỤNG] " + subjects)
        if include_structure and paths:
            lines.append("[VỊ TRÍ TRONG VĂN BẢN] " + paths[0])
        lines.append("[BẰNG CHỨNG TRẢ LỜI] " + primary)
        text = "\n".join(lines)
        views.append({"kind": "applicability", "text": text, "routing_score": lexical_view_relevance(query, scope + " " + subjects)})

    relations = _relation_text(candidate)
    if include_relations and relations:
        lines = list(identity)
        lines.append("[QUAN HỆ PHÁP LÝ] " + relations)
        if include_structure and paths:
            lines.append("[VỊ TRÍ TRONG VĂN BẢN] " + paths[0])
        lines.append("[BẰNG CHỨNG TRẢ LỜI] " + primary)
        text = "\n".join(lines)
        views.append({"kind": "relations", "text": text, "routing_score": lexical_view_relevance(query, relations)})
    return views


def select_views(query: str, views: Sequence[Mapping[str, Any]], *, uncertain_margin: float = 0.05) -> list[dict[str, Any]]:
    """Always retain base; low-confidence routing retains both best auxiliaries."""
    if not views:
        raise ValueError("candidate has no views")
    base = dict(next((view for view in views if view.get("kind") == "base"), views[0]))
    auxiliaries = sorted(
        (dict(view) for view in views if view.get("kind") != "base"),
        key=lambda row: (-float(row.get("routing_score", 0.0)), str(row.get("kind"))),
    )
    if not auxiliaries:
        return [base]
    signals = query_signals(query)
    if signals["explicit_subject"] or signals["explicit_scope"]:
        applicability = next((row for row in auxiliaries if row.get("kind") == "applicability"), None)
        if applicability:
            return [base, applicability]
    if len(auxiliaries) == 1:
        return [base, auxiliaries[0]]
    margin = float(auxiliaries[0].get("routing_score", 0.0)) - float(auxiliaries[1].get("routing_score", 0.0))
    return [base, auxiliaries[0], auxiliaries[1]] if margin < uncertain_margin else [base, auxiliaries[0]]


def truncate_pair_document(tokenizer: Any, query: str, document: str, max_length: int) -> tuple[str, int]:
    """Truncate only the document and verify the *actual pair* token count."""
    def length(text: str) -> int:
        encoded = tokenizer(query, text, add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"]
        return int(ids.shape[-1] if hasattr(ids, "shape") else len(ids))

    full_length = length(document)
    if full_length <= max_length:
        return document, full_length
    document_ids = tokenizer(
        document, add_special_tokens=False, truncation=False, verbose=False,
    )["input_ids"]
    low, high, best = 0, len(document_ids), ""
    best_length = length("")
    while low <= high:
        middle = (low + high) // 2
        text = tokenizer.decode(document_ids[:middle], skip_special_tokens=True)
        current = length(text)
        if current <= max_length:
            best, best_length, low = text, current, middle + 1
        else:
            high = middle - 1
    if best_length > max_length:
        raise ValueError("query and special tokens alone exceed pair budget")
    return best, best_length


def aggregate_view_scores(scores: Sequence[float], *, method: str = "max", temperature: float = 1.0) -> float:
    if not scores:
        raise ValueError("cannot aggregate empty view scores")
    if method == "max":
        return float(max(scores))
    if method == "mean":
        return float(np.mean(scores))
    if method == "logsumexp":
        values = np.asarray(scores, dtype=np.float64) / temperature
        peak = float(values.max())
        return float(temperature * (peak + math.log(float(np.exp(values - peak).mean()))))
    raise ValueError(f"unknown aggregation method: {method}")


QWEN_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)
QWEN_INSTRUCTION = (
    "Given a Vietnamese legal-information query, retrieve authoritative legal documents that answer the query"
)
QWEN_PREFIX = f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n<|im_start|>user\n"
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def qwen_format_instruction(query: str, document: str) -> str:
    return f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"


def _torch_dtype(precision: str) -> Any:
    import torch
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unsupported floating-point precision: {precision}")


def configure_hf_modules_cache() -> Path:
    """Bind Transformers remote-code modules to a writable project cache."""
    _HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(_HF_MODULES_CACHE)
    # Guard against a transitive import that happened before this function.
    import transformers.dynamic_module_utils as dynamic_module_utils
    dynamic_module_utils.HF_MODULES_CACHE = str(_HF_MODULES_CACHE)
    return _HF_MODULES_CACHE


def _load_model(
    spec: ModelSpec, *, device: str, local_only: bool = False, precision: str = "fp32",
) -> tuple[Any, Any]:
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    configure_hf_modules_cache()
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    if spec.kind == "jina":
        model = AutoModel.from_pretrained(
            spec.model_id, trust_remote_code=True, local_files_only=local_only, dtype=_torch_dtype(precision),
        ).to(device).eval()
        model._ensure_tokenizer()
        # The native implementation still receives all K documents in one
        # rerank call, but uses this capacity to split them into internal
        # blocks and aggregate query/document embeddings.  The published 131k
        # packing window is not feasible on a 6 GB GPU for 64 x 512-token docs.
        model._tokenizer.model_max_length = JINA_RUNTIME_CONTEXT
        return model, model._tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=local_only,
        padding_side="left" if spec.kind == "qwen" else "right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    factory = AutoModelForCausalLM if spec.kind == "qwen" else AutoModelForSequenceClassification
    model = factory.from_pretrained(
        spec.model_id,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=local_only,
        dtype=_torch_dtype(precision),
    )
    # The cached GTE remote implementation expects this deterministic buffer,
    # but some Transformers versions do not restore non-persistent buffers.
    if spec.key == "gte" and hasattr(getattr(model, "new", None), "embeddings"):
        import torch
        embeddings = model.new.embeddings
        embeddings.position_ids = torch.arange(int(model.config.max_position_embeddings), dtype=torch.long)
    return model.to(device).eval(), tokenizer


def _pair_logits(
    spec: ModelSpec, model: Any, tokenizer: Any, query: str, documents: Sequence[str],
    *, device: str, max_length: int,
) -> list[float]:
    import torch
    batch = tokenizer(
        [query] * len(documents), list(documents), padding=True, truncation="only_second",
        max_length=max_length, return_tensors="pt",
    )
    if spec.key == "gte":
        batch.pop("token_type_ids", None)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        return model(**batch, return_dict=True).logits.reshape(-1).float().cpu().tolist()


def _qwen_logits(
    model: Any, tokenizer: Any, query: str, documents: Sequence[str],
    *, device: str, max_length: int,
) -> list[float]:
    import torch
    prefix_tokens = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
    budget = max_length - len(prefix_tokens) - len(suffix_tokens)
    if budget <= 0:
        raise ValueError("Qwen prefix/suffix exceed max length")
    pairs = [qwen_format_instruction(query, document) for document in documents]
    encoded = tokenizer(
        pairs, padding=False, truncation="longest_first", max_length=budget,
        return_attention_mask=False,
    )
    rows = [prefix_tokens + ids + suffix_tokens for ids in encoded["input_ids"]]
    # Rows are already explicitly budgeted above; max_length is ignored by
    # Transformers when padding=True and emits a misleading runtime warning.
    inputs = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if yes_id is None or no_id is None or yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
        raise ValueError("Qwen yes/no token contract is not satisfied")
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, [no_id, yes_id]]
        return torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist()


def score_query_record(
    *, spec: ModelSpec, model: Any, tokenizer: Any, row: Mapping[str, Any], device: str,
    identity_variant: str = "both", aggregation: str = "max", max_length: int = COMMON_MAX_LENGTH,
    batch_size: int = 8, capsule_variant: str | None = None,
) -> dict[str, Any]:
    query = str(row["query"])
    config = CAPSULE_CONFIGS[capsule_variant] if capsule_variant else {
        "identity_variant": identity_variant, "structure": True, "scope": True, "relations": True, "view_policy": "routed",
    }
    selected_by_doc: list[list[dict[str, Any]]] = []
    for candidate in row["candidates"]:
        views = build_candidate_views(
            query, candidate, identity_variant=str(config["identity_variant"]),
            include_structure=bool(config["structure"]), include_scope=bool(config["scope"]),
            include_relations=bool(config["relations"]),
        )
        selected_by_doc.append(views if config["view_policy"] == "all" else (select_views(query, views) if config["view_policy"] == "routed" else [views[0]]))

    if spec.kind == "jina":
        # Preserve the native K-way interface.  Each view position is one K-way
        # list; documents without that auxiliary view fall back to their base.
        maximum_views = max(len(views) for views in selected_by_doc)
        per_doc: list[list[float]] = [[] for _ in selected_by_doc]
        for view_index in range(maximum_views):
            documents = [views[min(view_index, len(views) - 1)]["text"] for views in selected_by_doc]
            if tokenizer is not None:
                documents = [
                    truncate_pair_document(tokenizer, query, document, max_length)[0]
                    for document in documents
                ]
            results = model.rerank(query, documents)
            if len(results) != len(documents):
                raise ValueError("Jina native rerank did not return every document")
            seen: set[int] = set()
            for result in results:
                index = int(result["index"] if isinstance(result, dict) else result.index)
                score = float(result["relevance_score"] if isinstance(result, dict) else result.relevance_score)
                if index in seen or index < 0 or index >= len(documents):
                    raise ValueError("invalid Jina listwise index mapping")
                seen.add(index)
                per_doc[index].append(score)
        document_scores = [aggregate_view_scores(values, method=aggregation) for values in per_doc]
        view_payload = per_doc
    else:
        flat_documents: list[str] = []
        ownership: list[int] = []
        token_lengths: list[int] = []
        for document_index, views in enumerate(selected_by_doc):
            for view in views:
                if spec.kind == "qwen":
                    document = str(view["text"])
                    token_length = len(tokenizer.encode(document, add_special_tokens=False))
                else:
                    document, token_length = truncate_pair_document(tokenizer, query, str(view["text"]), max_length)
                flat_documents.append(document)
                ownership.append(document_index)
                token_lengths.append(token_length)
        flat_scores: list[float] = []
        for start in range(0, len(flat_documents), batch_size):
            batch = flat_documents[start:start + batch_size]
            if spec.kind == "qwen":
                flat_scores.extend(_qwen_logits(model, tokenizer, query, batch, device=device, max_length=max_length))
            else:
                flat_scores.extend(_pair_logits(spec, model, tokenizer, query, batch, device=device, max_length=max_length))
        grouped: list[list[float]] = [[] for _ in selected_by_doc]
        for owner, score in zip(ownership, flat_scores):
            grouped[owner].append(float(score))
        document_scores = [aggregate_view_scores(values, method=aggregation) for values in grouped]
        view_payload = grouped
    return {
        "schema_version": SCHEMA,
        "qid": str(row["qid"]),
        "model": spec.key,
        "identity_variant": str(config["identity_variant"]),
        "capsule_variant": capsule_variant,
        "aggregation": aggregation,
        "scores": [
            {
                "doc_id": str(candidate["doc_id"]),
                "score": float(score),
                "view_scores": [float(value) for value in views],
            }
            for candidate, score, views in zip(row["candidates"], document_scores, view_payload)
        ],
    }


def score_capsules(
    *, spec: ModelSpec, capsules: Path, output_dir: Path, qids: set[str], device: str,
    local_only: bool, identity_variant: str, aggregation: str, max_length: int,
    resume: bool = False, capsule_variant: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.jsonl"
    existing = {str(row["qid"]): row for row in read_jsonl(score_path)} if resume and score_path.exists() else {}
    model, tokenizer = _load_model(spec, device=device, local_only=local_only)
    started = time.perf_counter()
    peak_vram = 0
    try:
        for number, row in enumerate(read_selected_jsonl(capsules, qids), 1):
            qid = str(row["qid"])
            if qid in existing:
                continue
            existing[qid] = score_query_record(
                spec=spec, model=model, tokenizer=tokenizer, row=row, device=device,
                identity_variant=identity_variant, aggregation=aggregation, max_length=max_length,
                capsule_variant=capsule_variant,
            )
            if number % 16 == 0:
                write_jsonl(score_path, (existing[key] for key in sorted(existing)))
            try:
                import torch
                if device.startswith("cuda"):
                    peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            except Exception:
                pass
        write_jsonl(score_path, (existing[key] for key in sorted(existing)))
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    report = {
        "schema_version": SCHEMA,
        "model": spec.key,
        "queries": len(existing),
        "seconds": time.perf_counter() - started,
        "peak_vram_bytes": peak_vram,
        "scores": str(score_path.resolve()),
    }
    atomic_json(output_dir / "score_report.json", report)
    return report


def _gold_rank_bucket(row: Mapping[str, Any], gold: set[str]) -> str:
    ranks = [index + 1 for index, candidate in enumerate(row["candidates"]) if str(candidate["doc_id"]) in gold]
    rank = min(ranks) if ranks else K + 1
    if rank <= 5:
        return "01-05"
    if rank <= 16:
        return "06-16"
    if rank <= 32:
        return "17-32"
    if rank <= 64:
        return "33-64"
    return "missing"


def stratified_sample(
    rows: Iterable[Mapping[str, Any]], answers: Mapping[str, set[str]], *, limit: int, seed: int = SEED,
) -> set[str]:
    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows:
        qid = str(row["qid"])
        gold = answers[qid]
        if not gold:
            continue
        query_length = len(normalized_tokens(str(row["query"])))
        length_bucket = "short" if query_length <= 8 else ("medium" if query_length <= 16 else "long")
        gold_scope = any(
            str(candidate["doc_id"]) in gold and bool(candidate.get("typed_scope"))
            for candidate in row["candidates"]
        )
        strata[(min(len(gold), 3), _gold_rank_bucket(row, gold), length_bucket, gold_scope)].append(qid)
    for key, qids in strata.items():
        qids.sort(key=lambda qid: _hash([seed, key, qid]))
    selected: list[str] = []
    ordered_keys = sorted(strata, key=str)
    cursor = 0
    while len(selected) < limit and ordered_keys:
        key = ordered_keys[cursor % len(ordered_keys)]
        if strata[key]:
            selected.append(strata[key].pop(0))
        else:
            ordered_keys.remove(key)
            cursor -= 1
        cursor += 1
    return set(selected)


def select_hard_negatives(
    row: Mapping[str, Any], answers: set[str], *, epoch: int, seed: int = SEED,
) -> dict[str, Any] | None:
    positives = [candidate for candidate in row["candidates"] if str(candidate["doc_id"]) in answers]
    negatives = [candidate for candidate in row["candidates"] if str(candidate["doc_id"]) not in answers]
    if not positives or len(negatives) < 8:
        return None
    positive_tokens = set().union(*(normalized_tokens(str(candidate.get("document_label") or "")) for candidate in positives))
    positive_scope = set().union(*(
        normalized_tokens(" ".join(str(scope.get("raw_text") or "") for scope in candidate.get("typed_scope", [])))
        for candidate in positives
    ))

    buckets: list[tuple[str, list[Mapping[str, Any]], int]] = []
    buckets.append(("top_rank", negatives[:8], 2))
    e5_only = [candidate for candidate in negatives if candidate.get("retrieval_sources") == ["e5"]]
    bm25_only = [candidate for candidate in negatives if candidate.get("retrieval_sources") == ["bm25"]]
    buckets.extend((("e5_only", e5_only, 1), ("bm25_only", bm25_only, 1)))
    family = sorted(
        negatives,
        key=lambda candidate: (
            -len(positive_tokens & normalized_tokens(str(candidate.get("document_label") or ""))),
            str(candidate["doc_id"]),
        ),
    )
    buckets.append(("legal_family", family, 2))
    scope_confusers = sorted(
        negatives,
        key=lambda candidate: (
            -len(positive_scope & normalized_tokens(" ".join(str(scope.get("raw_text") or "") for scope in candidate.get("typed_scope", [])))),
            str(candidate["doc_id"]),
        ),
    )
    buckets.append(("scope_actor", scope_confusers, 1))
    tail = negatives[32:] or negatives
    tail = sorted(tail, key=lambda candidate: _hash([seed, epoch, row["qid"], candidate["doc_id"]]))
    buckets.append(("seeded_tail", tail, 1))

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for bucket_name, candidates, count in buckets:
        rotated = list(candidates)
        if rotated:
            shift = epoch % len(rotated)
            rotated = rotated[shift:] + rotated[:shift]
        for candidate in rotated:
            doc_id = str(candidate["doc_id"])
            if doc_id in used:
                continue
            chosen.append({"bucket": bucket_name, "candidate": candidate})
            used.add(doc_id)
            if sum(item["bucket"] == bucket_name for item in chosen) >= count:
                break
    for candidate in negatives:
        if len(chosen) >= 8:
            break
        doc_id = str(candidate["doc_id"])
        if doc_id not in used:
            chosen.append({"bucket": "rank_backfill", "candidate": candidate})
            used.add(doc_id)
    if len(chosen) != 8:
        raise ValueError(f"could not build eight unique negatives for {row['qid']}")
    return {"qid": str(row["qid"]), "query": row["query"], "positives": positives, "negatives": chosen}


def discover_lora_targets(spec: ModelSpec, model: Any) -> dict[str, Any]:
    names = [name for name, _ in model.named_modules()]
    if spec.lora_family in {"xlmr", "minilm"}:
        targets = ("query", "key", "value")
        matched = [name for name in names if ".attention." in name and name.rsplit(".", 1)[-1] in targets]
    elif spec.lora_family == "qwen":
        targets = ("q_proj", "k_proj", "v_proj", "o_proj")
        matched = [name for name in names if name.rsplit(".", 1)[-1] in targets and ".self_attn." in name]
    elif spec.lora_family == "discover":
        candidates = (
            (("qkv_proj", "o_proj"), ".attention."),
            (("q_proj", "k_proj", "v_proj", "o_proj"), ".attention."),
            (("query", "key", "value"), ".attention."),
            (("q_proj", "k_proj", "v_proj", "o_proj"), ""),
        )
        targets, matched = (), []
        for suffixes, required in candidates:
            rows = [name for name in names if name.rsplit(".", 1)[-1] in suffixes and required in name]
            present = {name.rsplit(".", 1)[-1] for name in rows}
            if set(suffixes).issubset(present):
                targets, matched = suffixes, rows
                break
    else:
        return {"eligible": False, "targets": [], "matched_modules": [], "reason": "no_published_train_contract"}
    missing = [target for target in targets if not any(name.endswith("." + target) for name in matched)]
    return {
        "eligible": bool(targets) and not missing,
        "targets": list(targets),
        "matched_modules": matched,
        "matched_count": len(matched),
        "missing": missing,
    }


def _differentiable_qwen_scores(
    model: Any, tokenizer: Any, query: str, documents: Sequence[str], *, device: str, max_length: int,
) -> Any:
    import torch
    prefix_tokens = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
    budget = max_length - len(prefix_tokens) - len(suffix_tokens)
    encoded = tokenizer(
        [qwen_format_instruction(query, document) for document in documents],
        padding=False, truncation="longest_first", max_length=budget, return_attention_mask=False,
    )
    rows = [prefix_tokens + ids + suffix_tokens for ids in encoded["input_ids"]]
    inputs = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    logits = model(**inputs).logits[:, -1, [no_id, yes_id]]
    return torch.log_softmax(logits, dim=-1)[:, 1]


def lora_backward_fixture(
    *, spec: ModelSpec, model: Any, tokenizer: Any, device: str, output_dir: Path,
) -> dict[str, Any]:
    if spec.kind == "jina":
        return {"state": "ZERO_SHOT_ONLY", "reason": "native_rerank_has_no_published_training_interface"}
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    contract = discover_lora_targets(spec, model)
    if not contract["eligible"]:
        return {"state": "ZERO_SHOT_ONLY", "target_contract": contract}
    task_type = "CAUSAL_LM" if spec.kind == "qwen" else "SEQ_CLS"
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, bias="none", task_type=task_type,
            target_modules=contract["targets"],
        ),
    )
    peft_model.train()
    query = "Đối tượng nào phải thực hiện quy định này?"
    documents = ["Quy định áp dụng cho doanh nghiệp vận tải.", "Thời tiết hôm nay có nắng."]
    if spec.kind == "qwen":
        scores = _differentiable_qwen_scores(peft_model, tokenizer, query, documents, device=device, max_length=256)
    else:
        batch = tokenizer([query, query], documents, padding=True, truncation="only_second", max_length=256, return_tensors="pt")
        if spec.key == "gte":
            batch.pop("token_type_ids", None)
        batch = {key: value.to(device) for key, value in batch.items()}
        scores = peft_model(**batch).logits.reshape(-1)
    loss = -torch.log_softmax(scores, dim=0)[0]
    loss.backward()
    gradients = [parameter.grad for parameter in peft_model.parameters() if parameter.requires_grad]
    finite = bool(gradients) and all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients)
    if not finite:
        raise RuntimeError(f"non-finite or missing LoRA gradients for {spec.key}")
    adapter_dir = output_dir / "adapter_fixture"
    peft_model.save_pretrained(adapter_dir)
    # Reload into the same base object to validate adapter serialization.  The
    # full worker reloads a fresh base in its own process.
    trainable = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    base_model = peft_model.unload()
    reloaded = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    del reloaded
    return {
        "state": "ELIGIBLE",
        "target_contract": contract,
        "loss": float(loss.detach().cpu()),
        "trainable_parameters": int(trainable),
        "checkpoint_reload": True,
    }


def preflight_models(*, output_dir: Path, device: str, local_only: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import bitsandbytes  # noqa: F401
        bnb_available = True
    except Exception:
        bnb_available = False
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "device": device,
        "bitsandbytes": bnb_available,
        "models": {},
    }
    for spec in MODELS:
        print(f"[preflight] START {spec.key} ({spec.model_id})", flush=True)
        model_dir = output_dir / spec.key
        model_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model = None
        try:
            try:
                import torch
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            model, tokenizer = _load_model(spec, device=device, local_only=local_only)
            total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
            smoke: dict[str, Any]
            if spec.kind == "jina":
                ramps = {}
                for size in (2, 8, 16, 32, 64):
                    # Use realistic near-budget documents: short strings would
                    # miss the K64 attention-memory failure this gate exists to catch.
                    documents = [
                        (f"văn bản pháp luật {index} " + "quy định áp dụng cho tổ chức cá nhân " * 90)
                        for index in range(size)
                    ]
                    documents = [
                        truncate_pair_document(tokenizer, "quy định áp dụng cho ai", document, COMMON_MAX_LENGTH)[0]
                        for document in documents
                    ]
                    values = model.rerank("quy định áp dụng cho ai", documents)
                    if len(values) != size or {int(value["index"]) for value in values} != set(range(size)):
                        raise ValueError(f"Jina invalid listwise mapping at K={size}")
                    ramps[str(size)] = "PASS"
                smoke = {
                    "listwise_ramp": ramps,
                    "pair_budget": COMMON_MAX_LENGTH,
                    "runtime_packing_context": JINA_RUNTIME_CONTEXT,
                }
                backward = {"state": "ZERO_SHOT_ONLY", "reason": "no_published_native_backward_contract"}
            else:
                docs = ["Quy định áp dụng cho tổ chức, cá nhân.", "Nội dung không liên quan."]
                values = (
                    _qwen_logits(model, tokenizer, "Đối tượng áp dụng là ai?", docs, device=device, max_length=256)
                    if spec.kind == "qwen" else
                    _pair_logits(spec, model, tokenizer, "Đối tượng áp dụng là ai?", docs, device=device, max_length=256)
                )
                if len(values) != 2 or not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite model scoring fixture")
                smoke = {"scores": values}
                backward = lora_backward_fixture(spec=spec, model=model, tokenizer=tokenizer, device=device, output_dir=model_dir)
            peak_vram = 0
            try:
                import torch
                if device.startswith("cuda"):
                    peak_vram = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
            report["models"][spec.key] = {
                "state": "PASS",
                "model_id": spec.model_id,
                "kind": spec.kind,
                "total_parameters": total_parameters,
                "full_finetune_candidate": bool(
                    spec.kind == "pair" and total_parameters <= 150_000_000
                ),
                "native_max_length": spec.native_max_length,
                "smoke": smoke,
                "lora": backward,
                "seconds": time.perf_counter() - started,
                "peak_vram_bytes": peak_vram,
            }
        except Exception as error:
            report["models"][spec.key] = {
                "state": "FAILED_MODEL",
                "model_id": spec.model_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        finally:
            if model is not None:
                del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(
            f"[preflight] {report['models'][spec.key]['state']} {spec.key} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
        atomic_json(output_dir / "preflight.json", report)
    report["status"] = "PASS" if any(value["state"] == "PASS" for value in report["models"].values()) else "FAIL"
    atomic_json(output_dir / "preflight.json", report)
    atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "status": report["status"]})
    return report


def score_metrics(scores_path: Path, answers: Mapping[str, set[str]]) -> dict[str, float]:
    predictions: dict[str, list[str]] = {}
    for row in read_jsonl(scores_path):
        qid = str(row["qid"])
        predictions[qid] = [
            str(item["doc_id"])
            for item in sorted(row["scores"], key=lambda item: (-float(item["score"]), str(item["doc_id"])))[:5]
        ]
    if not predictions:
        raise ValueError("no score rows")
    return evaluate_rankings(predictions, {qid: answers[qid] for qid in predictions}, ks=(5,))


def paired_bootstrap_delta(
    baseline_scores: Path, variant_scores: Path, answers: Mapping[str, set[str]], *, samples: int = 2000, seed: int = SEED,
) -> dict[str, float]:
    def recalls(path: Path) -> dict[str, float]:
        result = {}
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if not answers[qid]:
                continue
            top = {
                str(item["doc_id"])
                for item in sorted(row["scores"], key=lambda item: (-float(item["score"]), str(item["doc_id"])))[:5]
            }
            result[qid] = len(top & answers[qid]) / len(answers[qid])
        return result
    baseline, variant = recalls(baseline_scores), recalls(variant_scores)
    qids = sorted(set(baseline) & set(variant))
    if set(baseline) != set(variant) or not qids:
        raise ValueError("paired bootstrap requires identical non-empty qid coverage")
    deltas = np.asarray([variant[qid] - baseline[qid] for qid in qids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = float(deltas[rng.integers(0, len(deltas), len(deltas))].mean())
    return {
        "delta_recall@5": float(deltas.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_positive": float(np.mean(means > 0)),
        "queries": len(qids),
    }


def capsule_gate(per_inner: Sequence[Mapping[str, Any]], *, bounded: bool) -> dict[str, Any]:
    minimum_delta = 0.003 if bounded else 0.002
    deltas = [float(row["delta_recall@5"]) for row in per_inner]
    aggregate = float(np.mean(deltas)) if deltas else float("-inf")
    positive = sum(delta > 0 for delta in deltas)
    passed = bool(deltas) and aggregate >= minimum_delta and min(deltas) >= -0.005
    if bounded:
        passed = passed and positive >= max(1, len(deltas) - 1)
    else:
        passed = passed and all(float(row.get("ci95_low", -1.0)) > 0 for row in per_inner)
    return {
        "passed": passed,
        "bounded": bounded,
        "minimum_delta": minimum_delta,
        "mean_delta_recall@5": aggregate,
        "positive_inner_folds": positive,
        "inner_folds": len(deltas),
        "worst_inner_delta": min(deltas) if deltas else None,
    }


def _load_train_model(
    spec: ModelSpec, *, device: str, local_only: bool, precision: str,
) -> tuple[Any, Any, bool]:
    if spec.kind == "jina":
        raise RuntimeError("Jina remains zero-shot-only without a verified native training contract")
    if precision != "qlora":
        model, tokenizer = _load_model(
            spec, device=device, local_only=local_only, precision=precision,
        )
        return model, tokenizer, False
    import torch
    module_cache = ROOT / "cache" / "exp030_legal_evidence_routing" / "hf_modules"
    module_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(module_cache)
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code, local_files_only=local_only,
        padding_side="left" if spec.kind == "qwen" else "right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    factory = AutoModelForCausalLM if spec.kind == "qwen" else AutoModelForSequenceClassification
    model = factory.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code, local_files_only=local_only,
        quantization_config=quantization, device_map={"": device}, dtype=torch.float16,
    )
    if spec.key == "gte" and hasattr(getattr(model, "new", None), "embeddings"):
        model.new.embeddings.position_ids = torch.arange(
            int(model.config.max_position_embeddings), dtype=torch.long, device=device,
        )
    return model, tokenizer, True


def _training_views(candidate: Mapping[str, Any], query: str, capsule_variant: str) -> list[str]:
    config = CAPSULE_CONFIGS[capsule_variant]
    views = build_candidate_views(
        query, candidate, identity_variant=str(config["identity_variant"]),
        include_structure=bool(config["structure"]), include_scope=bool(config["scope"]),
        include_relations=bool(config["relations"]),
    )
    selected = views if config["view_policy"] == "all" else (select_views(query, views) if config["view_policy"] == "routed" else [views[0]])
    return [str(view["text"]) for view in selected]


def _differentiable_candidate_score(
    *, spec: ModelSpec, model: Any, tokenizer: Any, query: str,
    candidate: Mapping[str, Any], capsule_variant: str, device: str,
) -> Any:
    """Score one candidate at a time so FP16 fallback has micro-batch one."""
    import torch
    scores = []
    for text in _training_views(candidate, query, capsule_variant):
        if spec.kind == "qwen":
            score = _differentiable_qwen_scores(
                model, tokenizer, query, [text], device=device, max_length=COMMON_MAX_LENGTH,
            )[0]
        else:
            batch = tokenizer(
                query, text, padding=False, truncation="only_second",
                max_length=COMMON_MAX_LENGTH, return_tensors="pt",
            )
            if spec.key == "gte":
                batch.pop("token_type_ids", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            score = model(**batch).logits.reshape(-1)[0]
        scores.append(score)
    if not scores:
        raise ValueError("candidate rendered no training view")
    return torch.stack(scores).max()


def _run_pairwise_training(
    *, spec: ModelSpec, model: Any, tokenizer: Any, capsules: Path,
    train_qids: set[str], answers: Mapping[str, set[str]], capsule_variant: str,
    device: str, optimizer: Any, save_epoch: Callable[[int, Mapping[str, Any]], None],
    start_epoch: int = 0, initial_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    initial = dict(initial_state or {})
    optimizer.zero_grad(set_to_none=True)
    update_steps = int(initial.get("optimizer_updates", 0))
    pair_steps = int(initial.get("pair_comparisons", 0))
    accumulation_steps = 0
    loss_sum = float(initial.get("loss_sum", 0.0))
    loss_count = int(initial.get("loss_count", 0))
    bucket_counts: Counter[str] = Counter(initial.get("negative_buckets", {}))
    for epoch in range(start_epoch, LORA["epochs"]):
        ordered_qids = sorted(train_qids, key=lambda qid: _hash([SEED, epoch, qid]))
        for row in read_selected_jsonl(capsules, ordered_qids):
            group = select_hard_negatives(row, answers[str(row["qid"])], epoch=epoch)
            if not group:
                continue
            query = str(row["query"])
            for positive in group["positives"]:
                for negative_entry in group["negatives"]:
                    positive_score = _differentiable_candidate_score(
                        spec=spec, model=model, tokenizer=tokenizer, query=query, candidate=positive,
                        capsule_variant=capsule_variant, device=device,
                    )
                    negative_score = _differentiable_candidate_score(
                        spec=spec, model=model, tokenizer=tokenizer, query=query,
                        candidate=negative_entry["candidate"], capsule_variant=capsule_variant, device=device,
                    )
                    loss = torch.nn.functional.softplus(negative_score - positive_score)
                    (loss / LORA["effective_batch"]).backward()
                    loss_sum += float(loss.detach().cpu())
                    loss_count += 1
                    pair_steps += 1
                    accumulation_steps += 1
                    bucket_counts[negative_entry["bucket"]] += 1
                    if accumulation_steps == LORA["effective_batch"]:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_steps += 1
                        accumulation_steps = 0
        if accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_steps += 1
            accumulation_steps = 0
        save_epoch(
            epoch + 1,
            {
                "pair_comparisons": pair_steps, "optimizer_updates": update_steps,
                "loss_sum": loss_sum, "loss_count": loss_count,
                "negative_buckets": dict(bucket_counts),
            },
        )
    if not loss_count:
        raise ValueError("training fixture contains no retained positive/hard-negative pair")
    return {
        "pair_comparisons": pair_steps,
        "optimizer_updates": update_steps,
        "mean_loss": loss_sum / loss_count,
        "negative_buckets": dict(bucket_counts),
    }


def _latest_epoch_checkpoint(output_dir: Path, artifact_name: str) -> tuple[int, dict[str, Any]] | None:
    for epoch in range(int(LORA["epochs"]), 0, -1):
        directory = output_dir / f"epoch_{epoch}"
        checkpoint = directory / "checkpoint.json"
        if checkpoint.exists() and (directory / artifact_name).exists() and (directory / "optimizer.pt").exists():
            payload = _json(checkpoint)
            if int(payload.get("epoch", -1)) == epoch:
                return epoch, payload
    return None


def train_lora(
    *, spec: ModelSpec, capsules: Path, train_qids: set[str], answers: Mapping[str, set[str]],
    output_dir: Path, device: str, local_only: bool, precision: str = "fp32",
    capsule_variant: str = "multi_view", resume: bool = False,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_epoch_checkpoint(output_dir, "adapter") if resume else None
    model, tokenizer, quantized = _load_train_model(
        spec, device=device, local_only=local_only, precision=precision,
    )
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    contract = discover_lora_targets(spec, model)
    if not contract["eligible"]:
        raise RuntimeError(f"unsafe LoRA target contract for {spec.key}: {contract}")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    task_type = "CAUSAL_LM" if spec.kind == "qwen" else "SEQ_CLS"
    if latest:
        model = PeftModel.from_pretrained(
            model, str(output_dir / f"epoch_{latest[0]}" / "adapter"), is_trainable=True,
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA["r"], lora_alpha=LORA["lora_alpha"], lora_dropout=LORA["lora_dropout"],
                bias="none", task_type=task_type, target_modules=contract["targets"],
            ),
        )
    model.train()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LORA["learning_rate"])
    if latest:
        optimizer.load_state_dict(
            torch.load(output_dir / f"epoch_{latest[0]}" / "optimizer.pt", map_location=device, weights_only=False)
        )
    started = time.perf_counter()
    def save_epoch(epoch: int, state: Mapping[str, Any]) -> None:
        checkpoint = output_dir / f"epoch_{epoch}" / "adapter"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        torch.save(optimizer.state_dict(), output_dir / f"epoch_{epoch}" / "optimizer.pt")
        atomic_json(
            output_dir / f"epoch_{epoch}" / "checkpoint.json",
            {"schema_version": SCHEMA, "epoch": epoch, "training_state": dict(state)},
        )
    training = _run_pairwise_training(
        spec=spec, model=model, tokenizer=tokenizer, capsules=capsules, train_qids=train_qids,
        answers=answers, capsule_variant=capsule_variant, device=device,
        optimizer=optimizer, save_epoch=save_epoch,
        start_epoch=latest[0] if latest else 0,
        initial_state=latest[1].get("training_state", {}) if latest else None,
    )
    peak_vram = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    # Serialization gate: unload and attach the saved adapter again.
    base_model = model.unload()
    reloaded = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    del reloaded, base_model, model
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "model": spec.key,
        "capsule_variant": capsule_variant,
        "qlora": quantized,
        "precision": precision,
        "lora": LORA,
        "target_contract": contract,
        "trainable_parameters": int(trainable),
        "queries": len(train_qids),
        **training,
        "checkpoint_reload": True,
        "peak_vram_bytes": peak_vram,
        "adapter": str(adapter_dir.resolve()),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "train_report.json", report)
    atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "status": "PASS"})
    return report


def train_full(
    *, spec: ModelSpec, capsules: Path, train_qids: set[str], answers: Mapping[str, set[str]],
    output_dir: Path, device: str, local_only: bool, precision: str = "fp32",
    capsule_variant: str = "multi_view", resume: bool = False,
) -> dict[str, Any]:
    if spec.kind in {"jina", "qwen"}:
        raise RuntimeError(f"full fine-tuning is not enabled for {spec.key}")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("full fine-tuning supports fp32 or fp16 only")
    import torch
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_epoch_checkpoint(output_dir, "model") if resume else None
    if latest:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        checkpoint_model = output_dir / f"epoch_{latest[0]}" / "model"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_model, local_files_only=True, dtype=_torch_dtype(precision),
        ).to(device)
    else:
        model, tokenizer = _load_model(
            spec, device=device, local_only=local_only, precision=precision,
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters > 150_000_000:
        raise RuntimeError(
            f"full fine-tuning safety cap exceeded: {total_parameters} > 150000000 parameters"
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LORA["learning_rate"])
    if latest:
        optimizer.load_state_dict(
            torch.load(output_dir / f"epoch_{latest[0]}" / "optimizer.pt", map_location=device, weights_only=False)
        )
    started = time.perf_counter()

    def save_epoch(epoch: int, state: Mapping[str, Any]) -> None:
        checkpoint = output_dir / f"epoch_{epoch}" / "model"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        torch.save(optimizer.state_dict(), output_dir / f"epoch_{epoch}" / "optimizer.pt")
        atomic_json(
            output_dir / f"epoch_{epoch}" / "checkpoint.json",
            {"schema_version": SCHEMA, "epoch": epoch, "training_state": dict(state)},
        )

    training = _run_pairwise_training(
        spec=spec, model=model, tokenizer=tokenizer, capsules=capsules, train_qids=train_qids,
        answers=answers, capsule_variant=capsule_variant, device=device,
        optimizer=optimizer, save_epoch=save_epoch,
        start_epoch=latest[0] if latest else 0,
        initial_state=latest[1].get("training_state", {}) if latest else None,
    )
    peak_vram = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    # Full checkpoint reload is a hard gate, just like adapter reload.
    from transformers import AutoModelForSequenceClassification
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=spec.trust_remote_code, local_files_only=True,
        dtype=_torch_dtype(precision),
    )
    del reloaded
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "model": spec.key,
        "training_mode": "full",
        "precision": precision,
        "capsule_variant": capsule_variant,
        "total_parameters": int(total_parameters),
        "queries": len(train_qids),
        **training,
        "checkpoint_reload": True,
        "peak_vram_bytes": peak_vram,
        "model_path": str(model_dir.resolve()),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "train_report.json", report)
    atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "status": "PASS"})
    return report


def score_full_model(
    *, spec: ModelSpec, capsules: Path, model_path: Path, output_dir: Path, qids: set[str],
    device: str, local_only: bool, precision: str, capsule_variant: str,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=spec.trust_remote_code, local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=spec.trust_remote_code, local_files_only=True,
        dtype=_torch_dtype(precision),
    ).to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    try:
        for row in read_selected_jsonl(capsules, qids):
            rows.append(
                score_query_record(
                    spec=spec, model=model, tokenizer=tokenizer, row=row, device=device,
                    capsule_variant=capsule_variant, max_length=COMMON_MAX_LENGTH,
                )
            )
            if len(rows) % 16 == 0:
                write_jsonl(output_dir / "scores.jsonl", rows)
        write_jsonl(output_dir / "scores.jsonl", rows)
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    report = {
        "schema_version": SCHEMA, "model": spec.key, "training_mode": "full",
        "precision": precision, "queries": len(rows), "seconds": time.perf_counter() - started,
        "scores": str((output_dir / "scores.jsonl").resolve()),
    }
    atomic_json(output_dir / "score_report.json", report)
    return report


def score_lora_capsules(
    *, spec: ModelSpec, capsules: Path, adapter: Path, output_dir: Path, qids: set[str],
    device: str, local_only: bool, identity_variant: str, capsule_variant: str = "multi_view",
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_model(spec, device=device, local_only=local_only)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).to(device).eval()
    rows = []
    started = time.perf_counter()
    try:
        for row in read_selected_jsonl(capsules, qids):
            rows.append(
                score_query_record(
                    spec=spec, model=model, tokenizer=tokenizer, row=row, device=device,
                    identity_variant=identity_variant, aggregation="max", max_length=COMMON_MAX_LENGTH,
                    capsule_variant=capsule_variant,
                )
            )
            if len(rows) % 16 == 0:
                write_jsonl(output_dir / "scores.jsonl", rows)
        write_jsonl(output_dir / "scores.jsonl", rows)
    finally:
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    report = {
        "schema_version": SCHEMA, "model": spec.key, "queries": len(rows),
        "seconds": time.perf_counter() - started, "scores": str((output_dir / "scores.jsonl").resolve()),
    }
    atomic_json(output_dir / "score_report.json", report)
    return report


def audit_data(
    *, candidates: Path, train: Path, folds_path: Path, v3_dir: Path,
    preprocessing_dir: Path, output_dir: Path,
    expected_label_accounting: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    require_success(candidates.parent)
    require_success(v3_dir)
    folds = load_folds(folds_path)
    train_payload = _json(train)
    answers, label_stats = canonical_answers(
        train, preprocessing_dir / "exclusions.json",
        preprocessing_dir / "train_label_impact.jsonl",
    )
    if expected_label_accounting:
        mismatches = {
            key: (label_stats.get(key), expected)
            for key, expected in expected_label_accounting.items()
            if label_stats.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"canonical label accounting mismatch: {mismatches}")
    questions = {str(qid): str(row["question"]) for qid, row in train_payload.items()}
    folded = {str(qid) for values in folds.values() for qid in values}
    documents = list(read_jsonl(v3_dir / "documents.jsonl"))
    candidate_rows = list(read_jsonl(candidates))
    if set(answers) != folded or set(answers) != {str(row["qid"]) for row in candidate_rows}:
        raise ValueError("train/fold/candidate qid membership mismatch")
    if any(len(row.get("candidates", [])) != 150 for row in candidate_rows):
        raise ValueError("frozen candidate artifact must contain exactly 150 parent documents per query")
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "queries": len(answers),
        "accented_queries": sum(has_vietnamese_accent(question) for question in questions.values()),
        "documents": len(documents),
        "accented_document_labels": sum(has_vietnamese_accent(str(row.get("document_label") or "")) for row in documents),
        "legacy_scope_nodes": sum(len(row.get("scope_node_ids", [])) for row in documents),
        "documents_with_legacy_scope": sum(bool(row.get("scope_node_ids")) for row in documents),
        "gold_count_distribution": dict(Counter(len(value) for value in answers.values())),
        "canonical_label_accounting": label_stats,
        "candidate_pairs": len(candidate_rows) * 150,
        "inputs": {
            "candidates_sha256": sha256_file(candidates),
            "train_sha256": sha256_file(train),
            "folds_sha256": sha256_file(folds_path),
            "v3_manifest_sha256": sha256_file(v3_dir / "manifest.json"),
            "preprocessing_manifest_sha256": sha256_file(preprocessing_dir / "manifest.json"),
            "exclusions_sha256": sha256_file(preprocessing_dir / "exclusions.json"),
            "train_label_impact_sha256": sha256_file(preprocessing_dir / "train_label_impact.jsonl"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "data_audit.json", report)
    atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "status": "PASS"})
    return report


def _state(root: Path, event: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA, "at": time.time(), **event}
    with (root / "state.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload) + "\n")
    atomic_json(root / "RUN_STATUS.json", payload)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event.get('job', 'runner')}: {event.get('state', '')}", flush=True)


def _job(
    root: Path, name: str, fingerprint: str, function: Callable[[], dict[str, Any]], *,
    resume: bool, retries: int = 0, isolate: str | None = None,
) -> dict[str, Any]:
    directory = root / "jobs" / name
    success, failed = directory / "_SUCCESS.json", directory / "_FAILED.json"
    if resume and success.exists() and _json(success).get("fingerprint") == fingerprint:
        _state(root, {"job": name, "state": "SKIPPED_RESUME", "fingerprint": fingerprint})
        return _json(success)
    directory.mkdir(parents=True, exist_ok=True)
    success.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    for attempt in range(retries + 1):
        _state(root, {"job": name, "state": "RUNNING", "attempt": attempt, "fingerprint": fingerprint})
        try:
            result = function()
            marker = {"fingerprint": fingerprint, "finished_at": time.time(), "payload": result}
            atomic_json(success, marker)
            _state(root, {"job": name, "state": "SUCCESS", "fingerprint": fingerprint})
            return marker
        except Exception as error:
            detail = {
                "fingerprint": fingerprint, "attempt": attempt,
                "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                "finished_at": time.time(),
            }
            atomic_json(failed, detail)
            _state(root, {"job": name, "state": "RETRY" if attempt < retries else (isolate or "FAILED"), **detail})
    if isolate:
        return {"state": isolate, "error": _json(failed)["error"]}
    raise RuntimeError(_json(failed)["error"])


def _qid_set(path: Path) -> set[str]:
    value = _json(path)
    values = value if isinstance(value, list) else [value]
    return {str(item) for item in values}


def _worker(
    *, command: str, output_dir: Path, model: str, capsules: Path, qids: set[str], device: str,
    local_only: bool, capsule_variant: str = "multi_view", adapter: Path | None = None,
    precision: str = "fp32",
    training_mode: str = "lora", model_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qids_path = output_dir / "worker_qids.json"
    atomic_json(qids_path, sorted(qids))
    arguments = [
        sys.executable, str(Path(__file__).resolve()), command,
        "--model", model, "--capsules", str(capsules), "--output", str(output_dir),
        "--qids-json", str(qids_path), "--device", device, "--capsule-variant", capsule_variant,
    ]
    if local_only:
        arguments.append("--local-only")
    if adapter:
        arguments.extend(("--adapter", str(adapter)))
    arguments.extend(("--precision", precision))
    arguments.extend(("--training-mode", training_mode))
    if model_path:
        arguments.extend(("--model-path", str(model_path)))
    if command in {"score-worker", "train-worker"}:
        arguments.append("--resume")
    environment = dict(os.environ)
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    with (output_dir / "worker.stdout.log").open("w", encoding="utf-8") as stdout, (output_dir / "worker.stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(arguments, cwd=str(ROOT), env=environment, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode:
        tail = (output_dir / "worker.stderr.log").read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"worker {command} exit={completed.returncode}: {tail}")
    return _json(output_dir / "worker_result.json")


def _combine_score_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            qid = str(row["qid"])
            if qid in rows:
                raise ValueError(f"duplicate score qid while combining folds: {qid}")
            rows[qid] = row
    return [rows[qid] for qid in sorted(rows)]


def error_analysis(scores_path: Path, answers: Mapping[str, set[str]], output: Path) -> dict[str, Any]:
    buckets: Counter[str] = Counter()
    avoidable = 0
    rows = []
    for row in read_jsonl(scores_path):
        qid = str(row["qid"])
        ordered = [str(item["doc_id"]) for item in sorted(row["scores"], key=lambda item: (-float(item["score"]), str(item["doc_id"]))) ]
        gold_ranks = [ordered.index(doc_id) + 1 for doc_id in answers[qid] if doc_id in ordered]
        missed = [doc_id for doc_id in answers[qid] if doc_id not in set(ordered[:5])]
        if not missed:
            continue
        for doc_id in missed:
            if doc_id not in ordered:
                bucket = "absent_k64"
            else:
                avoidable += 1
                rank = ordered.index(doc_id) + 1
                bucket = "rank_06_10" if rank <= 10 else ("rank_11_32" if rank <= 32 else "rank_33_64")
            buckets[bucket] += 1
        rows.append({"qid": qid, "missed_gold": missed, "gold_ranks": gold_ranks})
    report = {
        "schema_version": SCHEMA, "miss_buckets": dict(buckets), "avoidable_misses": avoidable,
        "dominant_rescue_buckets": [key for key, value in buckets.items() if avoidable and value / avoidable >= 0.25 and key != "absent_k64"],
        "queries_with_misses": len(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {**report, "rows": rows})
    return report


def final_report(*, results_root: Path, answers: Mapping[str, set[str]], folds: Mapping[str, list[str]]) -> dict[str, Any]:
    selection_path = results_root / "selection.json"
    selection = _json(selection_path) if selection_path.exists() else {}
    zero_paths: list[Path] = []
    capsule_baseline_paths: list[Path] = []
    trained_paths: list[Path] = []
    per_fold: dict[str, Any] = {}
    for outer in sorted(folds):
        winner = selection.get("winners", {}).get(outer, {}).get("model")
        if not winner:
            continue
        zero = results_root / "heldout_zero" / outer / winner / "scores.jsonl"
        capsule_baseline = results_root / "heldout_zero_baseline" / outer / winner / "scores.jsonl"
        trained = results_root / "heldout_trained" / outer / winner / "scores.jsonl"
        row: dict[str, Any] = {"model": winner}
        if zero.exists():
            zero_paths.append(zero)
            row["zero_shot"] = score_metrics(zero, answers)
        if capsule_baseline.exists():
            capsule_baseline_paths.append(capsule_baseline)
            row["unaccented_capsule_baseline"] = score_metrics(capsule_baseline, answers)
        if trained.exists():
            trained_paths.append(trained)
            row["fine_tuned"] = score_metrics(trained, answers)
            training_selection = results_root / "training_selection" / f"{outer}.json"
            if training_selection.exists():
                row["training"] = _json(training_selection)
        per_fold[outer] = row
    aggregate: dict[str, Any] = {}
    temporary = results_root / "report_inputs"
    temporary.mkdir(parents=True, exist_ok=True)
    if zero_paths:
        write_jsonl(temporary / "zero_scores.jsonl", _combine_score_rows(zero_paths))
        aggregate["zero_shot"] = score_metrics(temporary / "zero_scores.jsonl", answers)
    if capsule_baseline_paths:
        write_jsonl(temporary / "capsule_baseline_scores.jsonl", _combine_score_rows(capsule_baseline_paths))
        aggregate["unaccented_capsule_baseline"] = score_metrics(
            temporary / "capsule_baseline_scores.jsonl", answers,
        )
    if trained_paths:
        write_jsonl(temporary / "trained_scores.jsonl", _combine_score_rows(trained_paths))
        aggregate["fine_tuned"] = score_metrics(temporary / "trained_scores.jsonl", answers)
        aggregate["error_analysis"] = error_analysis(
            temporary / "trained_scores.jsonl", answers, results_root / "error_analysis.json",
        )
    complete_training = len(trained_paths) == len(folds) and len(zero_paths) == len(folds)
    fold_recall_deltas = {
        outer: row["fine_tuned"]["recall@5"] - row["zero_shot"]["recall@5"]
        for outer, row in per_fold.items() if "fine_tuned" in row and "zero_shot" in row
    }
    aggregate_recall_delta = (
        aggregate["fine_tuned"]["recall@5"] - aggregate["zero_shot"]["recall@5"]
        if "fine_tuned" in aggregate and "zero_shot" in aggregate else None
    )
    aggregate_precision_delta = (
        aggregate["fine_tuned"]["precision@5"] - aggregate["zero_shot"]["precision@5"]
        if "fine_tuned" in aggregate and "zero_shot" in aggregate else None
    )
    promotion_gate = {
        "complete_all_outer_folds": complete_training,
        "aggregate_recall_delta": aggregate_recall_delta,
        "required_aggregate_recall_delta": 0.002,
        "aggregate_precision_delta": aggregate_precision_delta,
        "required_min_precision_delta": 0.0,
        "per_fold_recall_deltas": fold_recall_deltas,
        "required_min_fold_recall_delta": -0.005,
    }
    promotable = bool(
        complete_training
        and aggregate_recall_delta is not None and aggregate_recall_delta >= 0.002
        and aggregate_precision_delta is not None and aggregate_precision_delta >= 0.0
        and len(fold_recall_deltas) == len(folds)
        and min(fold_recall_deltas.values(), default=-math.inf) >= -0.005
    )
    promotion_gate["promotable"] = promotable
    target_met = complete_training and aggregate.get("fine_tuned", {}).get("recall@5", 0.0) >= 0.970
    if target_met and promotable:
        status = "TARGET_MET"
    elif promotable:
        status = "IMPROVED_BUT_BELOW_TARGET"
    elif aggregate.get("fine_tuned"):
        status = "NON_PROMOTABLE"
    elif (results_root / "GATE_BLOCKED.json").exists():
        status = "REJECTED"
    else:
        status = "PARTIAL"
    events = list(read_jsonl(results_root / "state.jsonl")) if (results_root / "state.jsonl").exists() else []
    report = {
        "schema_version": SCHEMA, "status": status, "target_recall@5": 0.970,
        "model_cards": MODEL_CARD_METADATA,
        "selection": selection, "per_fold": per_fold, "aggregate": aggregate,
        "promotion_gate": promotion_gate,
        "completed_fine_tuned_folds": len(trained_paths), "required_folds": len(folds),
        "failures": [event for event in events if str(event.get("state", "")).startswith("FAILED")],
    }
    atomic_json(results_root / "REPORT.json", report)
    return report


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "candidates": args.candidates,
        "sidecar": args.sidecar,
        "features": args.features,
        "train": args.train,
        "folds": args.folds,
        "preprocessing": args.preprocessing,
        "v3": args.v3,
        "bm25_db": args.bm25_db,
        "cache": args.cache_root,
        "results": args.results_root,
    }


def training_attempts(model_preflight: Mapping[str, Any], *, bitsandbytes: bool) -> list[tuple[str, str]]:
    """Prefer full FP32, then progressively lower precision/parameter scope."""
    attempts: list[tuple[str, str]] = []
    if model_preflight.get("full_finetune_candidate"):
        attempts.extend(("full", precision) for precision in ("fp32", "fp16"))
    if model_preflight.get("lora", {}).get("state") == "ELIGIBLE":
        attempts.extend(("lora", precision) for precision in ("fp32", "fp16"))
        if bitsandbytes:
            attempts.append(("lora", "qlora"))
    return attempts


def overnight(args: argparse.Namespace) -> dict[str, Any]:
    from exp029_nested_lora_reranker import build_cascade

    paths = _paths(args)
    root = args.results_root
    answers, label_stats = canonical_answers(
        args.train, args.preprocessing / "exclusions.json",
        args.preprocessing / "train_label_impact.jsonl",
    )
    label_fingerprint = str(label_stats["label_fingerprint"])
    dependency_code = {
        name: sha256_file(ROOT / "src" / name)
        for name in (
            "exp030_legal_evidence_routing.py",
            "exp029_nested_lora_reranker.py",
            "exp028_lambdamart_shortlist.py",
            "exp027_lambdamart_shortlist.py",
            "exp026_lambdamart_capsules.py",
            "exp012b_core.py",
        )
    }
    code_sha256 = _hash(dependency_code)
    fingerprint = _hash(
        {
            "schema": SCHEMA,
            "code_sha256": code_sha256,
            "label_policy": LABEL_POLICY,
            "label_fingerprint": label_fingerprint,
            "inputs": {
                key: sha256_file(path if path.is_file() else path / "manifest.json")
                for key, path in paths.items()
                if key not in {"cache", "results", "bm25_db"}
            },
            "bm25_manifest_sha256": sha256_file(args.bm25_db.parent / "manifest.json"),
            "configs": CAPSULE_CONFIGS,
            "lora": LORA,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    resolved_paths = {key: str(value.resolve()) for key, value in paths.items()}
    model_contract = [asdict(spec) for spec in MODELS]
    compatibility = None
    existing_manifest_path = root / "run_manifest.json"
    if args.resume and existing_manifest_path.exists():
        existing_manifest = _json(existing_manifest_path)
        existing_fingerprint = str(existing_manifest.get("fingerprint") or "")
        if (
            existing_fingerprint in HOTFIX_COMPATIBLE_RUNS
            and existing_manifest.get("paths") == resolved_paths
            and existing_manifest.get("capsule_configs") == CAPSULE_CONFIGS
            and existing_manifest.get("models") == model_contract
        ):
            fingerprint = existing_fingerprint
            compatibility = HOTFIX_COMPATIBLE_RUNS[existing_fingerprint]
    atomic_json(
        existing_manifest_path,
        {
            "schema_version": SCHEMA, "fingerprint": fingerprint,
            "current_code_sha256": code_sha256, "code_fingerprints": dependency_code,
            "resume_compatibility": compatibility,
            "label_policy": LABEL_POLICY, "label_fingerprint": label_fingerprint,
            "label_accounting": label_stats,
            "paths": resolved_paths, "models": model_contract, "capsule_configs": CAPSULE_CONFIGS,
        },
    )
    _job(
        root, "audit-data", fingerprint,
        lambda: audit_data(
            candidates=args.candidates, train=args.train, folds_path=args.folds,
            v3_dir=args.v3, preprocessing_dir=args.preprocessing, output_dir=root / "audit",
            expected_label_accounting=EXPECTED_LABEL_ACCOUNTING,
        ),
        resume=args.resume,
    )
    metadata_dir = args.cache_root / "metadata"
    _job(
        root, "build-metadata", fingerprint,
        lambda: build_metadata_sidecar(v3_dir=args.v3, preprocessing_dir=args.preprocessing, output_dir=metadata_dir),
        resume=args.resume,
    )
    # A label-dependent cascade must never share EXP-029's historical cache or
    # a cache generated by another EXP-030 label policy.
    cascade_dir = args.cache_root / "canonical_cascade" / label_fingerprint[:16]
    cascade_paths = {
        "candidates": args.candidates, "sidecar": args.sidecar, "features": args.features,
        "train": args.train, "folds": args.folds, "preprocessing": args.preprocessing,
        "v3": args.v3, "bm25_db": args.bm25_db, "cache": args.exp029_cache,
        "results": ROOT / "results" / "exp029_nested_lora_reranker",
    }
    _job(
        root, "build-fold-isolated-cascade", fingerprint,
        lambda: build_cascade(paths=cascade_paths, output=cascade_dir, answers_override=answers),
        resume=args.resume,
    )
    folds = load_folds(args.folds)
    evaluable_qids = {qid for qid, gold in answers.items() if gold}
    enriched_by_outer: dict[str, Path] = {}
    for outer in sorted(folds):
        source = cascade_dir / outer / "capsules" / "capsules.jsonl"
        output = args.cache_root / "capsules" / label_fingerprint[:16] / outer
        enriched_by_outer[outer] = output / "capsules.jsonl"
        _job(
            root, f"enrich-{outer}", _hash([fingerprint, outer, sha256_file(source)]),
            lambda source=source, output=output: enrich_capsules(
                capsules=source, metadata_dir=metadata_dir, v3_dir=args.v3, output_dir=output,
                frozen_candidates=args.candidates,
            ),
            resume=args.resume,
        )
    preflight_fingerprint = _hash([fingerprint, "preflight", PREFLIGHT_CONTRACT])
    preflight_marker = _job(
        root, "preflight-models", preflight_fingerprint,
        lambda: preflight_models(output_dir=root / "preflight", device=args.device, local_only=args.local_only),
        resume=args.resume,
    )
    preflight = preflight_marker["payload"]
    anchors = ("bge_m3", "gte")
    unavailable_anchors = [model for model in anchors if preflight["models"].get(model, {}).get("state") != "PASS"]
    if unavailable_anchors:
        blocked = {"schema_version": SCHEMA, "reason": "required_anchor_preflight_failed", "models": unavailable_anchors}
        atomic_json(root / "GATE_BLOCKED.json", blocked)
        return final_report(results_root=root, answers=answers, folds=folds)

    screen_rows: list[dict[str, Any]] = []
    variants = tuple(CAPSULE_CONFIGS)
    for outer in sorted(folds):
        capsule = enriched_by_outer[outer]
        for inner in sorted(set(folds) - {outer}):
            inner_qids = {str(qid) for qid in folds[inner]}
            sample = stratified_sample(
                read_selected_jsonl(capsule, inner_qids), answers, limit=SCREEN_PER_INNER, seed=SEED,
            )
            for model_key in anchors:
                outputs: dict[str, Path] = {}
                for variant in variants:
                    output = root / "bounded_screen" / outer / inner / model_key / variant
                    marker = _job(
                        root, f"bounded-{outer}-{inner}-{model_key}-{variant}",
                        _hash([fingerprint, outer, inner, model_key, variant, sorted(sample)]),
                        lambda model_key=model_key, capsule=capsule, output=output, sample=sample, variant=variant: _worker(
                            command="score-worker", output_dir=output, model=model_key, capsules=capsule,
                            qids=sample, device=args.device, local_only=args.local_only, capsule_variant=variant,
                        ),
                        resume=args.resume, retries=1, isolate="FAILED_MODEL",
                    )
                    if marker.get("state") == "FAILED_MODEL":
                        continue
                    outputs[variant] = output / "scores.jsonl"
                baseline = outputs.get("unaccented_base")
                if not baseline:
                    continue
                for variant, score_path in outputs.items():
                    if variant == "unaccented_base":
                        continue
                    delta = paired_bootstrap_delta(baseline, score_path, answers, samples=1000, seed=SEED)
                    screen_rows.append(
                        {
                            "outer": outer, "inner": inner, "model": model_key, "variant": variant,
                            "baseline": score_metrics(baseline, answers), "metrics": score_metrics(score_path, answers), **delta,
                        }
                    )
    gate_rows: dict[str, Any] = {}
    passing_variants: list[dict[str, Any]] = []
    for variant in variants:
        if variant == "unaccented_base":
            continue
        per_anchor = {}
        for model_key in anchors:
            outer_values = []
            for outer in sorted(folds):
                values = [
                    row for row in screen_rows
                    if row["variant"] == variant and row["model"] == model_key and row["outer"] == outer
                ]
                if len(values) == len(folds) - 1:
                    outer_values.append({"outer": outer, "delta_recall@5": float(np.mean([row["delta_recall@5"] for row in values]))})
            per_anchor[model_key] = capsule_gate(outer_values, bounded=True)
        passed = all(value["passed"] for value in per_anchor.values())
        mean_delta = float(np.mean([value["mean_delta_recall@5"] for value in per_anchor.values()]))
        gate_rows[variant] = {"passed": passed, "mean_delta_recall@5": mean_delta, "per_anchor": per_anchor}
        if passed:
            passing_variants.append({"variant": variant, "mean_delta_recall@5": mean_delta})
    gate_report = {"schema_version": SCHEMA, "rows": screen_rows, "variants": gate_rows}
    atomic_json(root / "bounded_gate.json", gate_report)
    if not passing_variants:
        atomic_json(
            root / "GATE_BLOCKED.json",
            {"schema_version": SCHEMA, "reason": "no_capsule_variant_passed_bounded_gate", "gate": gate_rows},
        )
        return final_report(results_root=root, answers=answers, folds=folds)
    selected_variant = sorted(passing_variants, key=lambda row: (-row["mean_delta_recall@5"], row["variant"]))[0]["variant"]

    # Full confirmation is separate from the bounded 64-query discovery gate.
    # Combine all four inner validation folds within each outer fold, then use
    # paired bootstrap and require stability on both anchor architectures.
    full_gate_rows: list[dict[str, Any]] = []
    for outer in sorted(folds):
        capsule = enriched_by_outer[outer]
        for model_key in anchors:
            baseline_paths: list[Path] = []
            variant_paths: list[Path] = []
            failed = False
            for inner in sorted(set(folds) - {outer}):
                qids = {str(qid) for qid in folds[inner]} & evaluable_qids
                baseline_output = root / "full_gate_baseline" / outer / inner / model_key
                baseline_marker = _job(
                    root, f"full-gate-baseline-{outer}-{inner}-{model_key}",
                    _hash([fingerprint, outer, inner, model_key, "unaccented_base"]),
                    lambda model_key=model_key, capsule=capsule, baseline_output=baseline_output,
                    qids=qids: _worker(
                        command="score-worker", output_dir=baseline_output, model=model_key,
                        capsules=capsule, qids=qids, device=args.device, local_only=args.local_only,
                        capsule_variant="unaccented_base",
                    ),
                    resume=True, retries=1, isolate="FAILED_MODEL",
                )
                variant_output = root / "full_screen" / outer / inner / model_key
                variant_marker = _job(
                    root, f"full-{outer}-{inner}-{model_key}",
                    _hash([fingerprint, outer, inner, model_key, selected_variant]),
                    lambda model_key=model_key, capsule=capsule, variant_output=variant_output,
                    qids=qids: _worker(
                        command="score-worker", output_dir=variant_output, model=model_key,
                        capsules=capsule, qids=qids, device=args.device, local_only=args.local_only,
                        capsule_variant=selected_variant,
                    ),
                    resume=args.resume, retries=1, isolate="FAILED_MODEL",
                )
                if baseline_marker.get("state") == "FAILED_MODEL" or variant_marker.get("state") == "FAILED_MODEL":
                    failed = True
                    break
                baseline_paths.append(baseline_output / "scores.jsonl")
                variant_paths.append(variant_output / "scores.jsonl")
            if failed:
                continue
            combined = root / "full_gate_combined" / outer / model_key
            combined.mkdir(parents=True, exist_ok=True)
            baseline_combined = combined / "baseline.jsonl"
            variant_combined = combined / "variant.jsonl"
            write_jsonl(baseline_combined, _combine_score_rows(baseline_paths))
            write_jsonl(variant_combined, _combine_score_rows(variant_paths))
            delta = paired_bootstrap_delta(
                baseline_combined, variant_combined, answers, samples=2000, seed=SEED,
            )
            full_gate_rows.append({"outer": outer, "model": model_key, **delta})
    full_gate_by_anchor = {
        model_key: capsule_gate(
            [row for row in full_gate_rows if row["model"] == model_key], bounded=False,
        )
        for model_key in anchors
    }
    full_gate_passed = all(
        value["passed"] and value["inner_folds"] == len(folds)
        for value in full_gate_by_anchor.values()
    )
    atomic_json(
        root / "full_capsule_gate.json",
        {
            "schema_version": SCHEMA, "variant": selected_variant,
            "passed": full_gate_passed, "rows": full_gate_rows, "per_anchor": full_gate_by_anchor,
        },
    )
    if not full_gate_passed:
        atomic_json(
            root / "GATE_BLOCKED.json",
            {"schema_version": SCHEMA, "reason": "capsule_variant_failed_full_confirmation", "gate": full_gate_by_anchor},
        )
        return final_report(results_root=root, answers=answers, folds=folds)

    full_screen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    available_models = [
        spec for spec in MODELS if preflight["models"].get(spec.key, {}).get("state") == "PASS"
    ]
    for outer in sorted(folds):
        capsule = enriched_by_outer[outer]
        for spec in available_models:
            per_inner = []
            failed = False
            for inner in sorted(set(folds) - {outer}):
                qids = {str(qid) for qid in folds[inner]} & evaluable_qids
                output = root / "full_screen" / outer / inner / spec.key
                marker = _job(
                    root, f"full-{outer}-{inner}-{spec.key}",
                    _hash([fingerprint, outer, inner, spec.key, selected_variant]),
                    lambda spec=spec, capsule=capsule, output=output, qids=qids: _worker(
                        command="score-worker", output_dir=output, model=spec.key, capsules=capsule, qids=qids,
                        device=args.device, local_only=args.local_only, capsule_variant=selected_variant,
                    ),
                    resume=True, retries=1, isolate="FAILED_MODEL",
                )
                if marker.get("state") == "FAILED_MODEL":
                    failed = True
                    break
                metrics = score_metrics(output / "scores.jsonl", answers)
                per_inner.append({"inner": inner, **metrics, "seconds": marker["payload"]["seconds"]})
            if not failed and len(per_inner) == len(folds) - 1:
                full_screen[outer].append(
                    {
                        "model": spec.key,
                        "recall@5": float(np.mean([row["recall@5"] for row in per_inner])),
                        "precision@5": float(np.mean([row["precision@5"] for row in per_inner])),
                        "seconds": sum(row["seconds"] for row in per_inner), "per_inner": per_inner,
                    }
                )
    canonical = {spec.key: index for index, spec in enumerate(MODELS)}
    winners = {
        outer: sorted(rows, key=lambda row: (-row["recall@5"], -row["precision@5"], row["seconds"], canonical[row["model"]]))[0]
        for outer, rows in full_screen.items() if rows
    }
    atomic_json(
        root / "selection.json",
        {"schema_version": SCHEMA, "capsule_variant": selected_variant, "winners": winners, "screen": full_screen},
    )
    for outer, winner in sorted(winners.items()):
        spec = MODEL_BY_KEY[winner["model"]]
        capsule = enriched_by_outer[outer]
        heldout = {str(qid) for qid in folds[outer]} & evaluable_qids
        baseline_output = root / "heldout_zero_baseline" / outer / spec.key
        _job(
            root, f"heldout-zero-baseline-{outer}",
            _hash([fingerprint, outer, spec.key, "unaccented_base", "heldout-zero-baseline"]),
            lambda spec=spec, capsule=capsule, heldout=heldout, baseline_output=baseline_output: _worker(
                command="score-worker", output_dir=baseline_output, model=spec.key, capsules=capsule, qids=heldout,
                device=args.device, local_only=args.local_only, capsule_variant="unaccented_base",
            ),
            resume=args.resume, retries=1, isolate="FAILED_OUTER",
        )
        zero_output = root / "heldout_zero" / outer / spec.key
        zero_marker = _job(
            root, f"heldout-zero-{outer}", _hash([fingerprint, outer, spec.key, selected_variant, "heldout-zero"]),
            lambda spec=spec, capsule=capsule, heldout=heldout, zero_output=zero_output: _worker(
                command="score-worker", output_dir=zero_output, model=spec.key, capsules=capsule, qids=heldout,
                device=args.device, local_only=args.local_only, capsule_variant=selected_variant,
            ),
            resume=args.resume, retries=1, isolate="FAILED_OUTER",
        )
        if zero_marker.get("state") == "FAILED_OUTER":
            continue
        model_preflight = preflight["models"][spec.key]
        can_full = bool(model_preflight.get("full_finetune_candidate"))
        can_lora = model_preflight.get("lora", {}).get("state") == "ELIGIBLE"
        if not can_full and not can_lora:
            continue
        train_qids = {
            str(qid) for fold, qids in folds.items() if fold != outer for qid in qids
        } & evaluable_qids
        attempts = training_attempts(model_preflight, bitsandbytes=bool(preflight.get("bitsandbytes")))
        trained: dict[str, Any] | None = None
        selected_mode = ""
        for training_mode, precision in attempts:
            train_output = root / "trained" / outer / spec.key / training_mode / precision
            marker = _job(
                root, f"train-{training_mode}-{precision}-{outer}",
                _hash([fingerprint, outer, spec.key, selected_variant, LORA, training_mode, precision]),
                lambda spec=spec, capsule=capsule, train_qids=train_qids,
                train_output=train_output, precision=precision, training_mode=training_mode: _worker(
                    command="train-worker", output_dir=train_output, model=spec.key,
                    capsules=capsule, qids=train_qids, device=args.device,
                    local_only=args.local_only, capsule_variant=selected_variant,
                    precision=precision, training_mode=training_mode,
                ),
                resume=args.resume, isolate="FAILED_PRECISION",
            )
            if marker.get("state") != "FAILED_PRECISION":
                trained = marker
                selected_mode = training_mode
                break
        if trained is None:
            _state(root, {"job": f"train-{outer}", "state": "FAILED_OUTER", "model": spec.key})
            continue
        artifact_path = Path(
            trained["payload"]["model_path" if selected_mode == "full" else "adapter"]
        )
        atomic_json(
            root / "training_selection" / f"{outer}.json",
            {
                "schema_version": SCHEMA, "model": spec.key, "training_mode": selected_mode,
                "precision": trained["payload"]["precision"], "artifact": str(artifact_path.resolve()),
                "total_parameters": model_preflight.get("total_parameters"),
            },
        )
        trained_output = root / "heldout_trained" / outer / spec.key
        score_command = "score-full-worker" if selected_mode == "full" else "score-lora-worker"
        _job(
            root, f"heldout-trained-{outer}",
            _hash([fingerprint, outer, spec.key, selected_variant, "heldout-trained", selected_mode]),
            lambda spec=spec, capsule=capsule, heldout=heldout, trained_output=trained_output,
            artifact_path=artifact_path, score_command=score_command, selected_mode=selected_mode,
            precision=trained["payload"]["precision"]: _worker(
                command=score_command, output_dir=trained_output, model=spec.key, capsules=capsule, qids=heldout,
                device=args.device, local_only=args.local_only, capsule_variant=selected_variant,
                adapter=artifact_path if selected_mode == "lora" else None,
                model_path=artifact_path if selected_mode == "full" else None,
                training_mode=selected_mode, precision=precision,
            ),
            resume=args.resume, retries=1, isolate="FAILED_OUTER",
        )
    return final_report(results_root=root, answers=answers, folds=folds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "audit-data", "build-metadata", "build-capsules", "preflight-models", "score-worker",
            "train-worker", "score-lora-worker", "score-full-worker", "analyze-errors", "report", "overnight",
        ),
    )
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "exp030_legal_evidence_routing")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "exp030_legal_evidence_routing")
    parser.add_argument("--exp029-cache", type=Path, default=ROOT / "cache" / "exp029_nested_lora_reranker")
    parser.add_argument("--candidates", type=Path, default=ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl")
    parser.add_argument("--sidecar", type=Path, default=ROOT / "cache" / "exp027_lambdamart_shortlist" / "provenance" / "provenance_sidecar.jsonl")
    parser.add_argument("--features", type=Path, default=ROOT / "cache" / "exp027_lambdamart_shortlist" / "features")
    parser.add_argument("--train", type=Path, default=ROOT / "public_test_dataset" / "train.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "cache" / "cv_folds.json")
    parser.add_argument("--preprocessing", type=Path, default=ROOT / "cache" / "final_preprocessed_v2")
    parser.add_argument("--v3", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1")
    parser.add_argument("--bm25-db", type=Path, default=ROOT / "cache" / "exp021_sparse" / "passage_hierarchy" / "fts5" / "bm25_v3.sqlite")
    parser.add_argument("--capsules", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_BY_KEY))
    parser.add_argument("--qids-json", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--capsule-variant", choices=tuple(CAPSULE_CONFIGS), default="multi_view")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16", "qlora"), default="fp32")
    parser.add_argument("--training-mode", choices=("full", "lora"), default="lora")
    args = parser.parse_args(argv)

    if args.local_only:
        # Must precede imports inside train/preflight functions; setting this
        # later does not stop PEFT/Hugging Face Hub adapter probes.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.stage == "audit-data":
        result = audit_data(
            candidates=args.candidates, train=args.train, folds_path=args.folds,
            v3_dir=args.v3, preprocessing_dir=args.preprocessing,
            output_dir=args.results_root / "audit",
            expected_label_accounting=EXPECTED_LABEL_ACCOUNTING,
        )
    elif args.stage == "build-metadata":
        result = build_metadata_sidecar(v3_dir=args.v3, preprocessing_dir=args.preprocessing, output_dir=args.cache_root / "metadata")
    elif args.stage == "build-capsules":
        if not args.capsules:
            raise SystemExit("--capsules is required")
        result = enrich_capsules(
            capsules=args.capsules, metadata_dir=args.metadata_dir or args.cache_root / "metadata",
            v3_dir=args.v3, output_dir=args.output or args.cache_root / "capsules" / "manual",
            frozen_candidates=args.candidates, max_queries=args.max_queries,
        )
    elif args.stage == "preflight-models":
        result = preflight_models(output_dir=args.results_root / "preflight", device=args.device, local_only=args.local_only)
    elif args.stage == "score-worker":
        if not all((args.model, args.capsules, args.output, args.qids_json)):
            raise SystemExit("score worker arguments missing")
        result = score_capsules(
            spec=MODEL_BY_KEY[args.model], capsules=args.capsules, output_dir=args.output,
            qids=_qid_set(args.qids_json), device=args.device, local_only=args.local_only,
            identity_variant=CAPSULE_CONFIGS[args.capsule_variant]["identity_variant"], aggregation="max",
            max_length=COMMON_MAX_LENGTH, resume=args.resume, capsule_variant=args.capsule_variant,
        )
        atomic_json(args.output / "worker_result.json", result)
    elif args.stage == "train-worker":
        if not all((args.model, args.capsules, args.output, args.qids_json)):
            raise SystemExit("train worker arguments missing")
        answers, _ = canonical_answers(args.train, args.preprocessing / "exclusions.json", args.preprocessing / "train_label_impact.jsonl")
        train_function = train_full if args.training_mode == "full" else train_lora
        result = train_function(
            spec=MODEL_BY_KEY[args.model], capsules=args.capsules, train_qids=_qid_set(args.qids_json),
            answers=answers, output_dir=args.output, device=args.device, local_only=args.local_only,
            precision=args.precision, capsule_variant=args.capsule_variant, resume=args.resume,
        )
        atomic_json(args.output / "worker_result.json", result)
    elif args.stage == "score-lora-worker":
        if not all((args.model, args.capsules, args.output, args.qids_json, args.adapter)):
            raise SystemExit("LoRA score worker arguments missing")
        result = score_lora_capsules(
            spec=MODEL_BY_KEY[args.model], capsules=args.capsules, adapter=args.adapter,
            output_dir=args.output, qids=_qid_set(args.qids_json), device=args.device,
            local_only=args.local_only, identity_variant=CAPSULE_CONFIGS[args.capsule_variant]["identity_variant"],
            capsule_variant=args.capsule_variant,
        )
        atomic_json(args.output / "worker_result.json", result)
    elif args.stage == "score-full-worker":
        if not all((args.model, args.capsules, args.output, args.qids_json, args.model_path)):
            raise SystemExit("full-model score worker arguments missing")
        result = score_full_model(
            spec=MODEL_BY_KEY[args.model], capsules=args.capsules, model_path=args.model_path,
            output_dir=args.output, qids=_qid_set(args.qids_json), device=args.device,
            local_only=args.local_only, precision=args.precision,
            capsule_variant=args.capsule_variant,
        )
        atomic_json(args.output / "worker_result.json", result)
    elif args.stage == "analyze-errors":
        if not args.scores:
            raise SystemExit("--scores is required")
        answers, _ = canonical_answers(args.train, args.preprocessing / "exclusions.json", args.preprocessing / "train_label_impact.jsonl")
        result = error_analysis(args.scores, answers, args.output or args.results_root / "error_analysis.json")
    elif args.stage == "report":
        answers, _ = canonical_answers(args.train, args.preprocessing / "exclusions.json", args.preprocessing / "train_label_impact.jsonl")
        result = final_report(results_root=args.results_root, answers=answers, folds=load_folds(args.folds))
    else:
        result = overnight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
