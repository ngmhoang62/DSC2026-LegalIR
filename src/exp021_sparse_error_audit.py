"""Describe where a sparse configuration succeeds and fails; it does not tune it."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from exp012b_bm25 import default_segmenter
from exp012b_core import atomic_json, read_jsonl


LEGAL_IDENTIFIER = re.compile(
    r"\b(?:điều|khoản|mục|chương)\s*\d+[\w.-]*|\b\d{1,4}/\d{2,4}/[\w-]+\b",
    re.IGNORECASE,
)
KS = (1, 5, 10, 20, 50, 100)


def _token_bin(query: str) -> str:
    size = len(default_segmenter(query).split())
    if size <= 5: return "1-5"
    if size <= 10: return "6-10"
    if size <= 20: return "11-20"
    if size <= 35: return "21-35"
    return "36+"


def _summary(rows: Iterable[tuple[list[str], set[str]]]) -> dict[str, Any]:
    values = list(rows)
    metrics = {
        f"recall@{k}": sum(len(set(predicted[:k]) & gold) / len(gold) for predicted, gold in values) / len(values)
        for k in KS
    } if values else {f"recall@{k}": 0.0 for k in KS}
    return {"queries": len(values), **metrics}


def audit_sparse_errors(*, rankings_path: Path, train_path: Path, v3_dir: Path, output_dir: Path) -> dict[str, Any]:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    docs = {str(row["doc_id"]): row for row in read_jsonl(v3_dir / "documents.jsonl")}
    groups: dict[str, dict[str, list[tuple[list[str], set[str]]]]] = defaultdict(lambda: defaultdict(list))
    rank_buckets: dict[str, int] = defaultdict(int)
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(rankings_path):
        qid = str(row["qid"]); seen.add(qid)
        gold = set(map(str, train[qid]["answer"]))
        predicted = [str(item["doc_id"]) for item in row["documents"]]
        ranks = [position for position, doc_id in enumerate(predicted, start=1) if doc_id in gold]
        first = min(ranks) if ranks else 101
        bucket = "1" if first == 1 else "2-5" if first <= 5 else "6-10" if first <= 10 else "11-20" if first <= 20 else "21-50" if first <= 50 else "51-100" if first <= 100 else ">100"
        rank_buckets[bucket] += 1
        groups["query_tokens"][_token_bin(str(row["query"]))].append((predicted, gold))
        groups["legal_identifier"]["yes" if LEGAL_IDENTIFIER.search(str(row["query"])) else "no"].append((predicted, gold))
        for doc_id in gold:
            document = docs.get(doc_id, {})
            groups["gold_parse_mode"][str(document.get("parse_mode", "missing"))].append((predicted, {doc_id}))
            groups["gold_scope"]["has_scope" if document.get("scope_node_ids") else "no_scope"].append((predicted, {doc_id}))
        if first > 50 and len(examples) < 50:
            examples.append({"qid": qid, "query": row["query"], "gold_doc_ids": sorted(gold), "first_gold_rank": None if first == 101 else first, "top5": predicted[:5]})
    if seen != set(map(str, train)):
        raise RuntimeError("rankings/train query set mismatch")
    report = {
        "queries": len(seen), "first_gold_rank_buckets": dict(rank_buckets),
        "slices": {name: {label: _summary(values) for label, values in sorted(parts.items())} for name, parts in groups.items()},
        "top50_miss_examples": examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "error_audit.json", report)
    return report
