"""Regex Symbolic Law Entity Matcher for EXP-014."""

from __future__ import annotations

import re
from typing import Any, Mapping


# Regex pattern to extract Vietnamese legal document numbers like 144/2021/NĐ-CP, 58/2020/TT-BCA, 100/2019/NĐ-CP, etc.
LAW_NUM_PATTERN = re.compile(
    r"\b(\d{1,4}/\d{4}/(?:NĐ-CP|TT-[A-Z0-9]+|QĐ-[A-Z0-9]+|QH\d+|NQ-[A-Z0-9]+|VBHN-[A-Z0-9]+))\b",
    re.IGNORECASE,
)

# Pattern for named laws e.g. "Luật Đất đai 2024", "Bộ luật Hình sự 2015"
LAW_NAME_PATTERN = re.compile(
    r"\b((?:Bộ\s+luật|Luật)\s+[A-ZÀ-Ỵa-zà-ỹ0-9\s]{3,35}(?:\s+năm\s+\d{4}|\s+\d{4})?)\b",
    re.IGNORECASE,
)


def extract_law_entities(query: str) -> dict[str, list[str]]:
    """Extract explicit legal numbers and named laws from a query string."""
    numbers = [match.group(1).strip() for match in LAW_NUM_PATTERN.finditer(query)]
    law_names = [match.group(1).strip() for match in LAW_NAME_PATTERN.finditer(query)]
    return {"numbers": numbers, "law_names": law_names}


def normalize_entity_str(text: str) -> str:
    """Normalize legal entity string for exact/fuzzy substring matching."""
    s = text.lower().replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_document_entities(query: str, doc_metadata: Mapping[str, Any]) -> dict[str, float]:
    """Check whether a document's metadata (document_label, name, link) matches entities in query."""
    entities = extract_law_entities(query)
    doc_label = str(doc_metadata.get("document_label", ""))
    doc_name = str(doc_metadata.get("name", ""))
    norm_label = normalize_entity_str(doc_label)
    norm_name = normalize_entity_str(doc_name)
    
    exact_match = 0.0
    num_matches = 0
    for num in entities["numbers"]:
        norm_num = normalize_entity_str(num)
        if norm_num and (norm_num in norm_label or norm_num in norm_name):
            exact_match = 1.0
            num_matches += 1
            
    name_match = 0.0
    for law in entities["law_names"]:
        norm_law = normalize_entity_str(law)
        if len(norm_law) > 5 and (norm_law in norm_label or norm_law in norm_name):
            name_match = 1.0

    return {
        "exact_match": exact_match,
        "name_match": name_match,
        "match_count": float(num_matches),
    }
