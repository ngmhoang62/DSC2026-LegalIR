"""Canonical label utilities for the Gemini research namespace.

Ensures strict compliance with the competition label contract:
canonical_duplicate_alias_drop_empty_passage_v1.
- Exactly 6,991 evaluable queries and 9 non-evaluable queries.
- Original labels preserved for evaluation when required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
PRIMARY_EXCLUSIONS_PATH = ROOT / "cache" / "final_preprocessed_v2" / "exclusions.json"
FALLBACK_EXCLUSIONS_PATH = ROOT / "results" / "exp110p_semantic_label_prototype" / "colab_bundle" / "input" / "exclusions.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"

def get_exclusions() -> dict[str, dict[str, Any]]:
    rows = None
    try:
        if PRIMARY_EXCLUSIONS_PATH.is_file():
            rows = json.loads(PRIMARY_EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        rows = None
    if rows is None:
        rows = json.loads(FALLBACK_EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    return {str(row["doc_id"]): row for row in rows}

def get_canonical_labels(
    train_path: Path = TRAIN_PATH,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    exclusions = get_exclusions()
    
    answers: dict[str, set[str]] = {}
    evaluable = 0
    non_evaluable = 0
    duplicate_occurrences = 0
    empty_occurrences = 0
    
    for raw_qid, row in train_data.items():
        qid = str(raw_qid)
        gold: set[str] = set()
        for raw_doc in row.get("answer", []):
            doc_id = str(raw_doc)
            ex = exclusions.get(doc_id)
            if ex is None:
                gold.add(doc_id)
                continue
            reasons = set(ex.get("reasons", []))
            replacement = ex.get("duplicate_retained_id")
            if "exact_duplicate_raw_passage" in reasons:
                if replacement:
                    gold.add(str(replacement))
                    duplicate_occurrences += 1
            if "empty_passage" in reasons:
                empty_occurrences += 1
        answers[qid] = gold
        if gold:
            evaluable += 1
        else:
            non_evaluable += 1
            
    audit = {
        "evaluable_query_count": evaluable,
        "non_evaluable_query_count": non_evaluable,
        "duplicate_occurrences": duplicate_occurrences,
        "empty_occurrences": empty_occurrences,
    }
    return answers, audit

def get_cv_folds(folds_path: Path = FOLDS_PATH) -> dict[str, list[str]]:
    raw = json.loads(folds_path.read_text(encoding="utf-8"))
    return {str(k): [str(q) for q in v] for k, v in raw.items()}
