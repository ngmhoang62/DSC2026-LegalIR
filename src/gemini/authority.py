"""Statutory Authority and Legal Hierarchy Features for Gemini namespace.

Features (12D):
0: doc_hierarchy_level (1..8)
1: is_law (bool: Constitution/Law/Code)
2: is_decree (bool: Government Decree)
3: is_circular (bool: Ministry Circular)
4: is_decision (bool: Prime Minister / Ministry / Court Decision)
5: query_mentions_law (bool)
6: query_mentions_decree (bool)
7: query_mentions_circular (bool)
8: query_mentions_decision (bool)
9: authority_exact_match (bool: query authority == doc authority)
10: authority_mismatch_penalty (bool: query asks for decree/circular, doc is umbrella law)
11: title_term_specificity (float: Jaccard overlap of non-generic terms)

Zero label leakage: Depends strictly on public document labels and input query text.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

HIERARCHY_LEVELS = {
    "hien phap": 1,
    "bo luat": 1,
    "luat": 1,
    "phap lenh": 2,
    "nghi quyet": 2,
    "nghi dinh": 3,
    "quyet dinh": 4,
    "quy dinh": 4,
    "thong tu": 5,
    "thong tu lien tich": 5,
    "cong van": 6,
    "qcvn": 7,
    "tcvn": 7,
}

STOPWORDS = frozenset([
    "quy", "dinh", "phap", "luat", "nam", "ve", "cua", "va", "cho",
    "la", "co", "duoc", "trong", "theo", "cac", "nhung", "nguoi", "so", "thi", "nhu", "the", "nao",
])


class AuthorityExtractor:
    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
        self.doc_labels: dict[str, str] = {}
        self.doc_hierarchy: dict[str, int] = {}
        self.doc_words: dict[str, set[str]] = {}
        
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for doc, payload in db.execute("SELECT doc, payload FROM documents"):
            meta = json.loads(payload)
            lbl = (meta.get("document_label") or meta.get("retrieval_name") or meta.get("name") or "").lower()
            self.doc_labels[str(doc)] = lbl
            
            level = 8
            for pfx, lvl in HIERARCHY_LEVELS.items():
                if lbl.startswith(pfx):
                    level = lvl
                    break
            self.doc_hierarchy[str(doc)] = level
            words = set(re.findall(r"\w+", lbl)) - STOPWORDS
            self.doc_words[str(doc)] = words
        db.close()

    def extract_single(self, q_text: str, doc_id: str) -> list[float]:
        q_lower = q_text.lower()
        level = self.doc_hierarchy.get(str(doc_id), 8)
        
        is_law = float(level == 1)
        is_decree = float(level == 3)
        is_circular = float(level == 5)
        is_decision = float(level == 4)
        
        q_has_law = float("luật" in q_lower or "bộ luật" in q_lower)
        q_has_decree = float("nghị định" in q_lower)
        q_has_circular = float("thông tư" in q_lower)
        q_has_decision = float("quyết định" in q_lower)
        
        exact_match = float(
            (q_has_law and is_law) or
            (q_has_decree and is_decree) or
            (q_has_circular and is_circular) or
            (q_has_decision and is_decision)
        )
        
        mismatch = float(
            (q_has_decree and not is_decree and is_law) or
            (q_has_circular and not is_circular and is_law)
        )
        
        q_words = set(re.findall(r"\w+", q_lower)) - STOPWORDS
        lbl_words = self.doc_words.get(str(doc_id), set())
        overlap = len(q_words & lbl_words) / max(1, len(lbl_words)) if lbl_words else 0.0
        
        return [
            float(level),
            is_law,
            is_decree,
            is_circular,
            is_decision,
            q_has_law,
            q_has_decree,
            q_has_circular,
            q_has_decision,
            exact_match,
            mismatch,
            float(overlap),
        ]

    def extract_block(self, questions: Mapping[str, str], qids: Sequence[str], docs_list: Sequence[Sequence[str]]) -> np.ndarray:
        rows = []
        for q, docs in zip(qids, docs_list):
            q_text = questions[q]
            for d in docs:
                rows.append(self.extract_single(q_text, d))
        return np.asarray(rows, dtype=np.float32)
