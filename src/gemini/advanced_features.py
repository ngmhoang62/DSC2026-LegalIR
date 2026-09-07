"""Advanced Statutory & Cross-Source Features for Gemini LegalIR (14D).

Features (14D):
0: is_amendment (bool: title contains 'sua doi' or 'bo sung')
1: is_official_letter_penalty (bool: doc is 'cong van' but query does not ask for 'cong van')
2: query_statute_number_match (bool: query mentions a statute number that appears in doc label)
3: query_statute_year_match (bool: query mentions a 4-digit year that appears in doc label)
4: exact_title_phrase_len (float: length of longest consecutive matching word sequence, clipped to 10)
5: title_jaccard_overlap (float: word intersection over union with query, non-stopwords)
6: reciprocal_rank_fusion_60 (float: sum(1.0 / (60 + r_s)) across available sources)
7: min_source_rank_recip (float: 1.0 / min(source ranks))
8: source_rank_spread (float: standard deviation of ranks across sources)
9: source_present_count (float: count of sources containing this doc, 1..5)
10: is_amendment_of_top1 (bool: doc is an amendment of the #1 ranked document)
11: is_amendment_of_top2 (bool: doc is an amendment of the #2 ranked document)
12: is_base_of_top1 (bool: doc is the base statute amended by the #1 ranked document)
13: is_base_of_top2 (bool: doc is the base statute amended by the #2 ranked document)

Zero label leakage: Depends strictly on public document labels, query text, and frozen source rankings.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

STOPWORDS = frozenset([
    "quy", "dinh", "phap", "luat", "nam", "ve", "cua", "va", "cho",
    "la", "co", "duoc", "trong", "theo", "cac", "nhung", "nguoi", "so",
    "thi", "nhu", "the", "nao", "muc", "bao", "nhieu", "khi", "ai", "dau",
    "phai", "mot", "tai", "truong", "hop", "doi", "voi", "tu", "den",
])


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.replace("đ", "d").replace("Đ", "D")


class AdvancedFeatureExtractor:
    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
        
        self.doc_labels: dict[str, str] = {}
        self.doc_words: dict[str, set[str]] = {}
        self.doc_numbers: dict[str, set[str]] = {}
        self.doc_years: dict[str, set[str]] = {}
        self.doc_is_amendment: dict[str, bool] = {}
        self.doc_is_cong_van: dict[str, bool] = {}
        
        # Statutory Kinship Maps (from labels)
        self.amendment_to_bases: dict[str, set[str]] = {}
        self.base_to_amendments: dict[str, set[str]] = {}

        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        raw_labels: dict[str, str] = {}
        for doc, payload in db.execute("SELECT doc, payload FROM documents"):
            meta = json.loads(payload)
            lbl = (meta.get("document_label") or meta.get("retrieval_name") or meta.get("name") or "").lower()
            lbl_clean = re.sub(r"[\s/_\-]+", " ", lbl).strip()
            raw_labels[str(doc)] = lbl_clean
        db.close()

        for doc_id, lbl in raw_labels.items():
            self.doc_labels[doc_id] = lbl
            words = set(re.findall(r"\w+", lbl)) - STOPWORDS
            self.doc_words[doc_id] = words
            
            nums = set(re.findall(r"\b(\d+)\b", lbl))
            self.doc_numbers[doc_id] = nums
            
            years = set(re.findall(r"\b(20\d{2}|19\d{2})\b", lbl))
            self.doc_years[doc_id] = years
            
            is_amend = bool("sua doi" in lbl or "bo sung" in lbl)
            self.doc_is_amendment[doc_id] = is_amend
            self.doc_is_cong_van[doc_id] = lbl.startswith("cong van")

        # Build kinship relations between documents from labels
        for doc_id, lbl in raw_labels.items():
            if self.doc_is_amendment[doc_id]:
                # Extract cited base numbers + years
                pairs = re.findall(r"\b(\d+)\s+(20\d{2}|19\d{2})\b", lbl)
                for num, yr in pairs:
                    token = f"{num} {yr}"
                    for other_id, other_lbl in raw_labels.items():
                        if other_id != doc_id and not self.doc_is_amendment[other_id]:
                            if token in other_lbl:
                                self.amendment_to_bases.setdefault(doc_id, set()).add(other_id)
                                self.base_to_amendments.setdefault(other_id, set()).add(doc_id)

    def extract_single(
        self,
        q_text: str,
        doc_id: str,
        source_ranks: dict[str, int],
        top_consensus_docs: list[str],
    ) -> list[float]:
        """Extract 14 features for a single (query, document) pair."""
        q_norm = strip_accents(q_text).lower()
        q_words = set(re.findall(r"\w+", q_norm)) - STOPWORDS
        q_nums = set(re.findall(r"\b(\d+)\b", q_norm))
        q_years = set(re.findall(r"\b(20\d{2}|19\d{2})\b", q_norm))
        
        lbl = self.doc_labels.get(doc_id, "")
        d_words = self.doc_words.get(doc_id, set())
        d_nums = self.doc_numbers.get(doc_id, set())
        d_years = self.doc_years.get(doc_id, set())
        
        # 0: is_amendment
        f0 = float(self.doc_is_amendment.get(doc_id, False))
        
        # 1: is_official_letter_penalty
        f1 = float(self.doc_is_cong_van.get(doc_id, False) and "cong van" not in q_norm)
        
        # 2: query_statute_number_match
        f2 = float(bool(q_nums & d_nums))
        
        # 3: query_statute_year_match
        f3 = float(bool(q_years & d_years))
        
        # 4: exact_title_phrase_len (longest consecutive matching words)
        q_tokens = re.findall(r"\w+", q_norm)
        lbl_tokens = re.findall(r"\w+", lbl)
        max_phrase = 0
        if q_tokens and lbl_tokens:
            lbl_joined = " " + " ".join(lbl_tokens) + " "
            for length in range(min(10, len(q_tokens)), 1, -1):
                found = False
                for start in range(len(q_tokens) - length + 1):
                    sub = " " + " ".join(q_tokens[start:start + length]) + " "
                    if sub in lbl_joined:
                        max_phrase = length
                        found = True
                        break
                if found:
                    break
        f4 = float(max_phrase)
        
        # 5: title_jaccard_overlap
        union_len = len(q_words | d_words)
        f5 = float(len(q_words & d_words) / union_len) if union_len > 0 else 0.0
        
        # Source ranks features
        ranks = [r for r in source_ranks.values() if r < 500]
        f6 = sum(1.0 / (60.0 + r) for r in ranks)
        f7 = (1.0 / min(ranks)) if ranks else 0.0
        f8 = float(np.std(ranks)) if len(ranks) > 1 else 0.0
        f9 = float(len(ranks))
        
        # Kinship features with top consensus docs
        top1 = top_consensus_docs[0] if len(top_consensus_docs) > 0 else ""
        top2 = top_consensus_docs[1] if len(top_consensus_docs) > 1 else ""
        
        f10 = float(doc_id in self.base_to_amendments.get(top1, set()))
        f11 = float(doc_id in self.base_to_amendments.get(top2, set()))
        f12 = float(doc_id in self.amendment_to_bases.get(top1, set()))
        f13 = float(doc_id in self.amendment_to_bases.get(top2, set()))
        
        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]

    def extract_block(
        self,
        questions: Mapping[str, str],
        qids: Sequence[str],
        docs_by_query: Sequence[Sequence[str]],
        source_maps: Mapping[str, Mapping[str, Mapping[str, int]]],
    ) -> np.ndarray:
        """Vectorized extraction of 14D features for a batch of queries."""
        rows = []
        for q, docs in zip(qids, docs_by_query):
            q_text = questions.get(q, "")
            
            # Find top 2 consensus documents across E5, LAL, BM25
            doc_scores: dict[str, float] = {}
            for s in ("e5", "lal", "bm25"):
                s_map = source_maps.get(s, {}).get(q, {})
                for d, r in s_map.items():
                    if r <= 20:
                        doc_scores[d] = doc_scores.get(d, 0.0) + 1.0 / (10.0 + r)
            top_consensus = sorted(doc_scores.keys(), key=lambda d: -doc_scores[d])[:2]
            
            for d in docs:
                s_ranks = {
                    s: source_maps.get(s, {}).get(q, {}).get(d, 501)
                    for s in ("e5", "lal", "bm25", "trigram", "jina")
                }
                rows.append(self.extract_single(q_text, d, s_ranks, top_consensus))
                
        return np.asarray(rows, dtype=np.float32)
