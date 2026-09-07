"""Extended Statutory Hierarchy & Multi-View Features (15D) for Gemini LegalIR (160D).

Hypothesis H67:
Adding 15 clean, label-free features capturing:
1. Primary law status & parent law linkage to top consensus decrees
2. Preamble citations
3. Explicit individual multi-view retriever reciprocal ranks (Jina late interaction, LAL, E5, BM25, Trigram)
4. Issuing agency consensus & distractor penalization
enables GBDT rankers to natively score primary laws and relevant evidence without heuristic swaps.

Zero label leakage: Depends strictly on public document labels, preambles, query text, and frozen source rankings.
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
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
DOC_PREAMBLES_PATH = ROOT / "cache/gemini/doc_preambles.json"

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


class ExtendedFeatureExtractor:
    """Extracts 15 additional statutory hierarchy and multi-view features."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = EVIDENCE_DB

        self.doc_labels: dict[str, str] = {}
        self.doc_type: dict[str, str] = {}
        self.doc_agency: dict[str, str] = {}
        self.doc_law_subject: dict[str, str] = {}
        self.doc_statute_tokens: dict[str, str] = {}
        self.doc_preambles: dict[str, set[str]] = {}

        if DOC_PREAMBLES_PATH.exists():
            try:
                raw_p = json.loads(DOC_PREAMBLES_PATH.read_text(encoding="utf-8"))
                for d, item in raw_p.items():
                    toks = set(item.get("nghi_dinh", [])) | set(item.get("luat", []))
                    self.doc_preambles[str(d)] = toks
            except Exception:
                pass

        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for doc, payload in db.execute("SELECT doc, payload FROM documents"):
            meta = json.loads(payload)
            lbl = (meta.get("document_label") or meta.get("retrieval_name") or meta.get("name") or "").lower()
            lbl_clean = re.sub(r"[\s/_\-]+", " ", lbl).strip()
            d_id = str(doc)
            self.doc_labels[d_id] = lbl_clean

            # doc type
            if lbl_clean.startswith("bo luat "):
                self.doc_type[d_id] = "bo_luat"
            elif lbl_clean.startswith("luat "):
                self.doc_type[d_id] = "luat"
            elif lbl_clean.startswith("nghi dinh "):
                self.doc_type[d_id] = "nghi_dinh"
            elif lbl_clean.startswith("thong tu "):
                self.doc_type[d_id] = "thong_tu"
            elif lbl_clean.startswith("quyet dinh "):
                self.doc_type[d_id] = "quyet_dinh"
            elif lbl_clean.startswith("cong van "):
                self.doc_type[d_id] = "cong_van"
            else:
                self.doc_type[d_id] = "other"

            # agency
            m_ag = re.search(r"\b(?:tt|nd|qd|cv|hd)\s+([a-z]+(?:\s+[a-z]+)?)\b", lbl_clean)
            self.doc_agency[d_id] = m_ag.group(1).strip() if m_ag else ""

            # statute token: number + year (e.g. "126 2020")
            m_num_yr = re.search(r"\b(\d+)\s+(20\d{2}|19\d{2})\b", lbl_clean)
            if m_num_yr:
                self.doc_statute_tokens[d_id] = f"{m_num_yr.group(1)} {m_num_yr.group(2)}"
            else:
                self.doc_statute_tokens[d_id] = ""

            # law subject
            if self.doc_type[d_id] in ("luat", "bo_luat"):
                m_sub = re.match(
                    r"(?:luat|bo luat)\s+(?:so\s+\d+/\d+/[a-z0-9]+\s+)?([a-z\s]+?)(?:\s+nam|\s+so|\s+\d{4}|$)",
                    lbl_clean,
                )
                if m_sub:
                    sub = m_sub.group(1).strip()
                    sub_words = [w for w in sub.split() if w not in STOPWORDS and len(w) > 1]
                    if len(sub_words) >= 2:
                        self.doc_law_subject[d_id] = " ".join(sub_words)
        db.close()

    def extract_single(
        self,
        q_text: str,
        doc_id: str,
        source_ranks: dict[str, int],
        top_consensus_docs: list[str],
    ) -> list[float]:
        """Extract 15 features for a single (query, document) pair."""
        top1 = top_consensus_docs[0] if len(top_consensus_docs) > 0 else ""
        top2 = top_consensus_docs[1] if len(top_consensus_docs) > 1 else ""
        top3 = top_consensus_docs[2] if len(top_consensus_docs) > 2 else ""

        lbl1 = self.doc_labels.get(top1, "")
        lbl2 = self.doc_labels.get(top2, "")
        lbl3 = self.doc_labels.get(top3, "")

        preambles1 = self.doc_preambles.get(top1, set())
        preambles2 = self.doc_preambles.get(top2, set())

        ag1 = self.doc_agency.get(top1, "")
        q_norm = strip_accents(q_text).lower()

        d_type = self.doc_type.get(doc_id, "other")
        d_tok = self.doc_statute_tokens.get(doc_id, "")
        d_law_sub = self.doc_law_subject.get(doc_id, "")
        d_ag = self.doc_agency.get(doc_id, "")

        # 1. is_primary_law
        f_is_primary = 1.0 if d_type in ("luat", "bo_luat") else 0.0

        # 2-4. parent law of top1..3
        f_parent_top1 = 1.0 if (f_is_primary and d_law_sub and (d_law_sub in lbl1)) else 0.0
        f_parent_top2 = 1.0 if (f_is_primary and d_law_sub and (d_law_sub in lbl2)) else 0.0
        f_parent_top3 = 1.0 if (f_is_primary and d_law_sub and (d_law_sub in lbl3)) else 0.0

        # 5-6. preamble citations
        f_preamble_top1 = 1.0 if (d_tok and d_tok in preambles1) else 0.0
        f_preamble_top2 = 1.0 if (d_tok and d_tok in preambles2) else 0.0

        # 7-11. explicit source reciprocal ranks
        f_rr_jina = 1.0 / source_ranks.get("jina", 500) if source_ranks.get("jina", 500) < 500 else 0.0
        f_rr_lal = 1.0 / source_ranks.get("lal", 500) if source_ranks.get("lal", 500) < 500 else 0.0
        f_rr_e5 = 1.0 / source_ranks.get("e5", 500) if source_ranks.get("e5", 500) < 500 else 0.0
        f_rr_bm25 = 1.0 / source_ranks.get("bm25", 500) if source_ranks.get("bm25", 500) < 500 else 0.0
        f_rr_tri = 1.0 / source_ranks.get("trigram", 500) if source_ranks.get("trigram", 500) < 500 else 0.0

        # 12. agency match
        f_ag_match = 1.0 if (d_ag and ag1 and d_ag == ag1) else 0.0

        # 13. guideline doc indicator
        f_is_guideline = 1.0 if d_type in ("thong_tu", "quyet_dinh", "cong_van") else 0.0

        # 14. law subject in query text
        f_law_in_q = 1.0 if (f_is_primary and d_law_sub and d_law_sub in q_norm) else 0.0

        # 15. top1 is guideline and candidate is primary law
        f_top1_guide_doc_law = 1.0 if (f_is_primary and self.doc_type.get(top1, "") in ("nghi_dinh", "thong_tu")) else 0.0

        return [
            f_is_primary, f_parent_top1, f_parent_top2, f_parent_top3,
            f_preamble_top1, f_preamble_top2,
            f_rr_jina, f_rr_lal, f_rr_e5, f_rr_bm25, f_rr_tri,
            f_ag_match, f_is_guideline, f_law_in_q, f_top1_guide_doc_law,
        ]

    def extract_block(
        self,
        questions: Mapping[str, str],
        qids: Sequence[str],
        docs_by_query: Sequence[Sequence[str]],
        source_maps: Mapping[str, Mapping[str, Mapping[str, int]]],
    ) -> np.ndarray:
        """Vectorized extraction of 15D features for a batch of queries."""
        rows = []
        for q, docs in zip(qids, docs_by_query):
            q_text = questions.get(q, "")
            # Compute top consensus docs from available sources
            scores: dict[str, float] = {}
            for s in ("e5", "lal", "bm25", "trigram", "jina"):
                s_map = source_maps.get(s, {}).get(q, {})
                for d, r in s_map.items():
                    scores[d] = scores.get(d, 0.0) + 1.0 / (60.0 + r)
            top_consensus = sorted(scores.keys(), key=lambda d: -scores[d])[:5]

            for d in docs:
                sr = {
                    s: source_maps.get(s, {}).get(q, {}).get(d, 500)
                    for s in ("e5", "lal", "bm25", "trigram", "jina")
                }
                feats = self.extract_single(q_text, d, sr, top_consensus)
                rows.append(feats)

        matrix = np.asarray(rows, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite values detected in ExtendedFeatureExtractor matrix")
        return matrix
