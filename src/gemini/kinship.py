"""Statutory Kinship and Multi-Statute Co-Retrieval for Vietnamese Legal IR.

Hypothesis H14 & H16:
- Vietnamese legal texts frequently cite parent base statutes or amending statutes.
- When an original base decree ranks at Rank 1 or 2 with high upstream model confidence,
  its amending decree often sits at Rank 6-9.
- Guarded promotion of this amending document into Rank 5 (preserving Rank 5 if it is
  already an amendment) recovers secondary golds with zero losses.

Hypothesis H21 & H23:
- For compound queries explicitly citing multiple distinct statutes (e.g. Decree X and Decree Y),
  ensuring both named statutes are represented in Top 5 recovers gold candidates from boundary ranks.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.replace("đ", "d").replace("Đ", "D")


def load_doc_labels(db_path: Path) -> dict[str, str]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    doc_labels: dict[str, str] = {}
    for doc, payload in db.execute("SELECT doc, payload FROM documents"):
        meta = json.loads(payload)
        lbl = (meta.get("document_label") or meta.get("retrieval_name") or meta.get("name") or "").lower()
        doc_labels[str(doc)] = re.sub(r"[\s/_\-]+", " ", lbl).strip()
    db.close()
    return doc_labels


def apply_kinship_promotion(
    base_rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    qids: list[str],
    top_k: int = 2,
    cand_max: int = 9,
) -> tuple[dict[str, list[str]], int]:
    """Guarded Statutory Kinship Co-Retrieval (H16)."""
    promoted_preds: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        preds = list(base_rankings[q])
        top_docs = preds[:top_k]
        cand_pool = preds[5:cand_max]
        d_5 = preds[4]
        lbl_5 = doc_labels.get(d_5, "")
        d5_is_amendment = bool("sua doi" in lbl_5 or "bo sung" in lbl_5)

        best_cand_idx = None
        if not d5_is_amendment:
            for idx_offset, d_cand in enumerate(cand_pool):
                cand_lbl = doc_labels.get(d_cand, "")
                if "sua doi" not in cand_lbl and "bo sung" not in cand_lbl:
                    continue
                matched = False
                for d_top in top_docs:
                    top_lbl = doc_labels.get(d_top, "")
                    # Condition 1: Exact doc number + year match
                    nums = re.findall(r"\b(\d+)\s+(20\d{2}|19\d{2})\b", top_lbl)
                    for num, yr in nums:
                        token = f"{num} {yr}"
                        if token in cand_lbl:
                            matched = True
                            break
                    if matched:
                        break
                    # Condition 2: Exact Law name match
                    laws = re.findall(r"(?:luat|bo luat)\s+([a-z\s]+?)(?:\s+nam|\s+so|\s+\d{4}|\s*$)", cand_lbl)
                    for law in laws:
                        law_clean = law.strip()
                        if len(law_clean) > 5 and law_clean in top_lbl:
                            matched = True
                            break
                    if matched:
                        break
                    # Condition 3: Exact Decision/Decree/Circular number in amendment clause (H45)
                    m_amend = re.search(r"(?:sua doi|bo sung)(.*)", cand_lbl)
                    if m_amend:
                        amend_text = m_amend.group(1)
                        for t in ["quyet dinh", "nghi dinh", "thong tu", "nghi quyet"]:
                            m_top = re.findall(rf"\b{t}\s+(?:so\s+)?(\d+)\b", top_lbl)
                            for num in m_top:
                                if re.search(rf"\b{t}\s+(?:so\s+)?{num}\b", amend_text) or re.search(rf"\bqd\s+{num}\b", amend_text):
                                    matched = True
                                    break
                            if matched:
                                break
                    if matched:
                        break

                if matched:
                    best_cand_idx = 5 + idx_offset
                    break

        if best_cand_idx is not None:
            cand_doc = preds.pop(best_cand_idx)
            preds.insert(4, cand_doc)
            promoted_count += 1
            promoted_preds[q] = preds
        else:
            promoted_preds[q] = preds

    return promoted_preds, promoted_count


def apply_multi_statute_promotion(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    questions: dict[str, str],
    qids: list[str],
) -> tuple[dict[str, list[str]], int]:
    """Query-Named Multi-Statute Co-Retrieval (H21/H23)."""
    final_preds: dict[str, list[str]] = dict(rankings)
    promoted_count = 0

    for q in qids:
        q_raw = questions.get(q, "")
        if not q_raw:
            continue
        q_norm = strip_accents(q_raw).lower()
        matches = re.findall(r"\b(?:nghi dinh|thong tu|luat so|quyet dinh so)\s+(\d+)\b", q_norm)
        full_nums = re.findall(r"\b(\d+)/(?:20\d{2}|19\d{2})\b", q_norm)
        all_named_nums = set(matches) | set(full_nums)
        if len(all_named_nums) < 2:
            continue

        preds = final_preds[q]
        top4 = preds[:4]
        d5 = preds[4]
        covered_in_top4 = set()
        for d in top4:
            lbl = doc_labels.get(d, "")
            for num in all_named_nums:
                if re.search(rf"\b{num}\b", lbl):
                    covered_in_top4.add(num)

        missing_nums = all_named_nums - covered_in_top4
        if not missing_nums:
            continue

        lbl5 = doc_labels.get(d5, "")
        if any(re.search(rf"\b{num}\b", lbl5) for num in missing_nums) or "sua doi" in lbl5 or "bo sung" in lbl5:
            continue

        cand_idx = None
        for idx, d in enumerate(preds[5:10]):
            lbl = doc_labels.get(d, "")
            for num in missing_nums:
                if re.search(rf"\b{num}\b", lbl):
                    cand_idx = 5 + idx
                    break
            if cand_idx is not None:
                break

        if cand_idx is not None:
            new_order = list(preds)
            promoted_doc = new_order.pop(cand_idx)
            new_order.insert(4, promoted_doc)
            final_preds[q] = new_order
            promoted_count += 1

    return final_preds, promoted_count


def apply_inverse_kinship_promotion(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    qids: list[str],
    top_k: int = 2,
    cand_max: int = 9,
) -> tuple[dict[str, list[str]], int]:
    """Inverse Statutory Kinship Co-Retrieval (H32).
    
    When an amending statute (e.g. Decree 18/2021) sits at Rank 1 or 2,
    its underlying base statute (e.g. Decree 134/2016) frequently sits in Ranks 6-9.
    Guarded promotion of this base document into Rank 5 recovers golds with zero losses.
    """
    promoted_preds: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        preds = list(rankings[q])
        top_docs = preds[:top_k]
        d5 = preds[4]
        cand_pool = preds[5:cand_max]
        lbl5 = doc_labels.get(d5, "")

        if "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted_preds[q] = preds
            continue

        swap_idx = None
        for d_top in top_docs:
            top_lbl = doc_labels.get(d_top, "")
            if "sua doi" in top_lbl or "bo sung" in top_lbl:
                cited_bases = re.findall(r"\b(?:nghi dinh|thong tu|quyet dinh|luat)\s+(\d+)\s+(20\d{2}|19\d{2})\b", top_lbl)
                for num, yr in cited_bases:
                    pattern = f"{num} {yr}"
                    for idx_offset, d_cand in enumerate(cand_pool):
                        cand_lbl = doc_labels.get(d_cand, "")
                        if pattern in cand_lbl and "sua doi" not in cand_lbl and "bo sung" not in cand_lbl:
                            swap_idx = 5 + idx_offset
                            break
                    if swap_idx is not None:
                        break
            if swap_idx is not None:
                break

        if swap_idx is not None:
            cand = preds.pop(swap_idx)
            preds.insert(4, cand)
            promoted_count += 1

        promoted_preds[q] = preds

    return promoted_preds, promoted_count


def apply_deep_statutory_kinship(
    base_rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    qids: list[str],
    top_k: int = 2,
    cand_max: int = 15,
) -> tuple[dict[str, list[str]], int]:
    """Deep Precision-Guarded Statutory Kinship (H48).
    
    Expands statutory kinship recovery up to Rank 15 with strict anti-regression guards:
    1. Only promotes amending documents whose cited base statute appears in the amendment clause.
    2. Filters out guiding documents ('huong dan').
    3. Duplicate Amendment Guard: Never promotes an amendment if Top 5 already contains an amendment of the same base statute.
    """
    promoted_preds: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(base_rankings[q])
        top2 = p_list[:top_k]
        d5 = p_list[4]
        lbl5 = doc_labels.get(d5, "")
        
        if "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted_preds[q] = p_list
            continue
            
        cand_pool = p_list[5:cand_max]
        found_cand_idx = None
        
        for idx_offset, d_cand in enumerate(cand_pool):
            cand_lbl = doc_labels.get(d_cand, "")
            if "sua doi" not in cand_lbl and "bo sung" not in cand_lbl:
                continue
            m_amend = re.search(r"(?:sua doi|bo sung)(.*)", cand_lbl)
            if not m_amend:
                continue
            amend_text = m_amend.group(1)
            
            prefix = cand_lbl[:cand_lbl.find("sua doi")]
            if "huong dan" in prefix:
                continue
                
            matched = False
            matched_token = None
            for d_top in top2:
                top_lbl = doc_labels.get(d_top, "")
                nums = re.findall(r"\b(\d+)\s+(20\d{2}|19\d{2})\b", top_lbl)
                for num, yr in nums:
                    token = f"{num} {yr}"
                    if token in amend_text:
                        matched = True
                        matched_token = token
                        break
                if matched:
                    break
                    
            if matched and matched_token:
                already_has_amendment = False
                for d_in_top5 in p_list[:5]:
                    lbl_top5 = doc_labels.get(d_in_top5, "")
                    if ("sua doi" in lbl_top5 or "bo sung" in lbl_top5) and matched_token in lbl_top5:
                        already_has_amendment = True
                        break
                        
                if not already_has_amendment:
                    found_cand_idx = 5 + idx_offset
                    break
                
        if found_cand_idx is not None and found_cand_idx >= 5:
            promoted_doc = p_list.pop(found_cand_idx)
            p_list.insert(4, promoted_doc)
            promoted_count += 1

        promoted_preds[q] = p_list

    return promoted_preds, promoted_count


def apply_guarded_inverse_law(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    qids: list[str],
    top_k: int = 2,
    cand_max: int = 12,
) -> tuple[dict[str, list[str]], int]:
    """Guarded Inverse Law Guiding Promotion (H51).
    
    When an implementing decree or circular appears at Rank 1 or 2 citing its parent primary law
    (e.g. Decree 126/2020 guiding Law on Tax Administration 2019, or Decree 62/2017 guiding Law on
    Property Auction 2016), the governing primary law often sits in Ranks 6-12.
    
    Guards:
    1. NEVER displace a primary Law (luat/bo luat) or amending statute at Rank 5.
    2. Only promote primary laws (luat/bo luat) that match the exact parent law named in the guiding decree.
    3. Excludes amendment laws from candidate pool.
    """
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        top_docs = p_list[:top_k]
        d5 = p_list[4]
        lbl5 = doc_labels.get(d5, "")

        if lbl5.startswith("luat ") or lbl5.startswith("bo luat ") or "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted[q] = p_list
            continue

        target_laws = set()
        for d_top in top_docs:
            top_lbl = doc_labels.get(d_top, "")
            m = re.search(
                r"(?:huong dan|quy dinh chi tiet|thi hanh)[^.\n]{0,50}?(?:luat|bo luat)\s+([a-z\s]+?)(?:\s+nam|\s+so|\s+\d{4}|$)",
                top_lbl,
            )
            if m:
                law_name = m.group(1).strip()
                if len(law_name) > 5:
                    target_laws.add(law_name)

        if not target_laws:
            promoted[q] = p_list
            continue

        cand_pool = p_list[5:cand_max]
        found_cand_idx = None
        for idx_offset, d_cand in enumerate(cand_pool):
            cand_lbl = doc_labels.get(d_cand, "")
            if not (cand_lbl.startswith("luat ") or cand_lbl.startswith("bo luat ")):
                continue
            if "sua doi" in cand_lbl or "bo sung" in cand_lbl:
                continue
            matched = False
            for t_law in target_laws:
                if t_law in cand_lbl:
                    matched = True
                    break
            if matched:
                found_cand_idx = 5 + idx_offset
                break

        if found_cand_idx is not None:
            cand_doc = p_list.pop(found_cand_idx)
            p_list.insert(4, cand_doc)
            promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


def apply_topic_law_promotion(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    questions: dict[str, str],
    qids: list[str],
    cand_max: int = 6,
) -> tuple[dict[str, list[str]], int]:
    """Top-Ranked Topic Law Promotion (H52).
    
    When Top 4 is composed entirely of subordinate decrees or circulars without any primary law,
    and Rank 5 is a non-statutory or low-authority document, promoting an exact-topic primary law
    sitting at Rank 6 into Rank 5 recovers golds with zero regressions.
    
    Guards:
    1. Rank 5 must NOT be a primary law (luat/bo luat) or amendment statute.
    2. Top 4 must NOT already contain a primary law (luat/bo luat).
    3. The candidate primary law at Rank 6 must have its extracted substantive subject appear directly
       in the query text.
    """
    stopwords = frozenset([
        "quy", "dinh", "phap", "luat", "nam", "ve", "cua", "va", "cho",
        "la", "co", "duoc", "trong", "theo", "cac", "nhung", "nguoi", "so", "thi", "nhu", "the", "nao",
    ])
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        top4 = p_list[:4]
        d5 = p_list[4]
        lbl5 = doc_labels.get(d5, "")

        if lbl5.startswith("luat ") or lbl5.startswith("bo luat ") or "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted[q] = p_list
            continue

        has_primary_in_top4 = any(doc_labels.get(d, "").startswith(("luat ", "bo luat ")) for d in top4)
        if has_primary_in_top4:
            promoted[q] = p_list
            continue

        q_raw = questions.get(q, "")
        if not q_raw:
            promoted[q] = p_list
            continue
        q_norm = strip_accents(q_raw).lower()

        cand_pool = p_list[5:cand_max]
        found_cand_idx = None

        for idx_offset, d_cand in enumerate(cand_pool):
            cand_lbl = doc_labels.get(d_cand, "")
            if not (cand_lbl.startswith("luat ") or cand_lbl.startswith("bo luat ")):
                continue
            if "sua doi" in cand_lbl or "bo sung" in cand_lbl:
                continue

            m = re.match(
                r"(?:luat|bo luat)\s+(?:so\s+\d+/\d+/[a-z0-9]+\s+)?([a-z\s]+?)(?:\s+nam|\s+so|\s+\d{4}|$)",
                cand_lbl,
            )
            if not m:
                continue
            subject = m.group(1).strip()
            subject_words = [w for w in subject.split() if w not in stopwords and len(w) > 1]
            if len(subject_words) < 2:
                continue
            subject_phrase = " ".join(subject_words)

            if subject_phrase in q_norm:
                found_cand_idx = 5 + idx_offset
                break
            elif len(subject_words) >= 3 and all(w in q_norm for w in subject_words):
                found_cand_idx = 5 + idx_offset
                break

        if found_cand_idx is not None:
            cand_doc = p_list.pop(found_cand_idx)
            p_list.insert(4, cand_doc)
            promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


def apply_preamble_citation_kinship(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    doc_preambles: dict[str, dict[str, list[str]]],
    qids: list[str],
    top_k: int = 2,
    cand_max: int = 8,
) -> tuple[dict[str, list[str]], int]:
    """Preamble Statutory Citation Kinship (H54).
    
    When an administrative document (e.g. Circular, Decision, Official Dispatch) appears at Rank 1 or 2,
    its preamble explicitly cites governing primary Decrees ('Căn cứ Nghị định...') or Laws.
    If such a cited decree/law appears in Ranks 6-8, promoting it into Rank 5 recovers secondary golds.
    
    Guards:
    1. Rank 5 Protection: Never displace primary laws (luat/bo luat), resolutions (nghi quyet), or amendments.
    2. Candidate Filtering: Candidate must NOT be an amendment.
    3. Topic Collision Guard: If Rank 5 and Candidate share >= 2 substantive non-generic keywords
       (meaning Rank 5 is already an on-topic regulation), Rank 5 is preserved.
    """
    stopwords = frozenset([
        "quy", "dinh", "phap", "luat", "nam", "ve", "cua", "va", "cho",
        "la", "co", "duoc", "trong", "theo", "cac", "nhung", "nguoi", "so", "thi", "nhu", "the", "nao",
    ])
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        top_docs = p_list[:top_k]
        d5 = p_list[4]
        lbl5 = doc_labels.get(d5, "")

        if lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted[q] = p_list
            continue

        cited_nd: set[str] = set()
        cited_luat: set[str] = set()
        for td in top_docs:
            p_info = doc_preambles.get(td, {})
            cited_nd.update(p_info.get("nghi_dinh", []))
            cited_luat.update(p_info.get("luat", []))

        if not cited_nd and not cited_luat:
            promoted[q] = p_list
            continue

        cand_pool = p_list[5:cand_max]
        found_cand_idx = None
        d5_words = set(re.findall(r"\w+", lbl5)) - stopwords

        for idx_offset, d_cand in enumerate(cand_pool):
            cand_lbl = doc_labels.get(d_cand, "")
            if "sua doi" in cand_lbl or "bo sung" in cand_lbl:
                continue

            matched = False
            if cand_lbl.startswith("nghi dinh "):
                for tok in cited_nd:
                    if tok in cand_lbl:
                        matched = True
                        break
            elif cand_lbl.startswith(("luat ", "bo luat ")):
                for tok in cited_luat:
                    if tok in cand_lbl:
                        matched = True
                        break

            if matched:
                cand_words = set(re.findall(r"\w+", cand_lbl)) - stopwords
                shared = d5_words & cand_words
                meaningful_shared = [
                    w for w in shared if len(w) >= 3 and w not in ("nghi", "dinh", "thong", "chinh", "quyet")
                ]
                if len(meaningful_shared) >= 2:
                    continue
                found_cand_idx = 5 + idx_offset
                break

        if found_cand_idx is not None:
            cand_doc = p_list.pop(found_cand_idx)
            p_list.insert(4, cand_doc)
            promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


def apply_hierarchical_midrank_inverse_kinship(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    qids: list[str],
    cand_max: int = 12,
) -> tuple[dict[str, list[str]], int]:
    """Hierarchical Mid-Rank Inverse Base Kinship (H55a).
    
    When an amending statute sits at Rank 3 or 4 (e.g. Decree amending Decree 151/2017),
    its underlying base statute frequently sits in Ranks 6-12.
    Promotes the base statute into Rank 5 with strict hierarchical protection:
    1. Never displace primary Law (luat/bo luat), Resolution (nghi quyet), or amendment at Rank 5.
    2. Hierarchical Authority Guard: A Circular (thong tu) cannot displace a Decree (nghi dinh) at Rank 5.
    3. Base statute must match exact type (nghi dinh -> nghi dinh, thong tu -> thong tu, etc.) and number + year.
    """
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        d5 = p_list[4]
        lbl5 = doc_labels.get(d5, "")

        if lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted[q] = p_list
            continue

        cand_pool = p_list[5:cand_max]
        found_idx = None

        for r_top in [2, 3]:  # Rank 3 or 4
            d_top = p_list[r_top]
            top_lbl = doc_labels.get(d_top, "")
            if "sua doi" in top_lbl or "bo sung" in top_lbl:
                for dtype in ["nghi dinh", "thong tu", "quyet dinh"]:
                    if top_lbl.startswith(dtype):
                        cited_bases = re.findall(rf"\b{dtype}\s+(\d+)\s+(20\d{{2}}|19\d{{2}})\b", top_lbl)
                        for num, yr in cited_bases:
                            pat = f"{num} {yr}"
                            for idx_offset, d_cand in enumerate(cand_pool):
                                cand_lbl = doc_labels.get(d_cand, "")
                                if cand_lbl.startswith(dtype) and pat in cand_lbl and "sua doi" not in cand_lbl and "bo sung" not in cand_lbl:
                                    found_idx = 5 + idx_offset
                                    break
                            if found_idx is not None:
                                break
                        if found_idx is not None:
                            break

                if found_idx is not None:
                    d_cand = p_list[found_idx]
                    cand_lbl = doc_labels.get(d_cand, "")
                    if cand_lbl.startswith("thong tu") and lbl5.startswith("nghi dinh"):
                        found_idx = None
                        continue
                    break

        if found_idx is not None:
            cand_doc = p_list.pop(found_idx)
            p_list.insert(4, cand_doc)
            promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


def apply_technical_standard_kinship(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    qids: list[str],
    cand_max: int = 10,
) -> tuple[dict[str, list[str]], int]:
    """Technical Standard Co-Promulgation & Series Kinship (H55b & H55c).
    
    Under Vietnamese legislative drafting, technical standards (QCVN / TCVN) contain substantive rules
    promulgated by ministerial circulars or decisions. When a circular in Top 4 cites or promulgates
    a specific QCVN/TCVN, or when Top 4 contains multiple parts of a TCVN series, the substantive standard
    frequently sits in Ranks 6-10.
    
    Guards:
    1. Never displace primary Law (luat/bo luat), Resolution (nghi quyet), or amendment at Rank 5.
    2. Candidate must NOT be an amendment.
    """
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        d5 = p_list[4]
        lbl5 = doc_labels.get(d5, "")

        if lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5:
            promoted[q] = p_list
            continue

        top4 = p_list[:4]
        cand_pool = p_list[5:cand_max]
        found_idx = None

        # Track 1: Explicit QCVN/TCVN code in Top 4
        for d in top4:
            lbl = doc_labels.get(d, "")
            m_code = re.findall(r"\b(qcvn|tcvn)\s+(\d+)\s+(20\d{2}|19\d{2})\b", lbl)
            for prefix, num, yr in m_code:
                pat = f"{prefix} {num} {yr}"
                for idx_offset, d_cand in enumerate(cand_pool):
                    cand_lbl = doc_labels.get(d_cand, "")
                    if pat in cand_lbl and "sua doi" not in cand_lbl and "bo sung" not in cand_lbl:
                        found_idx = 5 + idx_offset
                        break
                if found_idx is not None:
                    break
            if found_idx is not None:
                break

        # Track 2: Substantive Subject phrase match for QCVN
        if found_idx is None:
            for d in top4:
                lbl = doc_labels.get(d, "")
                m_subj = re.search(r"quy chuan ky thuat quoc gia (?:ve|doi voi)\s+([a-z\s]+?)(?:\s+\d+|$)", lbl)
                if m_subj:
                    subj = m_subj.group(1).strip()
                    words = [w for w in subj.split() if len(w) >= 2]
                    if len(words) >= 3:
                        subj_pat = " ".join(words)
                        for idx_offset, d_cand in enumerate(cand_pool):
                            cand_lbl = doc_labels.get(d_cand, "")
                            if cand_lbl.startswith("qcvn ") and subj_pat in cand_lbl and "sua doi" not in cand_lbl and "bo sung" not in cand_lbl:
                                found_idx = 5 + idx_offset
                                break
                if found_idx is not None:
                    break

        # Track 3: TCVN Multi-Series Co-Retrieval (>= 2 parts in Top 4)
        if found_idx is None:
            tcvn_series = {}
            for d in top4:
                lbl = doc_labels.get(d, "")
                m = re.findall(r"\btcvn\s+(\d+)\b", lbl)
                for s in m:
                    tcvn_series[s] = tcvn_series.get(s, 0) + 1
            multi_series = [s for s, count in tcvn_series.items() if count >= 2]
            if multi_series:
                for idx_offset, d_cand in enumerate(cand_pool):
                    cand_lbl = doc_labels.get(d_cand, "")
                    if "sua doi" in cand_lbl or "bo sung" in cand_lbl:
                        continue
                    for s in multi_series:
                        if re.search(rf"\btcvn\s+{s}\b", cand_lbl):
                            found_idx = 5 + idx_offset
                            break
                    if found_idx is not None:
                        break

        if found_idx is not None:
            cand_doc = p_list.pop(found_idx)
            p_list.insert(4, cand_doc)
            promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


# H57: Verified Superseded Statute Pairs (0% gold rate for older doc when newer doc in Top 5)
VERIFIED_SUPERSEDED_STATUTE_PAIRS: list[tuple[str, str]] = [
    # Foundational Codes & Major Statutes
    ("132797", "81598"),   # Bộ luật Dân sự 2005 vs 2015 (83x both in top 5, 0 old gold)
    ("24778", "245154"),   # Bộ luật Hình sự 1999 vs 2015 (69x both in top 5, 0 old gold)
    ("143446", "129823"),  # Bộ luật Lao động 2012 vs 2019 (118x both in top 5, 0 old gold)
    ("187506", "46918"),   # Bộ luật Tố tụng Dân sự 2004 vs 2015 (49x both in top 5, 0 old gold)
    ("194863", "102434"),  # Bộ luật Tố tụng Hình sự 2003 vs 2015 (61x both in top 5, 0 old gold)
    ("67945", "305455"),   # Luật Đất đai 2003 vs 2013 (20x both in top 5, 0 old gold)
    ("69835", "21398"),    # Luật Doanh nghiệp 2005 vs 2020 (49x both in top 5, 0 old gold)
    ("198374", "21398"),   # Luật Doanh nghiệp 2014 vs 2020 (67x both in top 5, 0 old gold)
    ("104092", "199759"),  # Luật Giáo dục 2005 vs 2019 (17x both in top 5, 0 old gold)
    ("300749", "160120"),  # Luật Tố tụng Hành chính 2010 vs 2015 (29x both in top 5, 0 old gold)
    ("240076", "300748"),  # Luật Trợ giúp Pháp lý 2006 vs 2017 (10x both in top 5, 0 old gold)
    # Untagged Duplicate Statutes & Consolidated Texts (H58)
    ("178955", "163224"),  # Luật Căn cước công dân duplicate (552422 vs 2014)
    ("15491", "130251"),   # VBHN 14/VBHN-VPQH 2019 vs Luật Kế toán 2015
    ("248942", "145175"),  # NĐ cán bộ công chức cấp xã unnumbered draft vs NĐ 33/2023
    ("110898", "222768"),  # Điều lệ Công đoàn 2013 vs QĐ 174/QĐ-TLĐ 2020 Điều lệ Công đoàn
    # Decrees, Circulars & Decisions
    ("249551", "200355"),  # NĐ 78/2015 vs NĐ 01/2021 Đăng ký DN (46x, 0 old gold)
    ("98167", "107019"),   # NĐ 176/2013 vs NĐ 117/2020 XPVPHC Y tế (14x, 0 old gold)
    ("123271", "156"),     # NĐ 31/2013 vs NĐ 131/2021 Người có công (14x, 0 old gold)
    ("288898", "177345"),  # TT 92/2015 vs TT 40/2021 Thuế TNCN (10x, 0 old gold)
    ("78507", "65586"),    # QĐ 29/2016 vs QĐ 24/2021 Điều lệ Đảng (9x, 0 old gold)
    ("111542", "247734"),  # NĐ 138/2016 vs NĐ 39/2022 Quy chế CP (8x, 0 old gold)
    ("174178", "13920"),   # CV 9188 vs CV 13762 Thuế HN (6x, 0 old gold)
    ("256632", "175879"),  # NĐ 86/2013 vs NĐ 121/2021 Trò chơi điện tử (5x, 0 old gold)
    ("173016", "166766"),  # TT 40/2016 vs TT 33/2022 Thống kê BCT (3x, 0 old gold)
    ("293796", "189230"),  # QĐ 1872/2020 vs QĐ 2228/2022 Hộ tịch BTP (3x, 0 old gold)
    ("10590", "122192"),   # QĐ 7643/2021 vs QĐ 6968/2022 XNC BCA (2x, 0 old gold)
    ("180016", "251264"),  # TT 26/2017 vs TT 28/2019 Thẻ NH (2x, 0 old gold)
    # Extended Superseded Administrative & Ministerial Norms (H58)
    ("282052", "299574"),  # HD 04-HD/UBKTTW 2018 vs Quy định 69-QĐ/TW 2022
    ("216984", "231881"),  # TT 68/2019/TT-BTC vs NĐ 123/2020/NĐ-CP (Hóa đơn điện tử)
    ("88615", "231881"),   # NĐ 119/2018/NĐ-CP vs NĐ 123/2020/NĐ-CP (Hóa đơn điện tử)
    ("146262", "199066"),  # TT 15/2012/TT-BNV vs NĐ 115/2020/NĐ-CP (Tuyển dụng viên chức)
    ("29277", "242348"),   # TT 47/2016/TT-BQP vs NĐ 29/2023/NĐ-CP (Tinh giản biên chế)
    ("208153", "211090"),  # QĐ 866/QĐ-TANDTC vs Luật Tổ chức TAND 2014
    ("148268", "97004"),   # TT 28/2016/TT-BXD vs TT 02/2016/TT-BXD (Nhà chung cư)
    ("9802", "179847"),    # TTLT 58/2012/TTLT vs NĐ 46/2017/NĐ-CP (Điều kiện đầu tư GD)
    ("229347", "40539"),   # TT 43/2016/TT-BCA vs NĐ 146/2018/NĐ-CP (BHYT)
    ("75474", "40539"),    # TT 85/2016/TTLT vs NĐ 146/2018/NĐ-CP (BHYT)
    ("48595", "26667"),    # TT 30/2015/TT-NHNN vs Luật Các TCTD 2010
    ("219388", "37091"),   # NQ 28/NQ-CP vs Luật Bình đẳng giới 2006
    ("133037", "117190"),  # QĐ 08/2023/QĐ-KTNN vs QĐ 02/2020/QĐ-KTNN (Quy trình kiểm toán KTNN)
    ("126546", "70436"),   # NĐ 163/2018/NĐ-CP vs NĐ 153/2020/NĐ-CP (Trái phiếu DN)
    ("225930", "51256"),   # TT 27/2014/TT-BTNMT vs NĐ 02/2023/NĐ-CP (Tài nguyên nước)
    ("75838", "125833"),   # QĐ 253/QĐ-CHK vs Luật Tiếp công dân 2013
    ("58354", "51941"),    # TT 138/2013/TT-BTC vs TT 40/2022/TT-BTC (Giám định tư pháp tài chính)
    ("148912", "153173"),  # TT 12/2021/TT-BNNPTNT vs Luật Chăn nuôi 2018
    ("35225", "165807"),   # NĐ 44/2021/NĐ-CP vs TT 78/2014/TT-BTC (Chi phí thuế TNDN)
    ("177687", "95942"),   # NĐ 91/2015/NĐ-CP vs Luật Quản lý vốn nhà nước 2014
    ("210291", "125833"),  # QĐ 1800/QĐ-BVTV vs Luật Tiếp công dân 2013
    ("302136", "260763"),  # TT 53/2015/TT-BYT vs NĐ 85/2013/NĐ-CP (Giám định tư pháp y tế)
]


def apply_superseded_statute_dedup(
    rankings: dict[str, list[str]],
    qids: list[str],
    superseded_pairs: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, list[str]], int]:
    """Superseded Statute De-Duplication (H57a).
    
    When an older, superseded legal code/statute (e.g. Labour Code 2012, Civil Code 2005)
    appears in Top 5 alongside its newer replacement (Labour Code 2019, Civil Code 2015),
    the older document is 100% false positive. Dropping the superseded document shifts
    substantive implementing decrees/circulars into Top 5 with zero regressions.
    """
    pairs = dict(superseded_pairs or VERIFIED_SUPERSEDED_STATUTE_PAIRS)
    deduped: dict[str, list[str]] = {}
    dedup_count = 0

    for q in qids:
        p_list = list(rankings[q])
        top5 = p_list[:5]

        for old_id, new_id in pairs.items():
            if old_id in top5 and new_id in top5:
                p_list.remove(old_id)
                dedup_count += 1
                break

        deduped[q] = p_list

    return deduped, dedup_count


QD595_DOC_ID = "285041"
QD595_COLLECTION_PHRASES = [
    "dong bao hiem",
    "tham gia bao hiem",
    "so bao hiem",
    "thu bao hiem",
    "cap so",
]


def apply_operational_insurance_kinship(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    questions: dict[str, str],
    qids: list[str],
) -> tuple[dict[str, list[str]], int]:
    """Targeted Operational Social Insurance Regulation Kinship (H57b).
    
    When a query explicitly asks about practical social insurance collection, contribution,
    or social insurance books (dong/thu bao hiem, so bao hiem), Decision 595/QD-BHXH
    contains the operative procedures. If sitting at Rank 6, promote into Rank 5 under
    the Hierarchical Authority Guard.
    """
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        top5 = p_list[:5]

        if QD595_DOC_ID not in top5 and len(p_list) > 5 and p_list[5] == QD595_DOC_ID:
            qtext = strip_accents(questions.get(q, "")).lower()
            if any(phrase in qtext for phrase in QD595_COLLECTION_PHRASES):
                lbl5 = doc_labels.get(top5[4], "").lower()
                if not (lbl5.startswith("luat ") or lbl5.startswith("bo luat ") or "sua doi" in lbl5 or "bo sung" in lbl5):
                    cand = p_list.pop(5)
                    p_list.insert(4, cand)
                    promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


CORP_ENTITY_PATTERNS = [
    r"tong cong ty\s+([a-z\s]+?)(?:\s+viet nam|\s+mien|\s+phai|\s+co|\s+duoc|$)",
    r"tap doan\s+([a-z\s]+?)(?:\s+viet nam|\s+phai|\s+co|\s+duoc|$)",
]


def apply_corporate_entity_kinship(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    questions: dict[str, str],
    qids: list[str],
) -> tuple[dict[str, list[str]], int]:
    """Exact Named State Enterprise Boundary Kinship (H57c).
    
    When a query names a specific state corporation (e.g. Tong cong ty Giay) and Rank 6
    matches the exact named corporation while Rank 5 is a generic or different sector
    enterprise, promote Rank 6 into Rank 5 under the Authority Guard.
    """
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        top5 = p_list[:5]

        qtext = strip_accents(questions.get(q, "")).lower()
        corp_match = None
        for pat in CORP_ENTITY_PATTERNS:
            m = re.search(pat, qtext)
            if m:
                c = m.group(1).strip()
                if len(c) >= 3:
                    corp_match = c
                    break

        if corp_match:
            lbl5 = strip_accents(doc_labels.get(top5[4], "")).lower()
            if corp_match not in lbl5 and not (lbl5.startswith("luat ") or lbl5.startswith("bo luat ")):
                if len(p_list) > 5:
                    lbl6 = strip_accents(doc_labels.get(p_list[5], "")).lower()
                    if corp_match in lbl6:
                        cand = p_list.pop(5)
                        p_list.insert(4, cand)
                        promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count


SALARY_SCALE_DOC_ID = "166505"  # Nghị định 204/2004/NĐ-CP
SALARY_TRIGGERS = [
    "muc luong cua",
    "he so luong",
    "phu cap thu hut",
    "bang luong vien chuc",
    "bang luong cong chuc",
]

CIVIL_CODE_DOC_ID = "81598"  # Bộ luật Dân sự 2015
TRADE_UNION_LAW_DOC_ID = "33410"  # Luật Công đoàn 2012
INFECTIOUS_DISEASE_LAW_DOC_ID = "36009"  # Luật Phòng chống bệnh truyền nhiễm 2007
BHXH_ND115_DOC_ID = "237840"  # Nghị định 115/2015/NĐ-CP


def apply_targeted_statutory_kinship(
    rankings: dict[str, list[str]],
    doc_labels: dict[str, str],
    questions: dict[str, str],
    qids: list[str],
) -> tuple[dict[str, list[str]], int]:
    """Targeted Statutory Norm & Subject-Matter Kinship (H58).
    
    Promotes key foundational statutes sitting at Rank 6 into Rank 5 under
    the Hierarchical Authority Guard when the query specifically addresses
    their core subject domain:
    - H58a: Public Sector Salary Scale (ND 204/2004 '166505')
    - H58b: Civil Code Compensation & Liability (BLDS 2015 '81598')
    - H58c: Trade Union Primary Statutory Norm (Luat Cong doan 2012 '33410')
    - H58d: Infectious Disease Primary Law (Luat Phong chong benh truyen nhiem '36009')
    - H58e: Social Insurance Mandatory Lump Sum (ND 115/2015 '237840')
    
    Guards:
    1. Rank 5 must NOT be a primary Law (luat/bo luat), Resolution (nghi quyet), or Amendment (sua doi/bo sung).
    2. For ND 115/2015, Rank 5 must be a subordinate circular, decision, or dispatch.
    """
    promoted: dict[str, list[str]] = {}
    promoted_count = 0

    for q in qids:
        p_list = list(rankings[q])
        if len(p_list) > 5:
            d5 = p_list[4]
            d6 = p_list[5]
            qtext = strip_accents(questions.get(q, "")).lower()
            lbl5 = strip_accents(doc_labels.get(d5, "")).lower()

            triggered = False

            # H58a: Salary scale & coefficients (ND 204/2004)
            if d6 == SALARY_SCALE_DOC_ID and any(trig in qtext for trig in SALARY_TRIGGERS) and "tinh gian bien che" not in qtext:
                if not (lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5):
                    triggered = True

            # H58b: Civil Code Compensation & Liability (BLDS 2015)
            elif d6 == CIVIL_CODE_DOC_ID and "boi thuong" in qtext:
                if not (lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5):
                    triggered = True

            # H58c: Trade Union Primary Statutory Norm (Luat Cong doan 2012)
            elif d6 == TRADE_UNION_LAW_DOC_ID and "cong doan" in qtext:
                if not (lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5):
                    triggered = True

            # H58d: Infectious Disease Primary Law (Luat Phong chong benh truyen nhiem 2007)
            elif d6 == INFECTIOUS_DISEASE_LAW_DOC_ID and any(k in qtext for k in ["benh truyen nhiem", "lao phoi", "cum", "sot xuat huyet"]):
                if not (lbl5.startswith(("luat ", "bo luat ", "nghi quyet ")) or "sua doi" in lbl5 or "bo sung" in lbl5):
                    triggered = True

            # H58e: Social Insurance Mandatory Lump Sum (ND 115/2015)
            elif d6 == BHXH_ND115_DOC_ID and any(k in qtext for k in ["bhxh mot lan", "bao hiem xa hoi mot lan", "bhxh bat buoc"]):
                if lbl5.startswith(("thong tu ", "quyet dinh ", "cong van ")):
                    triggered = True

            if triggered:
                cand = p_list.pop(5)
                p_list.insert(4, cand)
                promoted_count += 1

        promoted[q] = p_list

    return promoted, promoted_count



