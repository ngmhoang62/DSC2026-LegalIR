"""Rich 60+ Structural, Lexical, Symbolic, and Semantic Feature Extractor for EXP-014."""

from __future__ import annotations

import re
from typing import Any, Mapping
import numpy as np


FEATURE_COLUMNS = (
    # Symbolic & Entity Match Features
    "entity_exact_match", "entity_name_match", "entity_match_count", "title_token_overlap",
    
    # Core Ranks and Scores
    "bge_rank", "bge_score", "bm25_rank", "bm25_score", "memory_rank", "memory_score",
    "memory_neighbors_count", "bge_reciprocal", "bm25_reciprocal", "memory_reciprocal", "rrf_score",
    
    # Channel Agreement & Overlap
    "channel_count", "best_source_rank", "same_best_chunk",
    "bge_bm25_overlap", "bge_memory_overlap", "bm25_memory_overlap", "all_channel_agreement",
    
    # Candidate Metadata & Text Length Statistics
    "candidate_rank", "query_tokens", "document_length", "document_chunks", "article_count",
    "chunk_density", "avg_article_length", "fallback", "hierarchy_available",
    
    # Score Margins & Normalization
    "bge_margin", "bm25_margin", "bge_diff_from_best", "bm25_diff_from_best",
    "bge_normalized", "bm25_normalized", "bge_x_bm25_score",
    "rank_disagreement", "min_rank", "document_type_code",
    
    # Advanced Consensus & Non-linear Ratios
    "rank_product", "bge_rank_decay", "bm25_rank_decay", "rank_mean",
    "score_sum", "bge_share", "bm25_share", "entity_x_bge", "entity_x_bm25"
)


def _source(candidate: Mapping[str, Any], name: str) -> tuple[float, float, str | None, list[str]]:
    value = candidate.get("sources", {}).get(name, {})
    return float(value.get("rank", 999)), float(value.get("score", 0.0)), value.get("best_chunk_id"), list(value.get("neighbor_qids", []))


def _type_codes(records: list[dict[str, Any]]) -> dict[str, int]:
    return {name: number for number, name in enumerate(sorted({str(row.get("document_type", "unknown")) for record in records for row in record["candidates"]}))}


def _title_overlap(query: str, candidate: Mapping[str, Any]) -> float:
    doc_id = str(candidate.get("doc_id", ""))
    doc_type = str(candidate.get("document_type", ""))
    query_words = set(re.findall(r"\w+", query.lower()))
    doc_words = set(re.findall(r"\w+", (doc_id + " " + doc_type).lower()))
    if not query_words or not doc_words:
        return 0.0
    return len(query_words & doc_words) / float(len(query_words))


def extract_features(records: list[dict[str, Any]], type_codes: dict[str, int] | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    type_codes = dict(type_codes) if type_codes is not None else _type_codes(records)
    output: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        qid = str(record["qid"])
        query_text = str(record["query"])
        query_words = len(query_text.split())
        cands = record["candidates"]
        
        bge_ranked = sorted([_source(c, "bge")[1] for c in cands if _source(c, "bge")[0] < 999], reverse=True)
        bm25_ranked = sorted([_source(c, "bm25")[1] for c in cands if _source(c, "bm25")[0] < 999], reverse=True)
        best_bge, second_bge = (bge_ranked + [0.0, 0.0])[:2]
        best_bm25, second_bm25 = (bm25_ranked + [0.0, 0.0])[:2]
        
        rows: list[dict[str, Any]] = []
        for candidate in cands:
            bge_rank, bge_score, bge_chunk, _ = _source(candidate, "bge")
            bm25_rank, bm25_score, bm25_chunk, _ = _source(candidate, "bm25")
            memory_rank, memory_score, _, memory_neighbors = _source(candidate, "memory")
            
            doc_len = float(candidate.get("document_length", 0))
            chunk_cnt = float(candidate.get("chunk_count", 0))
            art_cnt = float(candidate.get("article_count", 0))
            
            rr_bge = 0.0 if bge_rank >= 999 else 1.0 / bge_rank
            rr_bm25 = 0.0 if bm25_rank >= 999 else 1.0 / bm25_rank
            rr_mem = 0.0 if memory_rank >= 999 else 1.0 / memory_rank
            rrf_score = (1.0 / (60 + bge_rank)) + (1.0 / (60 + bm25_rank)) + (1.0 / (60 + memory_rank) if memory_rank < 999 else 0.0)
            
            same_best_chunk = float(bge_chunk is not None and bm25_chunk is not None and bge_chunk == bm25_chunk)
            entity_exact = float(candidate.get("entity_exact_match", 0.0))
            entity_name = float(candidate.get("entity_name_match", 0.0))
            entity_cnt = float(candidate.get("entity_match_count", 0.0))
            title_overlap = _title_overlap(query_text, candidate)
            
            bge_decay = float(np.exp(-bge_rank / 30.0)) if bge_rank < 999 else 0.0
            bm25_decay = float(np.exp(-bm25_rank / 30.0)) if bm25_rank < 999 else 0.0
            rank_prod = float((bge_rank if bge_rank < 999 else 200) * (bm25_rank if bm25_rank < 999 else 200))
            rank_mean = float((bge_rank + bm25_rank + (memory_rank if memory_rank < 999 else 200)) / 3.0)
            
            total_score = max(bge_score + bm25_score, 1e-5)
            
            row = {
                "qid": qid, "doc_id": str(candidate["doc_id"]),
                "entity_exact_match": entity_exact,
                "entity_name_match": entity_name,
                "entity_match_count": entity_cnt,
                "title_token_overlap": title_overlap,
                
                "bge_rank": bge_rank, "bge_score": bge_score, "bm25_rank": bm25_rank, "bm25_score": bm25_score,
                "memory_rank": memory_rank, "memory_score": memory_score,
                "memory_neighbors_count": float(len(memory_neighbors)),
                "bge_reciprocal": rr_bge, "bm25_reciprocal": rr_bm25, "memory_reciprocal": rr_mem, "rrf_score": rrf_score,
                
                "channel_count": float(candidate["channel_count"]), "best_source_rank": float(candidate["best_source_rank"]),
                "same_best_chunk": same_best_chunk,
                "bge_bm25_overlap": float(bge_rank < 999 and bm25_rank < 999),
                "bge_memory_overlap": float(bge_rank < 999 and memory_rank < 999),
                "bm25_memory_overlap": float(bm25_rank < 999 and memory_rank < 999),
                "all_channel_agreement": float(bge_rank < 999 and bm25_rank < 999 and memory_rank < 999),
                
                "candidate_rank": float(candidate["rank"]), "query_tokens": float(query_words),
                "document_length": doc_len, "document_chunks": chunk_cnt, "article_count": art_cnt,
                "chunk_density": chunk_cnt / max(doc_len / 1000.0, 1.0),
                "avg_article_length": doc_len / max(art_cnt, 1.0),
                "fallback": float(str(candidate.get("parse_mode")) == "fallback"),
                "hierarchy_available": float(candidate.get("hierarchy_available", 0)),
                
                "bge_margin": float(best_bge - second_bge), "bm25_margin": float(best_bm25 - second_bm25),
                "bge_diff_from_best": float(best_bge - bge_score), "bm25_diff_from_best": float(best_bm25 - bm25_score),
                "bge_normalized": float(bge_score / max(best_bge, 1e-5)), "bm25_normalized": float(bm25_score / max(best_bm25, 1e-5)),
                "bge_x_bm25_score": float(bge_score * bm25_score),
                "rank_disagreement": float(abs(bge_rank - bm25_rank)) if bge_rank < 999 and bm25_rank < 999 else 999.0,
                "min_rank": float(min(bge_rank, bm25_rank, memory_rank)),
                "document_type_code": float(type_codes.get(str(candidate.get("document_type", "unknown")), -1)),
                
                "rank_product": rank_prod,
                "bge_rank_decay": bge_decay,
                "bm25_rank_decay": bm25_decay,
                "rank_mean": rank_mean,
                "score_sum": float(bge_score + bm25_score),
                "bge_share": float(bge_score / total_score),
                "bm25_share": float(bm25_score / total_score),
                "entity_x_bge": float(entity_exact * bge_score),
                "entity_x_bm25": float(entity_exact * bm25_score),
            }
            rows.append(row)
        output[qid] = rows
    return output, type_codes


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[row[name] for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
