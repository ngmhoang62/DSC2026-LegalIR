"""EXP-103: Preamble Citation Expansion for Vietnamese LegalIR.

Builds a deterministic, corpus-internal citation graph from document preambles ('Căn cứ ...')
and expands Top-K retrieval candidates to systematically recover cited foundation laws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(r"D:\Study\DSC2026\LegalIR")
CORPUS_DIR = ROOT / "public_test_dataset" / "selected-contexts"
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
AUDIT_EXP034 = ROOT / "results" / "exp034_shallow_retrieval" / "audit" / "REPORT.json"
EXP102_CACHE = ROOT / "cache" / "exp102_mil_nce_retrieval"
EXP102_REPORT = ROOT / "results" / "exp102_mil_nce_retrieval" / "REPORT_full_oof.json"
RESULTS_DIR = ROOT / "results" / "exp103_preamble_citation"
CACHE_DIR = ROOT / "cache" / "exp103_preamble_citation"

DIMENSION = 1024
RANK = 32
KS = (5, 16, 24, 32, 40, 50, 64, 100, 150)

# Regex patterns for Vietnamese statutory codes
SELF_CODE_REGEX = re.compile(
    r"(?:Số|Luật số|Nghị quyết số):\s*([0-9]+/[0-9]+/[A-ZĐ0-9a-z\-]+(?:\s*\d+)?)",
    re.IGNORECASE,
)
CITED_CODE_REGEX = re.compile(
    r"([0-9]+/[0-9]+/[A-ZĐ0-9a-z\-]+)",
    re.IGNORECASE,
)
CITED_LAW_REGEX = re.compile(
    r"Căn cứ\s+(?:Bộ luật|Luật)\s+([^;,\n\.]+)",
    re.IGNORECASE,
)


def normalize_code(code: str) -> str:
    """Normalizes document codes like '02/2009/TT-BTP' or '204/2004/NĐ-CP'."""
    return re.sub(r"\s+", "", code.strip()).upper()


def normalize_law_title(title: str) -> str:
    """Normalizes law titles for lookup."""
    title = re.sub(r"^(Bộ luật|Luật)\s+", "", title.strip(), flags=re.IGNORECASE)
    title = re.sub(r"\s+năm\s+\d+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(sửa đổi\)", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title.strip().lower())


def build_corpus_citation_graph(corpus_dir: Path, cache_dir: Path) -> dict[str, Any]:
    """Extracts official legal numbers and preamble citations across all corpus documents."""
    cache_file = cache_dir / "corpus_citation_graph.json"
    if cache_file.exists():
        print(f"Loading cached citation graph from: {cache_file}", flush=True)
        return json.loads(cache_file.read_text(encoding="utf-8"))
        
    print("[1/3] Scanning corpus context files for document codes & preambles...", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(corpus_dir.glob("context_*.json"))
    code_to_doc_id: dict[str, str] = {}
    title_to_doc_id: dict[str, str] = {}
    doc_preambles: dict[str, list[str]] = {}
    
    for f in files:
        doc_id = f.stem.replace("context_", "")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        passage = data.get("passage", "")
        lines = [l.strip() for l in passage.split("\n") if l.strip()]
        header_text = "\n".join(lines[:30])
        
        # 1. Extract official document number
        m = SELF_CODE_REGEX.search(header_text)
        if m:
            clean_code = normalize_code(m.group(1))
            code_to_doc_id[clean_code] = doc_id
            
        # 2. Extract title if present in passage or link
        for line in lines[:15]:
            if line.isupper() and len(line) > 10:
                norm_title = normalize_law_title(line)
                if norm_title and len(norm_title) > 5:
                    title_to_doc_id[norm_title] = doc_id
                    
        # 3. Store preamble lines (lines 0 to 60)
        preamble_lines = [l for l in lines[:60] if "căn cứ" in l.lower()]
        if preamble_lines:
            doc_preambles[doc_id] = preamble_lines
            
    print(f"[2/3] Mapped {len(code_to_doc_id)} unique legal codes and {len(title_to_doc_id)} normalized law titles.", flush=True)
    
    # 4. Resolve citations for each document
    print("[3/3] Resolving citation outgoing edges...", flush=True)
    citation_graph: dict[str, list[str]] = defaultdict(list)
    
    for doc_id, p_lines in doc_preambles.items():
        seen_cited_docs = set([doc_id])  # Avoid self-loops
        
        for line in p_lines:
            # Match cited document numbers (e.g. 204/2004/NĐ-CP)
            for raw_code in CITED_CODE_REGEX.findall(line):
                code_norm = normalize_code(raw_code)
                target_doc = code_to_doc_id.get(code_norm)
                if target_doc and target_doc not in seen_cited_docs:
                    seen_cited_docs.add(target_doc)
                    citation_graph[doc_id].append(target_doc)
                    
            # Match cited law names (e.g. Căn cứ Luật thi đua, khen thưởng)
            for raw_law in CITED_LAW_REGEX.findall(line):
                law_norm = normalize_law_title(raw_law)
                target_doc = title_to_doc_id.get(law_norm)
                if target_doc and target_doc not in seen_cited_docs:
                    seen_cited_docs.add(target_doc)
                    citation_graph[doc_id].append(target_doc)
                    
    result = {
        "citation_graph": {k: v for k, v in citation_graph.items() if v},
        "stats": {
            "total_documents": len(files),
            "documents_with_codes": len(code_to_doc_id),
            "documents_with_citations": len(citation_graph),
            "total_citation_edges": sum(len(v) for v in citation_graph.values()),
        },
    }
    
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved citation graph ({result['stats']['total_citation_edges']} edges across {result['stats']['documents_with_citations']} docs) to: {cache_file}", flush=True)
    return result


def expand_single_ranking(
    raw_ranking: list[str],
    citation_graph: dict[str, list[str]],
    seed_k: int = 3,
    max_citations_per_seed: int = 3,
    max_total_citations: int = 5,
    max_output: int = 150,
) -> list[str]:
    """Expands a single query ranking by injecting cited parent documents behind top seeds."""
    if not raw_ranking:
        return raw_ranking
        
    seen = set()
    expanded = []
    
    # 1. Take the top seed candidates
    seeds = raw_ranking[:seed_k]
    for doc_id in seeds:
        if doc_id not in seen:
            seen.add(doc_id)
            expanded.append(doc_id)
            
    # 2. Extract citations from the top seeds
    citations_to_inject = []
    for seed_doc in seeds:
        cited_parents = citation_graph.get(seed_doc, [])
        for p_doc in cited_parents[:max_citations_per_seed]:
            if p_doc not in seen and p_doc not in citations_to_inject:
                citations_to_inject.append(p_doc)
                if len(citations_to_inject) >= max_total_citations:
                    break
        if len(citations_to_inject) >= max_total_citations:
            break
            
    # 3. Inject cited parent documents right after the seeds
    for p_doc in citations_to_inject:
        seen.add(p_doc)
        expanded.append(p_doc)
        
    # 4. Append remaining original candidates
    for doc_id in raw_ranking[seed_k:]:
        if doc_id not in seen:
            seen.add(doc_id)
            expanded.append(doc_id)
            if len(expanded) >= max_output:
                break
                
    return expanded[:max_output]


def compute_metrics(rankings: dict[str, list[str]], train_data: dict[str, Any], qids: list[str]):
    recalls = {k: [] for k in KS}
    rr_5 = []
    
    for qid in qids:
        gold_set = set(map(str, train_data[qid]["answer"]))
        if not gold_set:
            continue
        predicted = rankings.get(qid, [])
        
        for k in KS:
            pred_k = set(predicted[:k])
            recalls[k].append(len(pred_k & gold_set) / len(gold_set))
            
        first_gold_rank = None
        for r, doc_id in enumerate(predicted[:5], 1):
            if doc_id in gold_set:
                first_gold_rank = r
                break
        rr_5.append(1.0 / first_gold_rank if first_gold_rank else 0.0)
        
    return {
        **{f"recall@{k}": float(np.mean(recalls[k])) for k in KS},
        "mrr@5": float(np.mean(rr_5)),
        "evaluable_queries": len(rr_5),
    }


def evaluate_exp103_expansion(
    seed_k: int = 3,
    max_citations_per_seed: int = 3,
    max_total_citations: int = 4,
):
    print(f"\n=======================================================")
    print(f"RUNNING EXP-103: Preamble Citation Expansion (seed_k={seed_k}, max_cits={max_citations_per_seed})")
    print(f"=======================================================")
    
    # 1. Load Citation Graph
    graph_data = build_corpus_citation_graph(CORPUS_DIR, CACHE_DIR)
    citation_graph = graph_data["citation_graph"]
    
    # 2. Load Train Labels & Folds
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    
    non_eval_qids = set()
    if AUDIT_EXP034.exists():
        audit_data = json.loads(AUDIT_EXP034.read_text(encoding="utf-8"))
        non_eval_qids = set(map(str, audit_data.get("label_stats", {}).get("non_evaluable_qids", [])))
        
    all_qids = sorted(train_data.keys())
    evaluable_all_qids = [q for q in all_qids if q not in non_eval_qids]
    
    # 3. Load EXP-102 or Generate base rankings
    # We will load EXP-102 model and evaluate on the fly or load rankings
    import exp102_mil_nce_retrieval as exp102
    data = exp102.load_corpus_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
    documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
    
    oof_raw_rankings = {}
    oof_expanded_rankings = {}
    per_fold_metrics = {}
    
    start_time = time.time()
    
    for fold_name, heldout_qids in sorted(folds.items()):
        heldout_qids = list(map(str, heldout_qids))
        ckpt_path = EXP102_CACHE / f"{fold_name}.pt"
        model = exp102.ResidualProjection(DIMENSION, RANK).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        # Base EXP-102 RRF ranking
        _, base_rankings = exp102.evaluate_model(
            data=data,
            qids=heldout_qids,
            model=model,
            documents=documents,
            device=device,
            rrf_mode="static",
            static_alpha=0.65,
            rrf_k=32,
        )
        oof_raw_rankings.update(base_rankings)
        
        # Apply Citation Expansion
        fold_expanded = {}
        for qid in heldout_qids:
            raw_r = base_rankings[qid]
            exp_r = expand_single_ranking(
                raw_ranking=raw_r,
                citation_graph=citation_graph,
                seed_k=seed_k,
                max_citations_per_seed=max_citations_per_seed,
                max_total_citations=max_total_citations,
            )
            fold_expanded[qid] = exp_r
            
        oof_expanded_rankings.update(fold_expanded)
        
        eval_heldout = [q for q in heldout_qids if q not in non_eval_qids]
        fold_m = compute_metrics(fold_expanded, train_data, eval_heldout)
        per_fold_metrics[fold_name] = fold_m
        print(f"[{fold_name}] R@5: {fold_m['recall@5']*100:.2f}% | R@32: {fold_m['recall@32']*100:.2f}% | R@50: {fold_m['recall@50']*100:.2f}% | R@64: {fold_m['recall@64']*100:.2f}% | MRR@5: {fold_m['mrr@5']:.4f}", flush=True)
        
    total_metrics_raw = compute_metrics(oof_raw_rankings, train_data, evaluable_all_qids)
    total_metrics_expanded = compute_metrics(oof_expanded_rankings, train_data, evaluable_all_qids)
    elapsed = time.time() - start_time
    
    print("\n=======================================================")
    print(f"EXP-103 FULL 5-FOLD OOF RESULTS:")
    print(f"  [RAW EXP-102] Recall@5: {total_metrics_raw['recall@5']*100:.2f}% | R@32: {total_metrics_raw['recall@32']*100:.2f}% | R@50: {total_metrics_raw['recall@50']*100:.2f}% | R@64: {total_metrics_raw['recall@64']*100:.2f}% | MRR@5: {total_metrics_raw['mrr@5']:.4f}")
    print(f"  [EXP-103 CIT] Recall@5: {total_metrics_expanded['recall@5']*100:.2f}% | R@32: {total_metrics_expanded['recall@32']*100:.2f}% | R@50: {total_metrics_expanded['recall@50']*100:.2f}% | R@64: {total_metrics_expanded['recall@64']*100:.2f}% | MRR@5: {total_metrics_expanded['mrr@5']:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print("=======================================================")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "exp103_preamble_citation",
        "hyperparameters": {
            "seed_k": seed_k,
            "max_citations_per_seed": max_citations_per_seed,
            "max_total_citations": max_total_citations,
        },
        "metrics_raw_exp102": total_metrics_raw,
        "metrics_expanded_exp103": total_metrics_expanded,
        "per_fold": per_fold_metrics,
        "elapsed_seconds": elapsed,
    }
    report_file = RESULTS_DIR / f"REPORT_seed{seed_k}_cit{max_citations_per_seed}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved EXP-103 report to: {report_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-k", type=int, default=3)
    parser.add_argument("--max-cit", type=int, default=3)
    parser.add_argument("--max-total", type=int, default=4)
    args = parser.parse_args()
    evaluate_exp103_expansion(seed_k=args.seed_k, max_citations_per_seed=args.max_cit, max_total_citations=args.max_total)
