"""EXP-102: Document-Level Multiple Instance Learning (MIL-NCE) with Dynamic Specificity Routing.

Includes Dynamic Query Specificity Routing:
- Statutory / Citation queries -> Sparse-favored RRF (alpha = 0.35)
- Pure Colloquial queries -> Dense-dominant RRF (alpha = 0.85 - 0.95)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
E5_DIR = ROOT / "cache" / "e5_final_v1"
QUERY_EMB_DIR = ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
EXCLUSIONS_PATH = ROOT / "cache" / "final_preprocessed_v2" / "exclusions.json"
CANDIDATES_EXP022 = ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"
AUDIT_EXP034 = ROOT / "results" / "exp034_shallow_retrieval" / "audit" / "REPORT.json"
RESULTS_DIR = ROOT / "results" / "exp102_mil_nce_retrieval"
CACHE_DIR = ROOT / "cache" / "exp102_mil_nce_retrieval"

DIMENSION = 1024
RANK = 32
SEED = 2026
KS = (5, 16, 24, 32, 40, 50, 64, 100, 150)

STATUTORY_REGEX = re.compile(
    r"(điều\s+\d+|khoản\s+\d+|nghị\s+định|thông\s+tư|luật\s+[a-zà-ỹ\s]+|bộ\s+luật|quyết\s+định\s+\d+)",
    re.IGNORECASE,
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class ResidualProjection(nn.Module):
    """Low-rank residual projection initialized close to identity."""

    def __init__(self, dimension: int = DIMENSION, rank: int = RANK) -> None:
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        delta = self.up(torch.nn.functional.gelu(self.down(query)))
        return torch.nn.functional.normalize(query + delta, p=2, dim=-1)


def mil_document_score(
    query: torch.Tensor,
    doc_chunk_indices: torch.Tensor,
    documents: torch.Tensor,
    tau_chunk: float = 15.0,
) -> torch.Tensor:
    chunk_vecs = documents.index_select(0, doc_chunk_indices)
    sims = torch.matmul(chunk_vecs, query)
    n_chunks = chunk_vecs.shape[0]
    lse = torch.logsumexp(sims * tau_chunk, dim=0) / tau_chunk
    if n_chunks > 1:
        lse = lse - (math.log(n_chunks) / tau_chunk)
    return lse


def compute_mil_nce_loss(
    query: torch.Tensor,
    pos_chunk_indices_list: list[torch.Tensor],
    neg_chunk_indices_list: list[torch.Tensor],
    documents: torch.Tensor,
    tau_chunk: float = 15.0,
    tau_doc: float = 0.05,
) -> torch.Tensor:
    pos_scores = torch.stack([
        mil_document_score(query, idx, documents, tau_chunk=tau_chunk)
        for idx in pos_chunk_indices_list
    ])
    neg_scores = torch.stack([
        mil_document_score(query, idx, documents, tau_chunk=tau_chunk)
        for idx in neg_chunk_indices_list
    ])
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(0).expand(pos_scores.shape[0], -1)], dim=1) / tau_doc
    loss = -torch.log_softmax(all_scores, dim=1)[:, 0].mean()
    return loss


def load_corpus_data():
    print("[1/5] Loading chunk index & embeddings...", flush=True)
    chunk_doc_ids = []
    with (E5_DIR / "chunk_ids.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                chunk_doc_ids.append(str(row["doc_id"]))
    
    chunk_embeddings = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
    
    print("[2/5] Loading query embeddings & labels...", flush=True)
    with (QUERY_EMB_DIR / "train_query_ids.json").open("r", encoding="utf-8") as f:
        query_ids = [str(x) for x in json.load(f)]
    query_embeddings = np.load(QUERY_EMB_DIR / "train_queries.f32.npy")
    qid_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    
    non_eval_qids = set()
    if AUDIT_EXP034.exists():
        audit_data = json.loads(AUDIT_EXP034.read_text(encoding="utf-8"))
        non_eval_qids = set(map(str, audit_data.get("label_stats", {}).get("non_evaluable_qids", [])))
    
    print("[3/5] Building doc to chunk index mapping...", flush=True)
    doc_to_chunk_indices = defaultdict(list)
    for c_idx, doc_id in enumerate(chunk_doc_ids):
        doc_to_chunk_indices[doc_id].append(c_idx)
    all_docs = sorted(doc_to_chunk_indices.keys())
    
    print("[4/5] Loading BM25 ranks from EXP-022...", flush=True)
    bm25_ranks_by_qid = defaultdict(dict)
    if CANDIDATES_EXP022.exists():
        with CANDIDATES_EXP022.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    qid = str(row["qid"])
                    for cand in row["candidates"]:
                        doc_id = str(cand["doc_id"])
                        b_rank = cand.get("sources", {}).get("bm25", {}).get("rank")
                        if b_rank is not None:
                            bm25_ranks_by_qid[qid][doc_id] = int(b_rank)
                            
    return {
        "chunk_doc_ids": chunk_doc_ids,
        "chunk_embeddings": chunk_embeddings,
        "query_ids": query_ids,
        "query_embeddings": query_embeddings,
        "qid_to_idx": qid_to_idx,
        "train_data": train_data,
        "folds": folds,
        "non_eval_qids": non_eval_qids,
        "doc_to_chunk_indices": doc_to_chunk_indices,
        "all_docs": all_docs,
        "bm25_ranks": bm25_ranks_by_qid,
    }


def get_query_alpha(query_text: str, dense_dominant_alpha: float = 0.85, sparse_dominant_alpha: float = 0.35) -> float:
    """Computes dynamic specificity alpha for a query."""
    if STATUTORY_REGEX.search(query_text):
        return sparse_dominant_alpha
    return dense_dominant_alpha


def evaluate_model(
    data: dict[str, Any],
    qids: list[str],
    model: nn.Module,
    documents: torch.Tensor,
    device: torch.device,
    top_k_chunks: int = 4096,
    max_cand: int = 150,
    rrf_mode: str | None = None,
    static_alpha: float = 0.65,
    dense_alpha: float = 0.85,
    sparse_alpha: float = 0.35,
    rrf_k: int = 32,
):
    model.eval()
    query_embeddings = data["query_embeddings"]
    qid_to_idx = data["qid_to_idx"]
    chunk_doc_ids = data["chunk_doc_ids"]
    train_data = data["train_data"]
    non_eval_qids = data["non_eval_qids"]
    bm25_ranks = data["bm25_ranks"]
    
    indices = [qid_to_idx[qid] for qid in qids]
    raw_query_vecs = torch.from_numpy(query_embeddings[indices]).to(device)
    
    with torch.inference_mode():
        projected_queries = model(raw_query_vecs)
        
    num_queries = len(qids)
    batch_size = 64
    rankings = {}
    
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        q_batch = projected_queries[start_idx:end_idx]
        
        scores = torch.matmul(q_batch, documents.T)
        top_scores, top_indices = torch.topk(scores, k=min(top_k_chunks, documents.shape[0]), dim=-1)
        
        top_scores_np = top_scores.cpu().numpy()
        top_indices_np = top_indices.cpu().numpy()
        
        for b in range(end_idx - start_idx):
            qid = qids[start_idx + b]
            q_scores = top_scores_np[b]
            q_indices = top_indices_np[b]
            
            doc_scores = defaultdict(list)
            for score, c_idx in zip(q_scores, q_indices):
                doc_id = chunk_doc_ids[c_idx]
                if len(doc_scores[doc_id]) < 2:
                    doc_scores[doc_id].append(float(score))
                    
            doc_ranked = []
            for doc_id, s_list in doc_scores.items():
                agg_score = sum(s_list) / len(s_list)
                doc_ranked.append((doc_id, agg_score))
                
            doc_ranked.sort(key=lambda x: -x[1])
            
            if rrf_mode is None:
                rankings[qid] = [d[0] for d in doc_ranked[:max_cand]]
            elif rrf_mode == "static":
                dense_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(doc_ranked, 1)}
                bm25_map = bm25_ranks.get(qid, {})
                all_candidate_docs = set(dense_rank_map.keys()) | set(bm25_map.keys())
                
                fused_scores = {}
                for doc_id in all_candidate_docs:
                    d_r = dense_rank_map.get(doc_id)
                    b_r = bm25_map.get(doc_id)
                    score = 0.0
                    if d_r is not None:
                        score += static_alpha * (1.0 / (rrf_k + d_r))
                    if b_r is not None:
                        score += (1.0 - static_alpha) * (1.0 / (rrf_k + b_r))
                    fused_scores[doc_id] = score
                    
                sorted_fused = sorted(fused_scores.keys(), key=lambda d: -fused_scores[d])
                rankings[qid] = sorted_fused[:max_cand]
            elif rrf_mode == "dynamic":
                # Dynamic Query Specificity Routing
                q_text = train_data[qid]["question"]
                alpha_q = get_query_alpha(q_text, dense_dominant_alpha=dense_alpha, sparse_dominant_alpha=sparse_alpha)
                
                dense_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(doc_ranked, 1)}
                bm25_map = bm25_ranks.get(qid, {})
                all_candidate_docs = set(dense_rank_map.keys()) | set(bm25_map.keys())
                
                fused_scores = {}
                for doc_id in all_candidate_docs:
                    d_r = dense_rank_map.get(doc_id)
                    b_r = bm25_map.get(doc_id)
                    score = 0.0
                    if d_r is not None:
                        score += alpha_q * (1.0 / (rrf_k + d_r))
                    if b_r is not None:
                        score += (1.0 - alpha_q) * (1.0 / (rrf_k + b_r))
                    fused_scores[doc_id] = score
                    
                sorted_fused = sorted(fused_scores.keys(), key=lambda d: -fused_scores[d])
                rankings[qid] = sorted_fused[:max_cand]
                
    evaluable_qids = [q for q in qids if q not in non_eval_qids]
    metrics = compute_metrics(rankings, train_data, evaluable_qids)
    return metrics, rankings


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


def evaluate_dynamic_routing_grid(dense_alphas: list[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
    print(f"\n=======================================================")
    print(f"RUNNING EXP-102 DYNAMIC SPECIFICITY ROUTING GRID EVALUATION")
    print(f"=======================================================")
    
    data = load_corpus_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[5/5] Moving 343k chunk embeddings to {device}...", flush=True)
    documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
    documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
    
    folds = data["folds"]
    all_qids = sorted(data["train_data"].keys())
    evaluable_all_qids = [q for q in all_qids if q not in data["non_eval_qids"]]
    
    # Load models for all 5 folds
    models = {}
    for fold_name in sorted(folds.keys()):
        ckpt_path = CACHE_DIR / f"{fold_name}.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Missing checkpoint: {ckpt_path}. Please train EXP-102 first.")
        model = ResidualProjection(dimension=DIMENSION, rank=RANK).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models[fold_name] = model
        
    grid_results = {}
    
    for dense_alpha in dense_alphas:
        print(f"\n--- Evaluating Dynamic Routing with dense_alpha={dense_alpha}, sparse_alpha=0.35 ---")
        oof_rankings = {}
        per_fold_metrics = {}
        
        for fold_name, heldout_qids in sorted(folds.items()):
            heldout_qids = list(map(str, heldout_qids))
            model = models[fold_name]
            metrics, rankings = evaluate_model(
                data=data,
                qids=heldout_qids,
                model=model,
                documents=documents,
                device=device,
                rrf_mode="dynamic",
                dense_alpha=dense_alpha,
                sparse_alpha=0.35,
                rrf_k=32,
            )
            oof_rankings.update(rankings)
            per_fold_metrics[fold_name] = metrics
            
        total_metrics = compute_metrics(oof_rankings, data["train_data"], evaluable_all_qids)
        grid_results[f"dense_{dense_alpha:.2f}"] = {
            "dense_alpha": dense_alpha,
            "sparse_alpha": 0.35,
            "total_metrics": total_metrics,
            "per_fold": per_fold_metrics,
        }
        
        print(f"[OOF dense_alpha={dense_alpha:.2f}] R@5: {total_metrics['recall@5']*100:.2f}% | R@32: {total_metrics['recall@32']*100:.2f}% | R@50: {total_metrics['recall@50']*100:.2f}% | R@64: {total_metrics['recall@64']*100:.2f}% | MRR@5: {total_metrics['mrr@5']:.4f}")
        
    report_file = RESULTS_DIR / "REPORT_dynamic_routing_grid.json"
    report_file.write_text(json.dumps(grid_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full dynamic routing grid report to: {report_file}")
    return grid_results


if __name__ == "__main__":
    evaluate_dynamic_routing_grid()
