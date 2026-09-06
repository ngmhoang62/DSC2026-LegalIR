"""EXP-101: Manifold Procrustes Subspace Alignment for Vietnamese LegalIR.

Evaluates orthogonal Procrustes rotation and Hybrid Procrustes + BM25 RRF on the 5-fold CV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(r"D:\Study\DSC2026\LegalIR")
E5_DIR = ROOT / "cache" / "e5_final_v1"
QUERY_EMB_DIR = ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
CANDIDATES_EXP022 = ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"
RESULTS_DIR = ROOT / "results" / "exp101_procrustes_alignment"


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def solve_orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Solves min_{W, W^T W = I} || X W - Y ||_F^2 via SVD: X^T Y = U \Sigma V^T => W = U V^T."""
    X_norm = l2_normalize(X)
    Y_norm = l2_normalize(Y)
    M = X_norm.T @ Y_norm
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    return W.astype(np.float32)


def solve_ridge_alignment(X: np.ndarray, Y: np.ndarray, lambda_reg: float = 1e-2) -> np.ndarray:
    """Solves min_W || X W - Y ||_F^2 + lambda || W ||_F^2."""
    X_norm = l2_normalize(X)
    Y_norm = l2_normalize(Y)
    D = X_norm.shape[1]
    W = np.linalg.solve(X_norm.T @ X_norm + lambda_reg * np.eye(D, dtype=np.float32), X_norm.T @ Y_norm)
    return W.astype(np.float32)


def load_corpus_and_queries():
    print("[1/5] Loading chunk embeddings and index...", flush=True)
    chunk_doc_ids = []
    with (E5_DIR / "chunk_ids.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                chunk_doc_ids.append(str(row["doc_id"]))
    
    chunk_embeddings = np.load(E5_DIR / "embeddings.f16.npy", mmap_mode="r")
    
    with (QUERY_EMB_DIR / "train_query_ids.json").open("r", encoding="utf-8") as f:
        query_ids = [str(x) for x in json.load(f)]
    query_embeddings = np.load(QUERY_EMB_DIR / "train_queries.f32.npy")
    qid_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    
    print("[2/5] Building doc-to-chunk index...", flush=True)
    doc_to_chunk_indices = defaultdict(list)
    for c_idx, doc_id in enumerate(chunk_doc_ids):
        doc_to_chunk_indices[doc_id].append(c_idx)
        
    print("[3/5] Loading BM25 ranks from EXP-022...", flush=True)
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
        "doc_to_chunk_indices": doc_to_chunk_indices,
        "bm25_ranks": bm25_ranks_by_qid,
    }


def build_training_pairs(context, train_qids: list[str], target_mode: str = "mean_doc"):
    X_list = []
    Y_list = []
    
    qid_to_idx = context["qid_to_idx"]
    query_embeddings = context["query_embeddings"]
    train_data = context["train_data"]
    doc_to_chunk_indices = context["doc_to_chunk_indices"]
    chunk_embeddings = context["chunk_embeddings"]
    
    for qid in train_qids:
        q_idx = qid_to_idx.get(qid)
        if q_idx is None:
            continue
        q_vec = query_embeddings[q_idx]
        gold_doc_ids = [str(g) for g in train_data[qid]["answer"]]
        
        for g_doc in gold_doc_ids:
            c_indices = doc_to_chunk_indices.get(g_doc)
            if not c_indices:
                continue
            
            if target_mode == "mean_doc":
                doc_vecs = np.asarray(chunk_embeddings[c_indices], dtype=np.float32)
                target_vec = np.mean(doc_vecs, axis=0)
            elif target_mode == "first_chunk":
                target_vec = np.asarray(chunk_embeddings[c_indices[0]], dtype=np.float32)
            else:
                raise ValueError(f"Unknown target_mode: {target_mode}")
                
            X_list.append(q_vec)
            Y_list.append(target_vec)
            
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y


def evaluate_queries(
    context,
    qids: list[str],
    transformed_query_vecs: np.ndarray,
    top_k_chunks: int = 2048,
    max_cand: int = 150,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    rrf_alpha: float | None = None,
    rrf_k: int = 32,
):
    chunk_embeddings = context["chunk_embeddings"]
    chunk_doc_ids = context["chunk_doc_ids"]
    train_data = context["train_data"]
    bm25_ranks_by_qid = context["bm25_ranks"]
    
    num_chunks, dim = chunk_embeddings.shape
    num_queries = len(qids)
    
    docs_tensor = torch.from_numpy(np.asarray(chunk_embeddings, dtype=np.float32)).to(device)
    docs_tensor = docs_tensor / torch.norm(docs_tensor, dim=-1, keepdim=True).clamp_min(1e-12)
    
    queries_tensor = torch.from_numpy(l2_normalize(transformed_query_vecs)).to(device)
    
    batch_size = 64
    rankings = {}
    
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        q_batch = queries_tensor[start_idx:end_idx]
        
        scores = torch.matmul(q_batch, docs_tensor.T)
        top_scores, top_indices = torch.topk(scores, k=min(top_k_chunks, num_chunks), dim=-1)
        
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
            
            if rrf_alpha is None:
                # Pure dense
                rankings[qid] = [d[0] for d in doc_ranked[:max_cand]]
            else:
                # RRF fusion between Dense and BM25
                dense_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(doc_ranked, 1)}
                bm25_map = bm25_ranks_by_qid.get(qid, {})
                all_candidate_docs = set(dense_rank_map.keys()) | set(bm25_map.keys())
                
                fused_scores = {}
                for doc_id in all_candidate_docs:
                    d_r = dense_rank_map.get(doc_id)
                    b_r = bm25_map.get(doc_id)
                    score = 0.0
                    if d_r is not None:
                        score += rrf_alpha * (1.0 / (rrf_k + d_r))
                    if b_r is not None:
                        score += (1.0 - rrf_alpha) * (1.0 / (rrf_k + b_r))
                    fused_scores[doc_id] = score
                    
                sorted_fused = sorted(fused_scores.keys(), key=lambda d: -fused_scores[d])
                rankings[qid] = sorted_fused[:max_cand]
            
    metrics = compute_metrics(rankings, train_data, qids)
    return metrics, rankings


def compute_metrics(rankings, train_data, qids):
    ks = (5, 10, 20, 32, 50, 64, 100, 150)
    recalls = {k: [] for k in ks}
    rr_5 = []
    
    for qid in qids:
        gold_set = set(map(str, train_data[qid]["answer"]))
        if not gold_set:
            continue
        predicted = rankings[qid]
        
        for k in ks:
            pred_k = set(predicted[:k])
            recalls[k].append(len(pred_k & gold_set) / len(gold_set))
            
        first_gold_rank = None
        for r, doc_id in enumerate(predicted[:5], 1):
            if doc_id in gold_set:
                first_gold_rank = r
                break
        rr_5.append(1.0 / first_gold_rank if first_gold_rank else 0.0)
        
    return {
        **{f"recall@{k}": float(np.mean(recalls[k])) for k in ks},
        "mrr@5": float(np.mean(rr_5)),
        "queries": len(rr_5),
    }


def run_experiment(method: str = "orthogonal", target_mode: str = "mean_doc", lambda_reg: float = 1e-2, rrf_alpha: float | None = None, rrf_k: int = 32):
    print(f"\n=======================================================")
    print(f"RUNNING EXP-101: method={method}, target_mode={target_mode}, rrf_alpha={rrf_alpha}, rrf_k={rrf_k}")
    print(f"=======================================================")
    
    context = load_corpus_and_queries()
    folds = context["folds"]
    qid_to_idx = context["qid_to_idx"]
    query_embeddings = context["query_embeddings"]
    
    all_qids = sorted(context["train_data"].keys())
    fold_metrics = {}
    oof_rankings = {}
    
    start_time = time.time()
    
    for fold_name, heldout_qids in sorted(folds.items()):
        heldout_set = set(map(str, heldout_qids))
        train_qids = [qid for qid in all_qids if qid not in heldout_set]
        
        X_train, Y_train = build_training_pairs(context, train_qids, target_mode=target_mode)
        
        if method == "orthogonal":
            W = solve_orthogonal_procrustes(X_train, Y_train)
        elif method == "ridge":
            W = solve_ridge_alignment(X_train, Y_train, lambda_reg=lambda_reg)
        elif method == "baseline":
            W = np.eye(X_train.shape[1], dtype=np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        heldout_indices = [qid_to_idx[qid] for qid in heldout_qids]
        raw_heldout_vecs = query_embeddings[heldout_indices]
        aligned_heldout_vecs = raw_heldout_vecs @ W
        
        metrics, rankings = evaluate_queries(context, heldout_qids, aligned_heldout_vecs, rrf_alpha=rrf_alpha, rrf_k=rrf_k)
        fold_metrics[fold_name] = metrics
        oof_rankings.update(rankings)
        
        print(f"[{fold_name}] R@5: {metrics['recall@5']:.4f} | R@20: {metrics['recall@20']:.4f} | R@50: {metrics['recall@50']:.4f} | R@64: {metrics['recall@64']:.4f} | MRR@5: {metrics['mrr@5']:.4f}")
        
    total_oof_metrics = compute_metrics(oof_rankings, context["train_data"], all_qids)
    elapsed = time.time() - start_time
    
    suffix = f"rrf_a{rrf_alpha}_k{rrf_k}" if rrf_alpha is not None else "dense"
    print("\n=======================================================")
    print(f"FULL 5-FOLD OOF RESULTS ({method}, {target_mode}, {suffix}):")
    print(f"  Recall@5:   {total_oof_metrics['recall@5']*100:.2f}%")
    print(f"  Recall@10:  {total_oof_metrics['recall@10']*100:.2f}%")
    print(f"  Recall@20:  {total_oof_metrics['recall@20']*100:.2f}%")
    print(f"  Recall@32:  {total_oof_metrics['recall@32']*100:.2f}%")
    print(f"  Recall@50:  {total_oof_metrics['recall@50']*100:.2f}%")
    print(f"  Recall@64:  {total_oof_metrics['recall@64']*100:.2f}%")
    print(f"  Recall@100: {total_oof_metrics['recall@100']*100:.2f}%")
    print(f"  Recall@150: {total_oof_metrics['recall@150']*100:.2f}%")
    print(f"  MRR@5:      {total_oof_metrics['mrr@5']:.4f}")
    print(f"  Time:       {elapsed:.1f}s")
    print("=======================================================")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "exp101_procrustes_alignment",
        "method": method,
        "target_mode": target_mode,
        "rrf_alpha": rrf_alpha,
        "rrf_k": rrf_k,
        "metrics_oof": total_oof_metrics,
        "per_fold": fold_metrics,
        "elapsed_seconds": elapsed,
    }
    report_file = RESULTS_DIR / f"REPORT_{method}_{target_mode}_{suffix}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to: {report_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("orthogonal", "ridge", "baseline"), default="orthogonal")
    parser.add_argument("--target-mode", choices=("mean_doc", "first_chunk"), default="mean_doc")
    parser.add_argument("--lambda-reg", type=float, default=1e-2)
    parser.add_argument("--rrf-alpha", type=float, default=None)
    parser.add_argument("--rrf-k", type=int, default=32)
    args = parser.parse_args()
    run_experiment(method=args.method, target_mode=args.target_mode, lambda_reg=args.lambda_reg, rrf_alpha=args.rrf_alpha, rrf_k=args.rrf_k)
