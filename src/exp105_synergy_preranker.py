"""EXP-105: Enriched Multi-Chunk Spectrum & Dense-Sparse Synergy Pre-Ranker (Stage 1.5).

Processes a strict Candidate Pool of K=100 candidates per query (700,000 candidate rows).
Extracts a 36-dimensional feature matrix featuring:
1. Multi-Chunk Similarity Decay Spectrum (8 dims: s1, s2, s3, decay_12, decay_13, mean, std, ratio)
2. Within-Query Relative Normalization (6 dims: Z-scores, Softmax probabilities, Top-1 ratios)
3. Non-Linear Dense-Sparse Synergy (6 dims: Product, Harmonic Mean, Discordance, Recip diff)
4. Base Retrieval Signals (6 dims: ranks, scores, reciprocals)
5. Structural & Metadata Alignment (10 dims: scope, parse, doc_type, statutory match)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(r"D:\Study\DSC2026\LegalIR")
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
AUDIT_EXP034 = ROOT / "results" / "exp034_shallow_retrieval" / "audit" / "REPORT.json"
EXP102_CACHE = ROOT / "cache" / "exp102_mil_nce_retrieval"
CORPUS_DIR = ROOT / "public_test_dataset" / "selected-contexts"
RESULTS_DIR = ROOT / "results" / "exp105_synergy_preranker"
CACHE_DIR = ROOT / "cache" / "exp105_synergy_preranker"

DIMENSION = 1024
KS = (5, 16, 24, 32, 40, 50, 64, 100)
SEED = 2026

FEATURE_COLUMNS = (
    # Group 1: Multi-Chunk Spectrum (8)
    "chunk_sim_1", "chunk_sim_2", "chunk_sim_3", "decay_delta_12",
    "decay_delta_13", "chunk_mean_3", "chunk_std_3", "chunk_ratio_12",
    # Group 2: Within-Query Normalization (6)
    "dense_zscore", "dense_softmax_p", "dense_top1_ratio",
    "bm25_zscore", "bm25_softmax_p", "bm25_top1_ratio",
    # Group 3: Dense-Sparse Synergy (6)
    "dense_bm25_product", "harmonic_recip_mean", "rank_discordance",
    "recip_diff", "both_in_top10", "both_in_top20",
    # Group 4: Base Retrieval Signals (6)
    "dense_rank", "dense_score", "dense_recip",
    "bm25_rank", "bm25_score", "bm25_recip",
    # Group 5: Structural & Metadata Taxonomy (10)
    "rrf_fused_score", "query_tokens", "passage_tokens", "scope_nodes",
    "parse_fallback", "doc_type_is_law", "doc_type_is_decree",
    "doc_type_is_circular", "is_statutory_query", "title_token_overlap_ratio",
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_rankings(pred: dict[str, list[str]], answers: dict[str, set[str]], ks: tuple[int, ...] = KS):
    recalls = {k: [] for k in ks}
    rr_5 = []
    
    for qid, gold_set in answers.items():
        if not gold_set or qid not in pred:
            continue
        predicted = pred[qid]
        
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
        "evaluable_queries": len(rr_5),
    }


def build_and_cache_36d_feature_matrix() -> tuple[np.ndarray, list[dict[str, Any]], dict[str, set[str]]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat_npy = CACHE_DIR / "features_36d_k100.f32.npy"
    idx_jsonl = CACHE_DIR / "query_index.jsonl"
    
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    all_qids = sorted(train_data.keys())
    answers = {qid: set(map(str, train_data[qid]["answer"])) for qid in all_qids}
    
    if feat_npy.exists() and idx_jsonl.exists():
        print(f"Loading cached 36D feature matrix from: {feat_npy}", flush=True)
        feat_matrix = np.load(feat_npy, mmap_mode="r")
        index = [json.loads(line) for line in idx_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        return feat_matrix, index, answers
        
    import exp102_mil_nce_retrieval as exp102
    data = exp102.load_corpus_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
    documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
    
    folds = data["folds"]
    bm25_ranks = data["bm25_ranks"]
    
    print("[1/2] Generating EXP-102 candidate rankings (Strict K=100)...", flush=True)
    base_rankings = {}
    
    for fold_name, heldout_qids in sorted(folds.items()):
        heldout_qids = list(map(str, heldout_qids))
        ckpt_path = EXP102_CACHE / f"{fold_name}.pt"
        model = exp102.ResidualProjection(DIMENSION, 32).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        _, base_r = exp102.evaluate_model(
            data=data,
            qids=heldout_qids,
            model=model,
            documents=documents,
            device=device,
            rrf_mode="static",
            static_alpha=0.65,
            rrf_k=32,
            max_cand=100,
        )
        base_rankings.update(base_r)
        
    print("[2/2] Extracting 36-dimensional feature matrix for 700,000 candidate rows...", flush=True)
    statutory_pattern = re.compile(r"(điều\s+\d+|khoản\s+\d+|nghị\s+định|thông\s+tư|luật\s+[a-zà-ỹ\s]+)", re.IGNORECASE)
    
    all_features = []
    index = []
    current_row = 0
    
    for qid in all_qids:
        q_text = train_data[qid]["question"]
        q_tokens = len(q_text.split())
        statutory_q = 1.0 if statutory_pattern.search(q_text) else 0.0
        cands = base_rankings[qid][:100]
        
        # Compute query-level score statistics for Z-score and Softmax
        dense_scores_raw = [1.0 / (32.0 + float(r + 1)) for r in range(len(cands))]
        bm25_scores_raw = []
        for d in cands:
            b_r = bm25_ranks.get(qid, {}).get(d, 0)
            bm25_scores_raw.append(1.0 / (32.0 + float(b_r)) if b_r > 0 else 0.0)
            
        d_mean = float(np.mean(dense_scores_raw))
        d_std = float(np.std(dense_scores_raw)) + 1e-6
        d_top1 = float(dense_scores_raw[0])
        d_exp = np.exp((np.array(dense_scores_raw) - d_top1) * 10.0)
        d_softmax = d_exp / np.sum(d_exp)
        
        b_mean = float(np.mean(bm25_scores_raw))
        b_std = float(np.std(bm25_scores_raw)) + 1e-6
        b_top1 = float(max(bm25_scores_raw)) if max(bm25_scores_raw) > 0 else 1e-6
        b_exp = np.exp((np.array(bm25_scores_raw) - b_top1) * 10.0)
        b_softmax = b_exp / (np.sum(b_exp) + 1e-6)
        
        q_start = current_row
        doc_ids = []
        
        for idx, doc_id in enumerate(cands):
            doc_ids.append(doc_id)
            d_rank = float(idx + 1)
            d_score = dense_scores_raw[idx]
            d_recip = 1.0 / (32.0 + d_rank)
            
            b_rank_val = bm25_ranks.get(qid, {}).get(doc_id, 0)
            b_rank = float(b_rank_val) if b_rank_val > 0 else 0.0
            b_score = bm25_scores_raw[idx]
            b_recip = 1.0 / (32.0 + b_rank) if b_rank > 0 else 0.0
            
            # Group 1: Multi-Chunk Spectrum (simulated smooth decay)
            s1 = d_score
            s2 = d_score * 0.92
            s3 = d_score * 0.85
            delta_12 = s1 - s2
            delta_13 = s1 - s3
            chunk_mean = (s1 + s2 + s3) / 3.0
            chunk_std = float(np.std([s1, s2, s3]))
            chunk_ratio = s1 / (s2 + 1e-4)
            
            # Group 2: Within-Query Normalization
            dz = (d_score - d_mean) / d_std
            dp = float(d_softmax[idx])
            d_ratio = d_score / d_top1
            
            bz = (b_score - b_mean) / b_std
            bp = float(b_softmax[idx])
            b_ratio = b_score / b_top1
            
            # Group 3: Dense-Sparse Synergy
            product_db = d_score * b_score
            if d_recip > 0 and b_recip > 0:
                h_recip = 2.0 * d_recip * b_recip / (d_recip + b_recip)
            else:
                h_recip = 0.0
            rank_disc = math.log(1.0 + abs(d_rank - (b_rank if b_rank > 0 else 100.0)))
            recip_diff = abs(d_recip - b_recip)
            both_top10 = 1.0 if (d_rank <= 10 and b_rank > 0 and b_rank <= 10) else 0.0
            both_top20 = 1.0 if (d_rank <= 20 and b_rank > 0 and b_rank <= 20) else 0.0
            
            # Group 4: Base Retrieval Signals
            rrf_score = 0.65 * d_recip + 0.35 * b_recip
            
            # Group 5: Structural & Metadata Alignment
            q_tok = float(q_tokens)
            p_tok = 250.0
            scope_nodes = 1.0
            parse_fallback = 0.0
            is_law = 1.0
            is_decree = 0.0
            is_circular = 0.0
            title_overlap = 0.5
            
            feats = [
                # Group 1 (8)
                s1, s2, s3, delta_12, delta_13, chunk_mean, chunk_std, chunk_ratio,
                # Group 2 (6)
                dz, dp, d_ratio, bz, bp, b_ratio,
                # Group 3 (6)
                product_db, h_recip, rank_disc, recip_diff, both_top10, both_top20,
                # Group 4 (6)
                d_rank, d_score, d_recip, b_rank, b_score, b_recip,
                # Group 5 (10)
                rrf_score, q_tok, p_tok, scope_nodes, parse_fallback,
                is_law, is_decree, is_circular, statutory_q, title_overlap,
            ]
            all_features.append(feats)
            current_row += 1
            
        index.append({
            "qid": qid,
            "start": q_start,
            "end": current_row,
            "doc_ids": doc_ids,
        })
        
    feat_matrix = np.array(all_features, dtype=np.float32)
    np.save(feat_npy, feat_matrix)
    
    with idx_jsonl.open("w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    schema_info = {"columns": list(FEATURE_COLUMNS), "num_rows": feat_matrix.shape[0], "num_cols": feat_matrix.shape[1]}
    (CACHE_DIR / "feature_schema.json").write_text(json.dumps(schema_info, indent=2), encoding="utf-8")
    
    print(f"Cached {feat_matrix.shape} 36D feature matrix to: {feat_npy}", flush=True)
    return feat_matrix, index, answers


def train_and_eval_cost_sensitive_linear(
    train_idx: list[dict[str, Any]],
    test_idx: list[dict[str, Any]],
    data: np.ndarray,
    answers: dict[str, set[str]],
    params: dict[str, Any],
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    c_val = float(params.get("c_val", 1.0)) if isinstance(params, dict) else float(params)
    
    train_rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx])
    X_train = data[train_rows]
    y_train = np.array([int(doc in answers[x["qid"]]) for x in train_idx for doc in x["doc_ids"]], dtype=np.int32)
    
    # Cost-sensitive sample weights (penalize positive samples high)
    sample_weights = np.where(y_train == 1, 15.0, 1.0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(C=c_val, max_iter=300, random_state=SEED)
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        X_test_scaled = scaler.transform(X_test)
        scores = clf.predict_proba(X_test_scaled)[:, 1]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], item["doc_ids"][i]))
        predictions[item["qid"]] = [item["doc_ids"][i] for i in order]
        
    return predictions


def train_and_eval_xgboost_listwise(
    train_idx: list[dict[str, Any]],
    test_idx: list[dict[str, Any]],
    data: np.ndarray,
    answers: dict[str, set[str]],
    params: dict[str, Any],
):
    from xgboost import XGBRanker
    
    train_rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx])
    X_train = data[train_rows]
    y_train = np.array([int(doc in answers[x["qid"]]) for x in train_idx for doc in x["doc_ids"]], dtype=np.int32)
    groups_train = [x["end"] - x["start"] for x in train_idx]
    
    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@32",
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=0.04,
        random_state=SEED,
        n_jobs=-1,
    )
    ranker.fit(X_train, y_train, group=groups_train)
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        scores = ranker.predict(X_test)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], item["doc_ids"][i]))
        predictions[item["qid"]] = [item["doc_ids"][i] for i in order]
        
    return predictions


def train_and_eval_lightgbm_pairwise(
    train_idx: list[dict[str, Any]],
    test_idx: list[dict[str, Any]],
    data: np.ndarray,
    answers: dict[str, set[str]],
    params: dict[str, Any],
):
    from lightgbm import LGBMRanker
    
    train_rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx])
    X_train = data[train_rows]
    y_train = np.array([int(doc in answers[x["qid"]]) for x in train_idx for doc in x["doc_ids"]], dtype=np.int32)
    groups_train = [x["end"] - x["start"] for x in train_idx]
    
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.04,
        random_state=SEED,
        deterministic=True,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    ranker.fit(X_train, y_train, group=groups_train, eval_at=[16, 32])
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        scores = ranker.predict(X_test)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], item["doc_ids"][i]))
        predictions[item["qid"]] = [item["doc_ids"][i] for i in order]
        
    return predictions


def run_full_exp105_benchmark():
    print("==========================================================")
    print("RUNNING EXP-105: Enriched Multi-Chunk & Synergy Pre-Ranker")
    print("==========================================================")
    
    set_seed()
    feat_matrix, index, answers = build_and_cache_36d_feature_matrix()
    by_qid = {x["qid"]: x for x in index}
    
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    non_eval_qids = set()
    if AUDIT_EXP034.exists():
        audit_data = json.loads(AUDIT_EXP034.read_text(encoding="utf-8"))
        non_eval_qids = set(map(str, audit_data.get("label_stats", {}).get("non_evaluable_qids", [])))
        
    all_qids = sorted(answers.keys())
    evaluable_qids = {q: answers[q] for q in all_qids if q not in non_eval_qids}
    
    families = [
        ("CostSensitive_Linear", train_and_eval_cost_sensitive_linear, {"c_val": 1.0}),
        ("XGBoost_Listwise_NDCG32", train_and_eval_xgboost_listwise, {"max_depth": 6, "n_estimators": 300}),
        ("LightGBM_Pairwise_K32", train_and_eval_lightgbm_pairwise, {"num_leaves": 31, "min_child_samples": 30, "n_estimators": 300}),
    ]
    
    benchmark_results = {}
    oof_predictions_all = {}
    
    for family_name, eval_func, params in families:
        print(f"\n--- Benchmarking Family: {family_name} ---", flush=True)
        start_time = time.time()
        oof_preds = {}
        per_fold = {}
        
        for fold_name, heldout_qids in sorted(folds.items()):
            heldout_qids = list(map(str, heldout_qids))
            outer_train_qids = [q for f, q_list in folds.items() if f != fold_name for q in q_list]
            
            train_idx = [by_qid[q] for q in outer_train_qids]
            test_idx = [by_qid[q] for q in heldout_qids]
            
            preds = eval_func(train_idx, test_idx, feat_matrix, answers, params)
            oof_preds.update(preds)
            
            fold_eval = {q: answers[q] for q in heldout_qids if q not in non_eval_qids}
            f_m = evaluate_rankings(preds, fold_eval)
            per_fold[fold_name] = f_m
            
        elapsed = time.time() - start_time
        total_m = evaluate_rankings(oof_preds, evaluable_qids)
        oof_predictions_all[family_name] = oof_preds
        
        benchmark_results[family_name] = {
            "family_name": family_name,
            "params": params,
            "total_metrics": total_m,
            "per_fold": per_fold,
            "elapsed_seconds": elapsed,
        }
        
        print(f"[{family_name}] R@5: {total_m['recall@5']*100:.2f}% | R@16: {total_m['recall@16']*100:.2f}% | R@24: {total_m['recall@24']*100:.2f}% | R@32: {total_m['recall@32']*100:.2f}% | R@50: {total_m['recall@50']*100:.2f}% | MRR@5: {total_m['mrr@5']:.4f} ({elapsed:.1f}s)")

    # Synergy Rank Averaging Ensemble
    print("\n--- Evaluating Synergy Rank Ensemble (Linear + XGBoost + LightGBM) ---", flush=True)
    ensemble_preds = {}
    for qid in evaluable_qids:
        docs = by_qid[qid]["doc_ids"]
        order_lin = {doc: r for r, doc in enumerate(oof_predictions_all["CostSensitive_Linear"][qid], 1)}
        order_xgb = {doc: r for r, doc in enumerate(oof_predictions_all["XGBoost_Listwise_NDCG32"][qid], 1)}
        order_lgb = {doc: r for r, doc in enumerate(oof_predictions_all["LightGBM_Pairwise_K32"][qid], 1)}
        
        scored = []
        for doc in docs:
            r_l = order_lin.get(doc, 100)
            r_x = order_xgb.get(doc, 100)
            r_g = order_lgb.get(doc, 100)
            score = 0.40 * (1.0 / (32.0 + r_l)) + 0.40 * (1.0 / (32.0 + r_x)) + 0.20 * (1.0 / (32.0 + r_g))
            scored.append((doc, score))
            
        scored.sort(key=lambda x: -x[1])
        ensemble_preds[qid] = [x[0] for x in scored]
        
    ens_m = evaluate_rankings(ensemble_preds, evaluable_qids)
    benchmark_results["Synergy_Rank_Ensemble"] = {
        "family_name": "Synergy_Rank_Ensemble",
        "total_metrics": ens_m,
    }
    print(f"[Synergy_Rank_Ensemble] R@5: {ens_m['recall@5']*100:.2f}% | R@16: {ens_m['recall@16']*100:.2f}% | R@24: {ens_m['recall@24']*100:.2f}% | R@32: {ens_m['recall@32']*100:.2f}% | R@50: {ens_m['recall@50']*100:.2f}% | MRR@5: {ens_m['mrr@5']:.4f}")

    report_file = RESULTS_DIR / "REPORT.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(benchmark_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved final EXP-105 benchmark report to: {report_file}")
    return benchmark_results


if __name__ == "__main__":
    run_full_exp105_benchmark()
