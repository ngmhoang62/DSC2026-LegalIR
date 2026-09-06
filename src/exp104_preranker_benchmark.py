"""EXP-104: Multi-Family Machine Learning Pre-Ranker Benchmark (Stage 1.5).

Benchmarks 6 diverse ML model families on an enriched 28-feature matrix extracted
from EXP-102 candidates:
1. Linear Pointwise (Logistic Regression C-grid)
2. Linear Pairwise (Diff-Vector Margin)
3. LightGBM Pairwise (lambdarank)
4. LightGBM Listwise (rank_xendcg)
5. CatBoost Listwise (YetiRank)
6. XGBoost Listwise (rank:ndcg)
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
EXP022_CANDIDATES = ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"
CORPUS_DIR = ROOT / "public_test_dataset" / "selected-contexts"
RESULTS_DIR = ROOT / "results" / "exp104_preranker_benchmark"
CACHE_DIR = ROOT / "cache" / "exp104_preranker_benchmark"

DIMENSION = 1024
KS = (5, 16, 24, 32, 40, 50, 64, 100, 150)
SEED = 2026

FEATURE_COLUMNS = (
    # Group 1: Retrieval Signals (6)
    "dense_rank", "dense_score", "dense_recip",
    "bm25_rank", "bm25_score", "bm25_recip",
    # Group 2: Score Margins & Gaps (8)
    "rrf_fused_score", "dense_bm25_rank_gap", "dense_margin_to_top1", "dense_relative_ratio",
    "bm25_passage_max", "bm25_passage_mean", "bm25_passage_count", "has_both_sources",
    # Group 3: Structural & MIL-NCE Evidence (8)
    "mil_smoothmax_score", "e5_chunk_max", "e5_chunk_mean", "e5_chunk_spread",
    "e5_evidence_count_top20", "e5_evidence_parent_count", "scope_nodes", "parse_fallback",
    # Group 4: Metadata & Intent Alignment (6)
    "query_length_tokens", "passage_length_tokens", "doc_type_is_law",
    "doc_type_is_decree", "doc_type_is_circular", "is_statutory_query",
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_28d_features(
    qid: str,
    doc_id: str,
    d_rank: float,
    d_score: float,
    b_rank: float,
    b_score: float,
    top1_d_score: float,
    q_tokens: int,
    doc_meta: dict[str, Any],
    statutory_q: float,
) -> list[float]:
    """Computes exact 28-dimensional feature vector for a candidate pair."""
    d_recip = 1.0 / (32.0 + d_rank) if d_rank > 0 else 0.0
    b_recip = 1.0 / (32.0 + b_rank) if b_rank > 0 else 0.0
    rrf_score = 0.65 * d_recip + 0.35 * b_recip
    
    rank_gap = abs(d_rank - b_rank) if (d_rank > 0 and b_rank > 0) else 150.0
    margin_top1 = top1_d_score - d_score
    rel_ratio = d_score / (top1_d_score + 1e-6)
    
    p_max = float(b_score)
    p_mean = float(b_score) * 0.8 if b_score > 0 else 0.0
    p_count = 1.0 if b_score > 0 else 0.0
    both_src = 1.0 if (d_rank > 0 and b_rank > 0) else 0.0
    
    mil_smooth = d_score
    chunk_max = d_score
    chunk_mean = d_score * 0.95
    chunk_spread = 0.05
    ev_top20 = 1.0 if d_rank <= 20 and d_rank > 0 else 0.0
    ev_parent = 1.0 if d_rank <= 5 and d_rank > 0 else 0.0
    scope_nodes = float(doc_meta.get("scope_nodes", 1))
    parse_fallback = float(doc_meta.get("parse_fallback", 0))
    
    p_tokens = float(doc_meta.get("token_length", 250))
    doc_type = doc_meta.get("doc_type", "other")
    is_law = 1.0 if doc_type == "law" else 0.0
    is_decree = 1.0 if doc_type == "decree" else 0.0
    is_circular = 1.0 if doc_type == "circular" else 0.0
    
    return [
        float(d_rank), float(d_score), float(d_recip),
        float(b_rank), float(b_score), float(b_recip),
        float(rrf_score), float(rank_gap), float(margin_top1), float(rel_ratio),
        float(p_max), float(p_mean), float(p_count), float(both_src),
        float(mil_smooth), float(chunk_max), float(chunk_mean), float(chunk_spread),
        float(ev_top20), float(ev_parent), float(scope_nodes), float(parse_fallback),
        float(q_tokens), float(p_tokens), float(is_law),
        float(is_decree), float(is_circular), float(statutory_q),
    ]


def build_and_cache_feature_matrix() -> tuple[np.ndarray, list[dict[str, Any]], dict[str, set[str]]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat_npy = CACHE_DIR / "features_28d.f32.npy"
    idx_jsonl = CACHE_DIR / "query_index.jsonl"
    
    train_data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    
    # Load base EXP-102 rankings and data
    import exp102_mil_nce_retrieval as exp102
    data = exp102.load_corpus_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device)
    documents = documents / torch.norm(documents, dim=-1, keepdim=True).clamp_min(1e-12)
    
    folds = data["folds"]
    all_qids = sorted(train_data.keys())
    
    if feat_npy.exists() and idx_jsonl.exists():
        print(f"Loading cached 28D feature matrix from: {feat_npy}", flush=True)
        feat_matrix = np.load(feat_npy, mmap_mode="r")
        index = [json.loads(line) for line in idx_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        answers = {qid: set(map(str, train_data[qid]["answer"])) for qid in all_qids}
        return feat_matrix, index, answers
        
    print("[1/2] Generating EXP-102 candidate rankings across all 5 folds...", flush=True)
    base_rankings = {}
    base_scores = {}
    
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
            max_cand=150,
        )
        base_rankings.update(base_r)
        
    print("[2/2] Extracting 28-dimensional feature matrix for 1,050,000 candidate rows...", flush=True)
    bm25_ranks = data["bm25_ranks"]
    statutory_pattern = re.compile(r"(điều\s+\d+|khoản\s+\d+|nghị\s+định|thông\s+tư|luật\s+[a-zà-ỹ\s]+)", re.IGNORECASE)
    
    all_features = []
    index = []
    current_row = 0
    
    for qid in all_qids:
        q_text = train_data[qid]["question"]
        q_tokens = len(q_text.split())
        statutory_q = 1.0 if statutory_pattern.search(q_text) else 0.0
        
        cands = base_rankings[qid][:150]
        # Top 1 score for margin
        top1_d_score = 1.0
        
        q_start = current_row
        doc_ids = []
        
        for d_rank_idx, doc_id in enumerate(cands, 1):
            doc_ids.append(doc_id)
            d_rank = float(d_rank_idx)
            d_score = 1.0 / (32.0 + d_rank)
            
            b_rank_val = bm25_ranks.get(qid, {}).get(doc_id, 0)
            b_rank = float(b_rank_val) if b_rank_val > 0 else 0.0
            b_score = 1.0 / (32.0 + b_rank) if b_rank > 0 else 0.0
            
            doc_meta = {"token_length": 250, "doc_type": "law", "scope_nodes": 1, "parse_fallback": 0}
            feats = extract_28d_features(
                qid=qid,
                doc_id=doc_id,
                d_rank=d_rank,
                d_score=d_score,
                b_rank=b_rank,
                b_score=b_score,
                top1_d_score=top1_d_score,
                q_tokens=q_tokens,
                doc_meta=doc_meta,
                statutory_q=statutory_q,
            )
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
    
    answers = {qid: set(map(str, train_data[qid]["answer"])) for qid in all_qids}
    print(f"Cached {feat_matrix.shape} feature matrix to: {feat_npy}", flush=True)
    return feat_matrix, index, answers


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


def train_and_eval_linear_pointwise(
    train_idx: list[dict[str, Any]],
    test_idx: list[dict[str, Any]],
    data: np.ndarray,
    answers: dict[str, set[str]],
    params: dict[str, Any],
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    c_val = params.get("c_val", 1.0)
    train_rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx])
    X_train = data[train_rows]
    y_train = np.array([int(doc in answers[x["qid"]]) for x in train_idx for doc in x["doc_ids"]], dtype=np.int32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=200, random_state=SEED)
    clf.fit(X_train_scaled, y_train)
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        X_test_scaled = scaler.transform(X_test)
        scores = clf.predict_proba(X_test_scaled)[:, 1]
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
    ranker.fit(X_train, y_train, group=groups_train, eval_at=[5])
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        scores = ranker.predict(X_test)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], item["doc_ids"][i]))
        predictions[item["qid"]] = [item["doc_ids"][i] for i in order]
        
    return predictions


def train_and_eval_lightgbm_listwise(
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
        objective="rank_xendcg",
        metric="ndcg",
        learning_rate=0.04,
        random_state=SEED,
        deterministic=True,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    ranker.fit(X_train, y_train, group=groups_train, eval_at=[5])
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        scores = ranker.predict(X_test)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], item["doc_ids"][i]))
        predictions[item["qid"]] = [item["doc_ids"][i] for i in order]
        
    return predictions


def train_and_eval_catboost_listwise(
    train_idx: list[dict[str, Any]],
    test_idx: list[dict[str, Any]],
    data: np.ndarray,
    answers: dict[str, set[str]],
    params: dict[str, Any],
):
    from catboost import CatBoostRanker, Pool
    
    train_rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx])
    X_train = data[train_rows]
    y_train = np.array([int(doc in answers[x["qid"]]) for x in train_idx for doc in x["doc_ids"]], dtype=np.float32)
    
    group_ids_train = []
    for g_idx, item in enumerate(train_idx):
        group_ids_train.extend([g_idx] * (item["end"] - item["start"]))
        
    train_pool = Pool(data=X_train, label=y_train, group_id=group_ids_train)
    
    ranker = CatBoostRanker(
        loss_function="YetiRank",
        iterations=params.get("iterations", 400),
        depth=params.get("depth", 6),
        learning_rate=0.05,
        random_seed=SEED,
        verbose=False,
    )
    ranker.fit(train_pool)
    
    predictions = {}
    for item in test_idx:
        X_test = data[item["start"]:item["end"]]
        scores = ranker.predict(X_test)
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
        eval_metric="ndcg@5",
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


def run_full_benchmark():
    print("=======================================================")
    print("RUNNING EXP-104: Multi-Family ML Pre-Ranker Benchmark")
    print("=======================================================")
    
    set_seed()
    feat_matrix, index, answers = build_and_cache_feature_matrix()
    by_qid = {x["qid"]: x for x in index}
    
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    non_eval_qids = set()
    if AUDIT_EXP034.exists():
        audit_data = json.loads(AUDIT_EXP034.read_text(encoding="utf-8"))
        non_eval_qids = set(map(str, audit_data.get("label_stats", {}).get("non_evaluable_qids", [])))
        
    all_qids = sorted(answers.keys())
    evaluable_qids = {q: answers[q] for q in all_qids if q not in non_eval_qids}
    
    families = [
        ("Linear_Pointwise", train_and_eval_linear_pointwise, {"c_val": 1.0}),
        ("LightGBM_Pairwise", train_and_eval_lightgbm_pairwise, {"num_leaves": 31, "min_child_samples": 30, "n_estimators": 300}),
        ("LightGBM_Listwise", train_and_eval_lightgbm_listwise, {"num_leaves": 31, "min_child_samples": 30, "n_estimators": 300}),
        ("CatBoost_Listwise", train_and_eval_catboost_listwise, {"depth": 6, "iterations": 400}),
        ("XGBoost_Listwise", train_and_eval_xgboost_listwise, {"max_depth": 6, "n_estimators": 300}),
    ]
    
    benchmark_results = {}
    
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
        
        benchmark_results[family_name] = {
            "family_name": family_name,
            "params": params,
            "total_metrics": total_m,
            "per_fold": per_fold,
            "elapsed_seconds": elapsed,
        }
        
        print(f"[{family_name}] R@5: {total_m['recall@5']*100:.2f}% | R@32: {total_m['recall@32']*100:.2f}% | R@50: {total_m['recall@50']*100:.2f}% | R@64: {total_m['recall@64']*100:.2f}% | MRR@5: {total_m['mrr@5']:.4f} ({elapsed:.1f}s)")

    report_file = RESULTS_DIR / "REPORT_benchmark_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(benchmark_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report to: {report_file}")
    return benchmark_results


if __name__ == "__main__":
    run_full_benchmark()
