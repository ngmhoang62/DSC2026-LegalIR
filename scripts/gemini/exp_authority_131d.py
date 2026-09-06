"""Statutory Authority and Legal Hierarchy Aware Feature Fusion (131D LTR).

Hypothesis H6:
Adding 12 statutory authority and hierarchy-aware features
(statute level, authority type matching, mismatch penalty, title specificity overlap)
to the 119D unified feature space directly resolves the General Law vs Specific Decree
boundary conflict (ranks 6-10), advancing 5-fold OOF Recall@5 toward the 0.96 target.

Zero label leakage:
Authority features depend exclusively on public document metadata and query text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.authority import AuthorityExtractor
from exp_final.data import SourceStore

CACHE_119 = ROOT / "cache/gemini/exp_unified_ltr"
CACHE_131 = ROOT / "cache/gemini/exp_authority_131d"
OUT_DIR = ROOT / "results/gemini/exp_authority_131d"
LGBM_119_DIR = ROOT / "results/gemini/exp_unified_ltr"
XGB_119_DIR = ROOT / "results/gemini/exp_unified_xgb"
PROFILE_DIR = ROOT / "results/exp_final_retrieval/profile_ltr_probe"
MEMORY_DIR = ROOT / "results/exp_final_retrieval/memory_ltr_probe"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_fold(outer: int, auth: AuthorityExtractor, questions: dict[str, str]) -> dict[str, Any]:
    print(f"\n=================== STARTING 131D FOLD {outer} ===================", flush=True)
    fold_cache = CACHE_131 / f"fold_{outer}"
    fold_out = OUT_DIR / f"fold_{outer}"
    fold_cache.mkdir(parents=True, exist_ok=True)
    fold_out.mkdir(parents=True, exist_ok=True)

    labels, _ = get_canonical_labels()
    folds = get_cv_folds()

    store = SourceStore(ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite")
    marker = read_json(ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{outer}/outer-ml.json")
    train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
    test_qids = [str(q) for q in folds[f"fold_{outer}"] if labels.get(str(q))]

    # 1. Prepare / Load 131D Training Matrix
    train_131_path = fold_cache / "train_131d.f32.npy"
    train_119_path = CACHE_119 / f"fold_{outer}/train_unified.f32.npy"
    
    train_docs_list = []
    groups = []
    target = []
    for q in train_qids:
        docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
        groups.append(len(docs))
        target.extend(d in labels[q] for d in docs)
        train_docs_list.append(docs)
    y_train = np.asarray(target, dtype=np.int8)

    if not train_131_path.exists():
        print(f"Extracting 12D authority features for {len(train_qids)} train queries...", flush=True)
        t0 = time.time()
        train_auth = auth.extract_block(questions, train_qids, train_docs_list)
        print(f"  Extracted train authority in {time.time()-t0:.1f}s, shape={train_auth.shape}", flush=True)
        
        train_119 = np.load(train_119_path, mmap_mode="r")
        assert train_119.shape[0] == train_auth.shape[0]
        assert train_119.shape[1] == 119 and train_auth.shape[1] == 12
        
        print("Assembling 131D train matrix memmap...", flush=True)
        t0 = time.time()
        mmap_train = np.lib.format.open_memmap(
            train_131_path, mode="w+", dtype=np.float32, shape=(train_119.shape[0], 131)
        )
        mmap_train[:, :119] = train_119
        mmap_train[:, 119:] = train_auth
        mmap_train.flush()
        print(f"  Saved 131D train matrix in {time.time()-t0:.1f}s", flush=True)
        train_131 = np.load(train_131_path, mmap_mode="r")
    else:
        print("Reusing existing 131D train matrix.", flush=True)
        train_131 = np.load(train_131_path, mmap_mode="r")

    # 2. Prepare / Load 131D Test Matrix
    test_131_path = fold_cache / "test_131d.f32.npy"
    test_119_path = CACHE_119 / f"fold_{outer}/test_matrix.f32.npy"
    test_docs = read_json(CACHE_119 / f"fold_{outer}/test_docs.json")
    test_groups = read_json(CACHE_119 / f"fold_{outer}/test_groups.json")
    test_ends = np.cumsum([0] + test_groups)

    if not test_131_path.exists():
        print(f"Extracting 12D authority features for {len(test_qids)} test queries...", flush=True)
        t0 = time.time()
        test_auth = auth.extract_block(questions, test_qids, test_docs)
        print(f"  Extracted test authority in {time.time()-t0:.1f}s, shape={test_auth.shape}", flush=True)
        
        test_119 = np.load(test_119_path)
        assert test_119.shape[0] == test_auth.shape[0]
        test_131 = np.concatenate([test_119, test_auth], axis=1)
        np.save(test_131_path, test_131)
        print(f"  Saved 131D test matrix, shape={test_131.shape}", flush=True)
    else:
        print("Reusing existing 131D test matrix.", flush=True)
        test_131 = np.load(test_131_path)

    store.close()

    fold_results = {}

    # Model A: XGBRanker on 131D
    print(f"\nFitting 131D XGBRanker on Fold {outer} (GPU)...", flush=True)
    t0 = time.time()
    xgb_model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        random_state=4200 + outer,
        n_jobs=4,
        tree_method="hist",
        device="cuda",
    )
    xgb_model.fit(train_131, y_train, group=groups)
    fit_time_xgb = time.time() - t0
    print(f"  Fitted in {fit_time_xgb:.1f}s", flush=True)

    s_xgb = xgb_model.predict(test_131)
    preds_xgb = {}
    for idx, (qid, docs) in enumerate(zip(test_qids, test_docs)):
        vals = s_xgb[test_ends[idx]:test_ends[idx + 1]]
        preds_xgb[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(vals[i]), docs[i]))]

    m_xgb = compute_metrics(preds_xgb, labels, test_qids)
    print(f"  Fold {outer} [131D XGBRanker]: Recall@5 = {m_xgb['recall_at_5']:.6f}, "
          f"Precision@5 = {m_xgb['precision_at_5']:.6f}, MRR@5 = {m_xgb['mrr_at_5']:.6f}", flush=True)
    write_json(fold_out / "xgb_131d_PREDICTIONS.json", preds_xgb)

    # Model B: LGBMRanker on 131D
    print(f"\nFitting 131D LGBMRanker on Fold {outer}...", flush=True)
    t0 = time.time()
    lgb_model = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=15,
        min_child_samples=50,
        lambdarank_truncation_level=5,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        random_state=4200 + outer,
        verbosity=-1,
    )
    lgb_model.fit(train_131, y_train, group=groups, eval_at=[5])
    fit_time_lgb = time.time() - t0
    print(f"  Fitted in {fit_time_lgb:.1f}s", flush=True)

    s_lgb = lgb_model.predict(test_131)
    preds_lgb = {}
    for idx, (qid, docs) in enumerate(zip(test_qids, test_docs)):
        vals = s_lgb[test_ends[idx]:test_ends[idx + 1]]
        preds_lgb[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(vals[i]), docs[i]))]

    m_lgb = compute_metrics(preds_lgb, labels, test_qids)
    print(f"  Fold {outer} [131D LGBMRanker]: Recall@5 = {m_lgb['recall_at_5']:.6f}, "
          f"Precision@5 = {m_lgb['precision_at_5']:.6f}, MRR@5 = {m_lgb['mrr_at_5']:.6f}", flush=True)
    write_json(fold_out / "lgbm_131d_PREDICTIONS.json", preds_lgb)

    fold_results["xgb_131d"] = {"metrics": m_xgb, "fit_time": fit_time_xgb}
    fold_results["lgbm_131d"] = {"metrics": m_lgb, "fit_time": fit_time_lgb}
    write_json(fold_out / "FOLD_REPORT.json", fold_results)
    return fold_results


def evaluate_oof():
    print("\n=================== COMPUTING FULL 5-FOLD OOF METRICS (131D) ===================", flush=True)
    labels, _ = get_canonical_labels()
    folds = get_cv_folds()
    all_eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]

    # Load baselines
    mem_preds = {}
    prof_preds = {}
    lgbm_119_preds = read_json(LGBM_119_DIR / "l15_t5_OOF_PREDICTIONS.json")
    xgb_119_preds = read_json(XGB_119_DIR / "xgb_d4_n400_OOF_PREDICTIONS.json")
    best_blend_119 = read_json(ROOT / "results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json")

    for f in range(5):
        mem_preds.update(read_json(MEMORY_DIR / f"fold_{f}/PREDICTIONS.json"))
        prof_preds.update(read_json(PROFILE_DIR / "l15_t5/PREDICTIONS.json"))

    m_prof = compute_metrics(prof_preds, labels, all_eval_qids)
    m_mem = compute_metrics(mem_preds, labels, all_eval_qids)
    m_lgbm_119 = compute_metrics(lgbm_119_preds, labels, all_eval_qids)
    m_xgb_119 = compute_metrics(xgb_119_preds, labels, all_eval_qids)
    m_blend_119 = compute_metrics(best_blend_119, labels, all_eval_qids)

    # Pool 131D predictions
    xgb_131_pooled = {}
    lgbm_131_pooled = {}
    for f in range(5):
        xgb_131_pooled.update(read_json(OUT_DIR / f"fold_{f}/xgb_131d_PREDICTIONS.json"))
        lgbm_131_pooled.update(read_json(OUT_DIR / f"fold_{f}/lgbm_131d_PREDICTIONS.json"))

    write_json(OUT_DIR / "xgb_131d_OOF_PREDICTIONS.json", xgb_131_pooled)
    write_json(OUT_DIR / "lgbm_131d_OOF_PREDICTIONS.json", lgbm_131_pooled)

    m_xgb_131 = compute_metrics(xgb_131_pooled, labels, all_eval_qids)
    m_lgbm_131 = compute_metrics(lgbm_131_pooled, labels, all_eval_qids)

    print("\n--- STANDALONE 131D SYSTEMS ---")
    print(f"131D XGBRanker OOF: Recall@5={m_xgb_131['recall_at_5']:.6f} "
          f"(119D XGB: {m_xgb_119['recall_at_5']:.6f}, Delta: {m_xgb_131['recall_at_5']-m_xgb_119['recall_at_5']:+.6f}), "
          f"P@5={m_xgb_131['precision_at_5']:.6f}, MRR@5={m_xgb_131['mrr_at_5']:.6f}")
    print(f"131D LGBMRanker OOF: Recall@5={m_lgbm_131['recall_at_5']:.6f} "
          f"(119D LGBM: {m_lgbm_119['recall_at_5']:.6f}, Delta: {m_lgbm_131['recall_at_5']-m_lgbm_119['recall_at_5']:+.6f}), "
          f"P@5={m_lgbm_131['precision_at_5']:.6f}, MRR@5={m_lgbm_131['mrr_at_5']:.6f}")

    # Evaluate 131D Multi-Architectural Blends
    print("\n--- 131D MULTI-ARCHITECTURAL BLENDS ---")
    k = 8
    blend_configs = [
        # (w_xgb_131, w_lgbm_131, w_prof, w_lgbm_119, name)
        (0.35, 0.35, 0.30, 0.00, "0.35_xgb131 + 0.35_lgb131 + 0.30_prof"),
        (0.40, 0.35, 0.25, 0.00, "0.40_xgb131 + 0.35_lgb131 + 0.25_prof"),
        (0.30, 0.30, 0.20, 0.20, "0.30_xgb131 + 0.30_lgb131 + 0.20_prof + 0.20_lgb119"),
        (0.50, 0.00, 0.25, 0.25, "0.50_xgb131 + 0.25_prof + 0.25_lgb119"),
        (0.30, 0.40, 0.30, 0.00, "0.30_xgb131 + 0.40_lgb131 + 0.30_prof"),
    ]

    best_blend_r5 = -1.0
    best_blend_name = None
    best_blend_preds = None

    summary = {
        "status": "COMPLETE_131D_AUTHORITY_OOF",
        "feature_count": 131,
        "systems": {
            "xgb_131d": m_xgb_131,
            "lgbm_131d": m_lgbm_131,
        },
        "blends": {},
    }

    for wx, wl, wp, wl119, bname in blend_configs:
        b_preds = {}
        for q in all_eval_qids:
            sc = {}
            if wx > 0:
                for r, d in enumerate(xgb_131_pooled[q][:64], 1): sc[d] = sc.get(d, 0.0) + wx / (k + r)
            if wl > 0:
                for r, d in enumerate(lgbm_131_pooled[q][:64], 1): sc[d] = sc.get(d, 0.0) + wl / (k + r)
            if wp > 0:
                for r, d in enumerate(prof_preds[q][:64], 1): sc[d] = sc.get(d, 0.0) + wp / (k + r)
            if wl119 > 0:
                for r, d in enumerate(lgbm_119_preds[q][:64], 1): sc[d] = sc.get(d, 0.0) + wl119 / (k + r)
            b_preds[q] = sorted(sc.keys(), key=lambda d: (-sc[d], d))
        
        m_b = compute_metrics(b_preds, labels, all_eval_qids)
        boot = paired_bootstrap(best_blend_119, b_preds, labels, all_eval_qids)
        delta_vs_119_best = m_b["recall_at_5"] - m_blend_119["recall_at_5"]
        print(f"  [{bname}]: Recall@5={m_b['recall_at_5']:.6f} (Delta vs Best 119D: {delta_vs_119_best:+.6f}), "
              f"P@5={m_b['precision_at_5']:.6f}, MRR@5={m_b['mrr_at_5']:.6f}, p={boot['p_value']:.4f}, W/L/T={boot['wins']}/{boot['losses']}/{boot['ties']}")

        summary["blends"][bname] = {
            "metrics": m_b,
            "delta_vs_best_119d": delta_vs_119_best,
            "bootstrap_vs_best_119d": boot,
        }

        if m_b["recall_at_5"] > best_blend_r5:
            best_blend_r5 = m_b["recall_at_5"]
            best_blend_name = bname
            best_blend_preds = b_preds

    write_json(OUT_DIR / "AUTHORITY_131D_SUMMARY.json", summary)
    print(f"\nSaved complete summary to {OUT_DIR / 'AUTHORITY_131D_SUMMARY.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run Gemini 131D Authority-Aware LTR")
    parser.add_argument("--fold", default="all", choices=["0", "1", "2", "3", "4", "all"])
    args = parser.parse_args()

    train_data = read_json(ROOT / "public_test_dataset/train.json")
    questions = {str(qid): item["question"] for qid, item in train_data.items()}
    auth = AuthorityExtractor()

    folds_to_run = range(5) if args.fold == "all" else [int(args.fold)]
    for f in folds_to_run:
        run_fold(f, auth, questions)

    if len(folds_to_run) == 5:
        evaluate_oof()


if __name__ == "__main__":
    main()
