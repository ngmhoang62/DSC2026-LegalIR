"""Hypothesis H66: Deeper Feature-Subsampled XGBoost (XGB-145D-D5) 5-Fold Training & Nested CV.

Upgrades Model A in the SOTA ensemble:
- max_depth: 4 -> 5 (32 leaves, +100% capacity)
- colsample_bytree: 1.0 -> 0.8 (feature subsampling for diverse splits)
- subsample: 1.0 -> 0.85 (stochastic row subsampling)
- n_estimators: 450 -> 500
- learning_rate: 0.04 -> 0.035
- tree_method: hist, device: cuda

Evaluates:
1. Standalone 5-fold OOF performance.
2. Nested CV Multi-Model AMFD Fusion + Unsupervised Statutory Kinship Suite.
3. Paired bootstrap vs honest SOTA baseline (0.955042) and Profile LTR anchor (0.946448).
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("D:/Study/DSC2026/LegalIR")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.kinship import (
    apply_kinship_promotion,
    apply_multi_statute_promotion,
    apply_inverse_kinship_promotion,
    apply_deep_statutory_kinship,
    apply_guarded_inverse_law,
    apply_topic_law_promotion,
    apply_preamble_citation_kinship,
    apply_hierarchical_midrank_inverse_kinship,
    apply_technical_standard_kinship,
    apply_corporate_entity_kinship,
    apply_superseded_statute_dedup,
    load_doc_labels,
    VERIFIED_SUPERSEDED_STATUTE_PAIRS,
)
from exp_final.data import SourceStore

OUT_DIR = ROOT / "results/gemini/exp_xgb_145d_d5"
SOURCES_DB = ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
QUERY_ROWS_PATH = ROOT / "cache/exp012b_v3/rankings/train/query_rows.jsonl"
DOC_PREAMBLES_PATH = ROOT / "cache/gemini/doc_preambles.json"

def main():
    print("=" * 80)
    print("HYPOTHESIS H66: DEEPER FEATURE-SUBSAMPLED XGB-145D-D5 ENSEMBLE")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, _ = get_canonical_labels()
    folds = get_cv_folds()
    eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]
    all_7000_qids = [q for f in range(5) for q in folds[f"fold_{f}"]]

    store = SourceStore(SOURCES_DB)
    doc_labels = load_doc_labels(EVIDENCE_DB)
    questions = {}
    if QUERY_ROWS_PATH.exists():
        with open(QUERY_ROWS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                questions[str(item.get("query_id") or item.get("id") or item.get("qid"))] = (
                    item.get("text") or item.get("query") or ""
                )
    doc_preambles = json.loads(DOC_PREAMBLES_PATH.read_text(encoding="utf-8")) if DOC_PREAMBLES_PATH.exists() else {}

    # 1. Train and Predict XGB-145D-D5 across all 5 folds
    xgb_d5_oof = {}
    standalone_fold_reports = []

    print("\n--- Training XGB-145D-D5 Across All 5 Folds (GPU) ---", flush=True)
    total_train_time = 0.0

    for f in range(5):
        fold_out = OUT_DIR / f"fold_{f}"
        fold_out.mkdir(parents=True, exist_ok=True)
        pred_path = fold_out / "xgb_145d_d5_PREDICTIONS.json"

        f_eval_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
        test_docs = json.loads((ROOT / f"cache/gemini/exp_unified_ltr/fold_{f}/test_docs.json").read_text(encoding="utf-8"))
        test_groups = json.loads((ROOT / f"cache/gemini/exp_unified_ltr/fold_{f}/test_groups.json").read_text(encoding="utf-8"))
        test_ends = np.cumsum([0] + test_groups)

        if pred_path.exists():
            print(f"Fold {f}: Loading existing predictions from {pred_path}...", flush=True)
            f_preds = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            train_145 = np.load(ROOT / f"cache/gemini/exp_145d_ranker/fold_{f}/train_145d.f32.npy", mmap_mode="r")
            test_145 = np.load(ROOT / f"cache/gemini/exp_145d_ranker/fold_{f}/test_145d.f32.npy")

            marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
            train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]

            train_groups = []
            train_y = []
            for q in train_qids:
                docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
                train_groups.append(len(docs))
                train_y.extend(d in labels[q] for d in docs)
            y_train = np.asarray(train_y, dtype=np.int8)

            print(f"Fold {f}: Fitting XGB-145D-D5 on {len(y_train)} rows ({len(train_groups)} queries)...", flush=True)
            model = xgb.XGBRanker(
                objective="rank:ndcg",
                eval_metric="ndcg@5",
                n_estimators=500,
                learning_rate=0.035,
                max_depth=5,
                colsample_bytree=0.8,
                subsample=0.85,
                random_state=4200 + f,
                n_jobs=4,
                tree_method="hist",
                device="cuda"
            )
            model.fit(train_145, y_train, group=train_groups)
            fit_time = time.time() - t0
            total_train_time += fit_time
            print(f"  Fold {f} fit completed in {fit_time:.1f}s", flush=True)

            scores = model.predict(test_145)
            f_preds = {}
            for idx, q in enumerate(f_eval_qids):
                b, e = test_ends[idx], test_ends[idx + 1]
                docs = test_docs[idx]
                s = scores[b:e]
                f_preds[q] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(s[i]), docs[i]))]

            pred_path.write_text(json.dumps(f_preds, ensure_ascii=False, indent=2), encoding="utf-8")

        xgb_d5_oof.update(f_preds)
        m_f = compute_metrics(f_preds, labels, f_eval_qids)
        standalone_fold_reports.append(m_f)
        print(f"Fold {f} Standalone: Recall@5 = {m_f['recall_at_5']:.6f} | Prec@5 = {m_f['precision_at_5']:.6f} | MRR@5 = {m_f['mrr_at_5']:.6f}", flush=True)

    print(f"\nTotal training time for 5 folds: {total_train_time:.1f}s")
    m_d5_standalone = compute_metrics(xgb_d5_oof, labels, eval_qids)
    print("\n" + "=" * 80)
    print("XGB-145D-D5 5-FOLD STANDALONE OOF PERFORMANCE:")
    print("=" * 80)
    print(f"Recall@5:       {m_d5_standalone['recall_at_5']:.6f}")
    print(f"Precision@5:    {m_d5_standalone['precision_at_5']:.6f}")
    print(f"MRR@5:          {m_d5_standalone['mrr_at_5']:.6f}")
    print(f"Multi-Gold R@5: {m_d5_standalone['multi_gold_recall_at_5']:.6f}")
    print("=" * 80)

    # 2. Load complementary models for Nested Cross-Validation Fusion
    print("\nLoading ensemble models for Nested CV Fusion...", flush=True)
    xgb_tuned_base = {}
    for f in range(5):
        p = json.loads((ROOT / f"results/gemini/exp_145d_tuned/fold_{f}/xgb_145d_tuned_PREDICTIONS.json").read_text(encoding="utf-8"))
        xgb_tuned_base.update(p)

    xgb131 = json.loads((ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    lgb145 = json.loads((ROOT / "results/gemini/exp_145d_ranker/lgbm_145d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    prof = json.loads((ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json").read_text(encoding="utf-8"))

    # Candidate configurations for Nested CV:
    # Option A: Replace xgb_base with xgb_d5
    # Option B: Dual XGBoost ensemble (xgb_base + xgb_d5 + lgb145 + xgb131 + prof)
    CANDIDATE_CONFIGS = [
        # Type A: Deeper XGB in place of base XGB (w_d5, w_lgb, w_131, w_prof, k_d5, k_lgb, k_131, k_prof)
        ("replace_d5_amfd", 0.41, 0.29, 0.12, 0.18, 18, 10, 15, 15),
        ("replace_d5_k15",  0.41, 0.29, 0.12, 0.18, 15, 15, 15, 15),
        ("replace_d5_k20",  0.43, 0.27, 0.12, 0.18, 20, 10, 15, 15),
        # Type B: Dual XGB (w_base, w_d5, w_lgb, w_131, w_prof)
        ("dual_xgb_equal",  0.22, 0.22, 0.28, 0.11, 0.17, 18, 18, 10, 15, 15),
        ("dual_xgb_d5heavy",0.18, 0.26, 0.27, 0.11, 0.18, 16, 18, 10, 15, 15),
    ]

    def blend_cfg(qids, cfg):
        cfg_name = cfg[0]
        blended = {}
        if cfg_name.startswith("replace_d5"):
            _, w_d5, w_lgb, w_131, w_prof, k_d5, k_lgb, k_131, k_prof = cfg
            for q in qids:
                sc = {}
                for r, d in enumerate(xgb_d5_oof.get(q, [])[:k_d5]):  sc[d] = sc.get(d, 0.0) + w_d5 / (r + 1.0)
                for r, d in enumerate(lgb145.get(q, [])[:k_lgb]):     sc[d] = sc.get(d, 0.0) + w_lgb / (r + 1.0)
                for r, d in enumerate(xgb131.get(q, [])[:k_131]):     sc[d] = sc.get(d, 0.0) + w_131 / (r + 1.0)
                for r, d in enumerate(prof.get(q, [])[:k_prof]):       sc[d] = sc.get(d, 0.0) + w_prof / (r + 1.0)
                cand = dict.fromkeys(
                    xgb_d5_oof.get(q, [])[:k_d5] + lgb145.get(q, [])[:k_lgb] + xgb131.get(q, [])[:k_131] + prof.get(q, [])[:k_prof]
                )
                blended[q] = sorted(cand.keys(), key=lambda d: (-sc.get(d, 0.0), d))
        else:
            _, w_base, w_d5, w_lgb, w_131, w_prof, k_b, k_d5, k_lgb, k_131, k_prof = cfg
            for q in qids:
                sc = {}
                for r, d in enumerate(xgb_tuned_base.get(q, [])[:k_b]): sc[d] = sc.get(d, 0.0) + w_base / (r + 1.0)
                for r, d in enumerate(xgb_d5_oof.get(q, [])[:k_d5]):   sc[d] = sc.get(d, 0.0) + w_d5 / (r + 1.0)
                for r, d in enumerate(lgb145.get(q, [])[:k_lgb]):      sc[d] = sc.get(d, 0.0) + w_lgb / (r + 1.0)
                for r, d in enumerate(xgb131.get(q, [])[:k_131]):      sc[d] = sc.get(d, 0.0) + w_131 / (r + 1.0)
                for r, d in enumerate(prof.get(q, [])[:k_prof]):        sc[d] = sc.get(d, 0.0) + w_prof / (r + 1.0)
                cand = dict.fromkeys(
                    xgb_tuned_base.get(q, [])[:k_b] + xgb_d5_oof.get(q, [])[:k_d5] + lgb145.get(q, [])[:k_lgb] + xgb131.get(q, [])[:k_131] + prof.get(q, [])[:k_prof]
                )
                blended[q] = sorted(cand.keys(), key=lambda d: (-sc.get(d, 0.0), d))
        return blended

    def apply_unsupervised_kinship(rankings, qids):
        f1, _ = apply_kinship_promotion(rankings, doc_labels, qids, top_k=2, cand_max=9)
        f2, _ = apply_multi_statute_promotion(f1, doc_labels, questions, qids)
        f3, _ = apply_inverse_kinship_promotion(f2, doc_labels, qids, top_k=2, cand_max=9)
        f4, _ = apply_deep_statutory_kinship(f3, doc_labels, qids, top_k=2, cand_max=15)
        f5, _ = apply_guarded_inverse_law(f4, doc_labels, qids, top_k=2, cand_max=12)
        f6, _ = apply_topic_law_promotion(f5, doc_labels, questions, qids, cand_max=6)
        f7, _ = apply_preamble_citation_kinship(f6, doc_labels, doc_preambles, qids, top_k=2, cand_max=8)
        f8, _ = apply_hierarchical_midrank_inverse_kinship(f7, doc_labels, qids, cand_max=12)
        f9, _ = apply_technical_standard_kinship(f8, doc_labels, qids, cand_max=10)
        fin, _ = apply_corporate_entity_kinship(f9, doc_labels, questions, qids)
        return fin

    # Strict Nested Cross-Validation Loop
    print("\n" + "=" * 80)
    print("STRICT NESTED CROSS-VALIDATION AUDIT")
    print("=" * 80)
    nested_oof_preds = {}
    nested_fold_reports = []

    for f_outer in range(5):
        outer_qids = [q for q in folds[f"fold_{f_outer}"] if labels.get(q)]
        inner_qids = [q for f in range(5) if f != f_outer for q in folds[f"fold_{f}"] if labels.get(q)]

        # Select Best Config on Inner Folds ONLY
        best_cfg = None
        best_inner_r5 = -1.0
        for cfg in CANDIDATE_CONFIGS:
            inner_blend = blend_cfg(inner_qids, cfg)
            inner_kin = apply_unsupervised_kinship(inner_blend, inner_qids)
            m = compute_metrics(inner_kin, labels, inner_qids)
            if m["recall_at_5"] > best_inner_r5:
                best_inner_r5 = m["recall_at_5"]
                best_cfg = cfg

        print(f"\nOuter Fold {f_outer}: Best Inner Config = {best_cfg[0]} (Inner R@5 = {best_inner_r5:.6f})")

        # Cross-fit superseded statute pairs on inner folds
        inner_blend = blend_cfg(inner_qids, best_cfg)
        inner_kin = apply_unsupervised_kinship(inner_blend, inner_qids)
        cross_fitted_pairs = []
        for old_id, new_id in VERIFIED_SUPERSEDED_STATUTE_PAIRS:
            both_in_top5 = sum(1 for q in inner_qids if old_id in inner_kin[q][:5] and new_id in inner_kin[q][:5])
            old_is_gold = sum(1 for q in inner_qids if old_id in inner_kin[q][:5] and new_id in inner_kin[q][:5] and old_id in labels[q])
            is_primary = (old_id, new_id) in VERIFIED_SUPERSEDED_STATUTE_PAIRS[:11]
            if (both_in_top5 >= 1 and old_is_gold == 0) or is_primary:
                cross_fitted_pairs.append((old_id, new_id))

        print(f"  Cross-fitted superseded pairs: {len(cross_fitted_pairs)}")

        # Evaluate on Outer Fold
        outer_blend = blend_cfg(outer_qids, best_cfg)
        outer_kin = apply_unsupervised_kinship(outer_blend, outer_qids)
        outer_dedup, dedup_cnt = apply_superseded_statute_dedup(outer_kin, outer_qids, superseded_pairs=cross_fitted_pairs)

        nested_oof_preds.update(outer_dedup)
        m_outer = compute_metrics(outer_dedup, labels, outer_qids)
        nested_fold_reports.append(m_outer)
        print(f"  Outer Fold {f_outer} STRICT-VALID: Recall@5 = {m_outer['recall_at_5']:.6f} | Prec@5 = {m_outer['precision_at_5']:.6f} | MRR@5 = {m_outer['mrr_at_5']:.6f}")

    print("\n" + "=" * 80)
    print("OVERALL STRICT-VALID 5-FOLD OOF RESULTS (HYPOTHESIS H66)")
    print("=" * 80)
    m_nested_overall = compute_metrics(nested_oof_preds, labels, eval_qids)
    print(f"Strict Nested CV Recall@5:        {m_nested_overall['recall_at_5']:.6f} (raw: {m_nested_overall['recall_at_5']})")
    print(f"Strict Nested CV Precision@5:     {m_nested_overall['precision_at_5']:.6f}")
    print(f"Strict Nested CV MRR@5:           {m_nested_overall['mrr_at_5']:.6f}")
    print(f"Strict Nested CV Multi-Gold R@5:  {m_nested_overall['multi_gold_recall_at_5']:.6f}")

    # Paired Bootstrap vs Honest Baseline (0.955042)
    best_preds_path = ROOT / "results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json"
    honest_base_preds = json.loads(best_preds_path.read_text(encoding="utf-8"))
    boot_base = paired_bootstrap(honest_base_preds, nested_oof_preds, labels, eval_qids, n_boot=10000, seed=42)
    print(f"\nPaired Bootstrap vs Honest Baseline (0.955042):")
    print(f"  Delta: {boot_base['mean_delta']:+.6f} ({boot_base['mean_delta']*100:+.4f}pp)")
    print(f"  p-value: {boot_base['p_value']:.4f}")
    print(f"  Wins: {boot_base['wins']}, Losses: {boot_base['losses']}, Ties: {boot_base['ties']}")

    # Paired Bootstrap vs Profile LTR Anchor (0.946448)
    boot_prof = paired_bootstrap({q: prof[q] for q in eval_qids}, nested_oof_preds, labels, eval_qids, n_boot=10000, seed=42)
    print(f"\nPaired Bootstrap vs Profile LTR Anchor (0.946448):")
    print(f"  Delta: {boot_prof['mean_delta']:+.6f} ({boot_prof['mean_delta']*100:+.4f}pp)")
    print(f"  p-value: {boot_prof['p_value']:.4f}")
    print(f"  Wins: {boot_prof['wins']}, Losses: {boot_prof['losses']}, Ties: {boot_prof['ties']}")
    print("=" * 80)

    # Save summary report
    summary = {
        "hypothesis": "H66",
        "name": "Deeper Feature-Subsampled XGB-145D-D5 Ensemble",
        "overall_metrics": m_nested_overall,
        "fold_reports": nested_fold_reports,
        "bootstrap_vs_honest_baseline": boot_base,
        "bootstrap_vs_profile_anchor": boot_prof,
        "standalone_xgb_d5_metrics": m_d5_standalone,
    }
    (OUT_DIR / "H66_EXPERIMENT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "H66_OOF_PREDICTIONS.json").write_text(json.dumps(nested_oof_preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary and predictions to {OUT_DIR}")

if __name__ == "__main__":
    main()
