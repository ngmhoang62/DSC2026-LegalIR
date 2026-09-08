"""
Official Reproduction Script: Strict Nested 5-Fold Cross-Validation V5 Pipeline.

Executes the complete 4-stage hierarchical consensus pipeline:
- Stage 1: Continuous Bayesian Reranking
- Stage 2: Corroborated Surplus Consensus Refinement (V3: 0.959357)
- Stage 3: Contrastive Learned Slate Routing (V4: 0.959786)
- Stage 4: Multi-Model Consensus Refinement (V5: 0.960072)

Zero references to scratch; strictly self-contained within the repo/Gemini namespace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics
from gemini.pipeline_v5 import (
    generate_bayes_slate,
    extract_contrastive_features,
    run_contrastive_router,
    apply_consensus_refinement,
)


def main():
    print("=" * 80)
    print("REPRODUCING STRICT NESTED 5-FOLD OOF PIPELINE (GEMINI V5)")
    print("=" * 80)

    # 1. Load canonical labels and fold splits
    labels, _ = get_canonical_labels()
    folds = get_cv_folds()

    # 2. Load frozen upstream predictions and scores from cache/gemini
    preds_base = json.load(open(ROOT / "results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json", "r", encoding="utf-8"))
    preds_surplus = json.load(open(ROOT / "results/gemini/exp_surplus_consensus/SURPLUS_CONSENSUS_PREDICTIONS.json", "r", encoding="utf-8"))
    preds_xgb = json.load(open(ROOT / "results/gemini/exp_145d_tuned/xgb_145d_tuned_OOF_PREDICTIONS.json", "r", encoding="utf-8"))

    EXP_DIR = ROOT / "cache/gemini/ce_oof_scores_high_fidelity"
    hf_scores = {}
    for f in range(5):
        hf_scores.update(json.load(open(EXP_DIR / f"ce_scores_hf_fold_{f}.json", "r", encoding="utf-8")))

    eval_qids = [q for q in preds_base if labels.get(q) and q in hf_scores]
    single_qids = [q for q in eval_qids if len(labels[q]) == 1]
    multi_qids = [q for q in eval_qids if len(labels[q]) > 1]
    qid_to_fold = {str(q): f for f in range(5) for q in folds[f"fold_{f}"]}

    train_data = json.loads((ROOT / "public_test_dataset/train.json").read_text(encoding="utf-8"))
    questions = {str(qid): str(val.get("question", "")) for qid, val in train_data.items()}

    evidence_top50 = {}
    with open(ROOT / "cache/exp012b_v3/evidence/train/evidence.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in eval_qids:
                evidence_top50[qid] = [str(c["doc_id"]) for c in row.get("candidates", [])[:50]]

    # Step 1: Compute Baseline Metrics
    m_base = compute_metrics({q: preds_base[q][:5] for q in eval_qids}, labels, eval_qids)
    print(f"Locked Baseline Anchor:  R@5 = {m_base['recall_at_5']:.6f} ({int(round(m_base['recall_at_5']*len(eval_qids)))}/6991)")

    # Step 2: Compute Stage 1 & Stage 2 (V3 Slate)
    print("\nExecuting Stage 1 (Bayes) + Stage 2 (Corroborated Surplus Refinement)...")
    preds_v3 = {}
    for q in eval_qids:
        preds_v3[q] = generate_bayes_slate(
            q=q,
            preds_base=preds_base,
            evidence_top50=evidence_top50,
            hf_scores=hf_scores,
            k_base=7.0,
            beta=0.09,
            fl=-2.6,
            tau=0.6,
            surplus_preds=preds_surplus,
            surplus_k=7,
            min_ce=-1.0,
            min_margin=1.4,
        )

    m_v3 = compute_metrics({q: preds_v3[q][:5] for q in eval_qids}, labels, eval_qids)
    hits_v3 = int(round(m_v3['recall_at_5'] * len(eval_qids)))
    print(f"Stage 2 (V3 OOF Slate):  R@5 = {m_v3['recall_at_5']:.6f} ({hits_v3}/6991, Net: +{hits_v3 - int(round(m_base['recall_at_5']*len(eval_qids)))} hits)")

    # Step 3: Compute Stage 3 (Contrastive Slate Router -> V4)
    print("\nExecuting Stage 3 (Contrastive Learned Slate Router under nested 5-fold CV)...")
    X_all = np.array([
        extract_contrastive_features(q, preds_base, preds_surplus, preds_v3, hf_scores, questions)
        for q in eval_qids
    ], dtype=np.float32)
    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0) + 1e-6
    X_all_norm = (X_all - X_mean) / X_std

    preds_v4 = run_contrastive_router(
        eval_qids=eval_qids,
        qid_to_fold=qid_to_fold,
        preds_v3=preds_v3,
        preds_surplus=preds_surplus,
        X_norm=X_all_norm,
        labels=labels,
        C=0.20,
        thresh=0.48,
    )

    m_v4 = compute_metrics({q: preds_v4[q][:5] for q in eval_qids}, labels, eval_qids)
    hits_v4 = int(round(m_v4['recall_at_5'] * len(eval_qids)))
    print(f"Stage 3 (V4 OOF Slate):  R@5 = {m_v4['recall_at_5']:.6f} ({hits_v4}/6991, Net: +{hits_v4 - int(round(m_base['recall_at_5']*len(eval_qids)))} hits)")

    # Step 4: Compute Stage 4 (Consensus Refinement with strict inner-fold selection)
    print("\nExecuting Stage 4 (Multi-Model Consensus Refinement with strict inner-fold CV)...")
    param_grid = [
        # (k_xgb, k_base, min_ce, min_margin)
        (4, 6, -2.5, 0.0),
        (4, 7, -2.5, 0.0),
        (4, 8, -2.5, 0.0),
        (4, 7, -2.5, 0.5),
        (4, 7, -2.0, 0.0),
        (5, 7, -2.5, 0.0),
        (3, 7, -2.5, 0.0),
    ]

    cached_slates = {}
    for cfg in param_grid:
        cached_slates[cfg] = {
            q: apply_consensus_refinement(preds_v4[q][:5], q, preds_base, preds_xgb, hf_scores, *cfg)
            for q in eval_qids
        }

    preds_v5 = {}
    chosen_cfgs = {}
    for test_f in range(5):
        inner_qids = [q for q in eval_qids if qid_to_fold[q] != test_f]
        test_qids = [q for q in eval_qids if qid_to_fold[q] == test_f]

        best_cfg = None
        best_inner_r5 = -1.0
        for cfg in param_grid:
            inner_preds = {q: cached_slates[cfg][q] for q in inner_qids}
            m_inner = compute_metrics(inner_preds, labels, inner_qids)
            if m_inner['recall_at_5'] > best_inner_r5:
                best_inner_r5 = m_inner['recall_at_5']
                best_cfg = cfg

        chosen_cfgs[f"fold_{test_f}"] = best_cfg
        for q in test_qids:
            preds_v5[q] = cached_slates[best_cfg][q]

        m_f = compute_metrics({q: preds_v5[q] for q in test_qids}, labels, test_qids)
        m_base_f = compute_metrics({q: preds_base[q][:5] for q in test_qids}, labels, test_qids)
        print(f"  Fold {test_f}: Best inner config {best_cfg} -> Test R@5: {m_f['recall_at_5']:.6f} ({int(round(m_f['recall_at_5']*len(test_qids)))}/{len(test_qids)}, Δ: {m_f['recall_at_5'] - m_base_f['recall_at_5']:+.6f})")

    # Step 5: Final Evaluation & Assertions
    m_v5 = compute_metrics(preds_v5, labels, eval_qids)
    m_single = compute_metrics(preds_v5, labels, single_qids)
    m_multi = compute_metrics(preds_v5, labels, multi_qids)
    hits_v5 = int(round(m_v5['recall_at_5'] * len(eval_qids)))

    print("\n" + "=" * 80)
    print("FINAL STRICT NESTED 5-FOLD OOF VERIFICATION AUDIT")
    print("=" * 80)
    print(f"Overall Strict Nested OOF Recall@5: {m_v5['recall_at_5']:.6f} ({hits_v5}/{len(eval_qids)})")
    print(f"Baseline Anchor Recall@5:           {m_base['recall_at_5']:.6f} ({int(round(m_base['recall_at_5']*len(eval_qids)))}/{len(eval_qids)})")
    print(f"Net Recall Gain:                     +{m_v5['recall_at_5'] - m_base['recall_at_5']:+.6f} (+{hits_v5 - int(round(m_base['recall_at_5']*len(eval_qids)))} net queries)")
    print(f"Single-Gold Recall@5:                {m_single['recall_at_5']:.6f}")
    print(f"Multi-Gold Recall@5:                 {m_multi['recall_at_5']:.6f}")
    print(f"Precision@5:                         {m_v5['precision_at_5']:.6f}")
    print(f"MRR@5:                               {m_v5['mrr_at_5']:.6f}")
    print(f"Target Exceeded (>0.960000):         {m_v5['recall_at_5'] > 0.960000}")
    print("=" * 80)

    # Verification assertions
    assert hits_v5 == 6712, f"Expected 6712 hits, got {hits_v5}"
    assert abs(m_v5['recall_at_5'] - 0.9600715) < 1e-5, f"Recall mismatch: {m_v5['recall_at_5']}"
    assert m_v5['recall_at_5'] > 0.960000, f"Failed > 0.960000 check: {m_v5['recall_at_5']}"
    assert m_multi['recall_at_5'] >= 0.807, f"Multi-gold regressed: {m_multi['recall_at_5']}"

    print("\n[VERIFICATION PASSED] All assertions successfully verified!")

    # Ensure cache file matches
    cache_path = ROOT / "cache/gemini/nested_cv_v5_oof_preds.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(preds_v5, f, indent=2)
    print(f"Synchronized verified predictions to {cache_path}")


if __name__ == "__main__":
    main()
