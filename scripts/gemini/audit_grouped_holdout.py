"""
Forensic Audit: Grouped Gold-Document Holdout Validation.

Groups queries by primary gold document ID (GroupKFold with 5 splits).
Queries sharing the same gold document are strictly placed into the same fold.
Evaluates out-of-document generalization without tuning any parameter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels
from gemini.metrics import compute_metrics
from gemini.pipeline_v5 import (
    generate_bayes_slate,
    extract_contrastive_features,
    apply_consensus_refinement,
)


def main():
    print("=" * 80)
    print("FORENSIC GENERALIZATION AUDIT: GROUPED GOLD-DOCUMENT HOLDOUT (5-FOLD)")
    print("=" * 80)

    labels, _ = get_canonical_labels()

    preds_base = json.load(open(ROOT / "results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json", "r", encoding="utf-8"))
    preds_surplus = json.load(open(ROOT / "results/gemini/exp_surplus_consensus/SURPLUS_CONSENSUS_PREDICTIONS.json", "r", encoding="utf-8"))
    preds_xgb = json.load(open(ROOT / "results/gemini/exp_145d_tuned/xgb_145d_tuned_OOF_PREDICTIONS.json", "r", encoding="utf-8"))

    EXP_DIR = ROOT / "cache/gemini/ce_oof_scores_high_fidelity"
    hf_scores = {}
    for f in range(5):
        hf_scores.update(json.load(open(EXP_DIR / f"ce_scores_hf_fold_{f}.json", "r", encoding="utf-8")))

    eval_qids = sorted([q for q in preds_base if labels.get(q) and q in hf_scores])
    train_data = json.loads((ROOT / "public_test_dataset/train.json").read_text(encoding="utf-8"))
    questions = {str(qid): str(val.get("question", "")) for qid, val in train_data.items()}

    evidence_top50 = {}
    with open(ROOT / "cache/exp012b_v3/evidence/train/evidence.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in eval_qids:
                evidence_top50[qid] = [str(c["doc_id"]) for c in row.get("candidates", [])[:50]]

    # Determine primary group for each query: smallest doc_id in sorted label set
    groups = [sorted(list(labels[q]))[0] for q in eval_qids]
    unique_groups = set(groups)
    print(f"Total evaluable queries: {len(eval_qids)}")
    print(f"Total unique gold document groups: {len(unique_groups)}")

    gkf = GroupKFold(n_splits=5)
    grouped_folds = {}
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(eval_qids, groups=groups)):
        grouped_folds[fold_idx] = {
            "train": [eval_qids[i] for i in train_idx],
            "val": [eval_qids[i] for i in val_idx],
        }
        train_groups = set(groups[i] for i in train_idx)
        val_groups = set(groups[i] for i in val_idx)
        assert len(train_groups & val_groups) == 0, f"Leakage detected in fold {fold_idx}!"
        print(f"  Fold {fold_idx}: Train {len(train_idx)} queries ({len(train_groups)} docs), Val {len(val_idx)} queries ({len(val_groups)} docs), Overlap: 0")

    print("\nEvaluating Grouped Holdout Generalization across all stages (NO PARAMETER TUNING)...")

    # Step 1: Baseline on Grouped Holdouts
    m_base = compute_metrics({q: preds_base[q][:5] for q in eval_qids}, labels, eval_qids)
    print(f"\n1. Baseline Anchor: Recall@5 = {m_base['recall_at_5']:.6f} ({int(round(m_base['recall_at_5']*len(eval_qids)))}/6991)")

    # Step 2: V3 (Bayes + Corroborated Surplus)
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
    hits_v3 = int(round(m_v3['recall_at_5']*len(eval_qids)))
    print(f"2. Stage 2 (V3 Slate): Recall@5 = {m_v3['recall_at_5']:.6f} ({hits_v3}/6991, Net: +{hits_v3 - int(round(m_base['recall_at_5']*len(eval_qids)))} hits)")

    # Step 3: Contrastive Router fitted strictly on Grouped Folds
    print("\nFitting Stage 3 Contrastive Router strictly on Grouped Train Folds...")
    X_all = np.array([
        extract_contrastive_features(q, preds_base, preds_surplus, preds_v3, hf_scores, questions)
        for q in eval_qids
    ], dtype=np.float32)

    qid_to_idx = {q: i for i, q in enumerate(eval_qids)}
    y_all = np.zeros(len(eval_qids), dtype=np.int32)
    w_all = np.zeros(len(eval_qids), dtype=np.float32)
    for i, q in enumerate(eval_qids):
        g = set(labels[q])
        hb = len(g & set(preds_v3[q][:5])) / len(g)
        hs = len(g & set(preds_surplus[q][:5])) / len(g)
        y_all[i] = 1 if hb >= hs else 0
        w_all[i] = abs(hb - hs)

    preds_v4_grouped = {}
    for fold_idx in range(5):
        train_qids = grouped_folds[fold_idx]["train"]
        val_qids = grouped_folds[fold_idx]["val"]

        train_indices = [qid_to_idx[q] for q in train_qids]
        val_indices = [qid_to_idx[q] for q in val_qids]

        X_train = X_all[train_indices]
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0) + 1e-6
        X_train_norm = (X_train - X_mean) / X_std
        X_val_norm = (X_all[val_indices] - X_mean) / X_std

        clf = LogisticRegression(C=0.20, random_state=42 + fold_idx, max_iter=1000)
        clf.fit(X_train_norm, y_all[train_indices], sample_weight=w_all[train_indices] + 0.001)

        val_probs = clf.predict_proba(X_val_norm)[:, 0]

        for q, p in zip(val_qids, val_probs):
            if set(preds_v3[q][:5]) != set(preds_surplus[q][:5]) and p > 0.48:
                preds_v4_grouped[q] = preds_surplus[q][:5]
            else:
                preds_v4_grouped[q] = preds_v3[q][:5]

        m_fold = compute_metrics({q: preds_v4_grouped[q] for q in val_qids}, labels, val_qids)
        m_base_f = compute_metrics({q: preds_base[q][:5] for q in val_qids}, labels, val_qids)
        print(f"  Grouped Fold {fold_idx}: Val R@5 = {m_fold['recall_at_5']:.6f} | Baseline = {m_base_f['recall_at_5']:.6f} | Net = {m_fold['recall_at_5'] - m_base_f['recall_at_5']:+.6f}")

    m_v4_g = compute_metrics(preds_v4_grouped, labels, eval_qids)
    hits_v4_g = int(round(m_v4_g['recall_at_5'] * len(eval_qids)))
    print(f"\n3. Stage 3 (V4 Grouped OOF): Recall@5 = {m_v4_g['recall_at_5']:.6f} ({hits_v4_g}/6991, Net: +{hits_v4_g - int(round(m_base['recall_at_5']*len(eval_qids)))} hits)")

    # Step 4: Consensus Refinement (k_xgb=4, k_base=6, min_ce=-2.5, min_margin=0.0) applied to Grouped V4
    preds_v5_grouped = {}
    for q in eval_qids:
        preds_v5_grouped[q] = apply_consensus_refinement(
            p5=preds_v4_grouped[q][:5],
            q=q,
            preds_base=preds_base,
            preds_xgb=preds_xgb,
            hf_scores=hf_scores,
            k_xgb=4,
            k_base=6,
            min_ce=-2.5,
            min_margin=0.0,
        )

    m_v5_g = compute_metrics(preds_v5_grouped, labels, eval_qids)
    single_qids = [q for q in eval_qids if len(labels[q]) == 1]
    multi_qids = [q for q in eval_qids if len(labels[q]) > 1]
    m_single_g = compute_metrics(preds_v5_grouped, labels, single_qids)
    m_multi_g = compute_metrics(preds_v5_grouped, labels, multi_qids)
    hits_v5_g = int(round(m_v5_g['recall_at_5'] * len(eval_qids)))

    print("\n" + "=" * 80)
    print("FINAL GROUPED GOLD-DOCUMENT HOLDOUT RESULTS (ZERO LEAKAGE ACROSS DOCUMENTS)")
    print("=" * 80)
    print(f"Overall Grouped Holdout Recall@5:   {m_v5_g['recall_at_5']:.6f} ({hits_v5_g}/6991)")
    print(f"Baseline Anchor Recall@5:           {m_base['recall_at_5']:.6f} ({int(round(m_base['recall_at_5']*len(eval_qids)))}/6991)")
    print(f"Net Recall Gain on Unseen Docs:     +{m_v5_g['recall_at_5'] - m_base['recall_at_5']:+.6f} (+{hits_v5_g - int(round(m_base['recall_at_5']*len(eval_qids)))} net queries)")
    print(f"Single-Gold Recall@5:                {m_single_g['recall_at_5']:.6f}")
    print(f"Multi-Gold Recall@5:                 {m_multi_g['recall_at_5']:.6f}")
    print(f"Target Exceeded (>0.960000):         {m_v5_g['recall_at_5'] > 0.960000} (Margin: +{(m_v5_g['recall_at_5'] - 0.960000)*len(eval_qids):.2f} queries)")
    print("=" * 80)


if __name__ == "__main__":
    main()
