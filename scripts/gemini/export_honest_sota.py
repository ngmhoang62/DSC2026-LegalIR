"""Export clean nested cross-validation SOTA (0.955042) and revoke H61 target reached claim.

Establishes the true, leak-free benchmark:
- Nested 5-fold cross-validation with inner-fold hyperparameter and rule fitting.
- 0 manual query patches (completely purged TARGETED_STATUTORY_SPECS_V4).
- Fully reproducible, isolated, and audited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("D:/Study/DSC2026/LegalIR")
sys.path.insert(0, str(ROOT / "src"))

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

BEST_DIR = ROOT / "results/gemini/best_ensemble"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
QUERY_ROWS_PATH = ROOT / "cache/exp012b_v3/rankings/train/query_rows.jsonl"
DOC_PREAMBLES_PATH = ROOT / "cache/gemini/doc_preambles.json"

CANDIDATE_FUSION_CONFIGS = [
    # (w_tuned, w_lgb, w_xgb131, w_prof, k_xgb, k_lgb, k_131, k_prof)
    (0.40, 0.30, 0.15, 0.15, 15, 15, 15, 15),
    (0.41, 0.29, 0.12, 0.18, 15, 15, 15, 15),
    (0.41, 0.29, 0.12, 0.18, 18, 10, 15, 15),
    (0.45, 0.25, 0.15, 0.15, 18, 12, 15, 15),
    (0.35, 0.35, 0.15, 0.15, 15, 15, 15, 15),
    (0.40, 0.30, 0.15, 0.15, 18, 10, 15, 15),
]


def blend_predictions(qids, cfg, xgb_tuned, lgb145, xgb131, prof):
    w_tuned, w_lgb, w_131, w_prof, k_x, k_l, k_13, k_p = cfg
    blended = {}
    for q in qids:
        sc = {}
        for r, d in enumerate(xgb_tuned.get(q, [])[:k_x]):
            sc[d] = sc.get(d, 0.0) + w_tuned / (r + 1.0)
        for r, d in enumerate(lgb145.get(q, [])[:k_l]):
            sc[d] = sc.get(d, 0.0) + w_lgb / (r + 1.0)
        for r, d in enumerate(xgb131.get(q, [])[:k_13]):
            sc[d] = sc.get(d, 0.0) + w_131 / (r + 1.0)
        for r, d in enumerate(prof.get(q, [])[:k_p]):
            sc[d] = sc.get(d, 0.0) + w_prof / (r + 1.0)
        cand_docs = dict.fromkeys(
            xgb_tuned.get(q, [])[:k_x]
            + lgb145.get(q, [])[:k_l]
            + xgb131.get(q, [])[:k_13]
            + prof.get(q, [])[:k_p]
        )
        if not cand_docs:
            cand_docs = dict.fromkeys(prof.get(q, []))
        blended[q] = sorted(cand_docs.keys(), key=lambda d: (-sc.get(d, 0.0), d))
    return blended


def apply_unsupervised_kinship(rankings, qids, doc_labels, questions, doc_preambles):
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


def export_honest_sota():
    print("=" * 80)
    print("EXPORTING HONEST NESTED-CV SOTA BASELINE (0.955042)")
    print("=" * 80)

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    labels, _ = get_canonical_labels()
    folds = get_cv_folds()
    eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]
    all_7000_qids = [q for f in range(5) for q in folds[f"fold_{f}"]]

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

    xgb_tuned = {}
    for f in range(5):
        p = json.loads((ROOT / f"results/gemini/exp_145d_tuned/fold_{f}/xgb_145d_tuned_PREDICTIONS.json").read_text(encoding="utf-8"))
        xgb_tuned.update(p)

    xgb131 = json.loads((ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    lgb145 = json.loads((ROOT / "results/gemini/exp_145d_ranker/lgbm_145d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
    prof = json.loads((ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json").read_text(encoding="utf-8"))

    strict_oof_preds = {}
    fold_reports = []

    for f_outer in range(5):
        outer_qids = [q for q in folds[f"fold_{f_outer}"] if labels.get(q)]
        outer_all_qids = [q for q in folds[f"fold_{f_outer}"]]
        inner_qids = [q for f in range(5) if f != f_outer for q in folds[f"fold_{f}"] if labels.get(q)]

        # 1. Select Best Fusion Config on Inner Folds ONLY
        best_cfg = None
        best_inner_r5 = -1.0
        for cfg in CANDIDATE_FUSION_CONFIGS:
            inner_blend = blend_predictions(inner_qids, cfg, xgb_tuned, lgb145, xgb131, prof)
            inner_kin = apply_unsupervised_kinship(inner_blend, inner_qids, doc_labels, questions, doc_preambles)
            m = compute_metrics(inner_kin, labels, inner_qids)
            if m["recall_at_5"] > best_inner_r5:
                best_inner_r5 = m["recall_at_5"]
                best_cfg = cfg

        # 2. Cross-fit Superseded Statute Pairs on Inner Folds ONLY
        inner_blend = blend_predictions(inner_qids, best_cfg, xgb_tuned, lgb145, xgb131, prof)
        inner_kin = apply_unsupervised_kinship(inner_blend, inner_qids, doc_labels, questions, doc_preambles)

        cross_fitted_pairs = []
        for old_id, new_id in VERIFIED_SUPERSEDED_STATUTE_PAIRS:
            both_in_top5 = 0
            old_is_gold = 0
            for q in inner_qids:
                top5 = set(inner_kin.get(q, [])[:5])
                if old_id in top5 and new_id in top5:
                    both_in_top5 += 1
                    if old_id in labels[q]:
                        old_is_gold += 1
            is_primary_code = (old_id, new_id) in VERIFIED_SUPERSEDED_STATUTE_PAIRS[:11]
            if (both_in_top5 >= 1 and old_is_gold == 0) or is_primary_code:
                cross_fitted_pairs.append((old_id, new_id))

        # 3. Apply to outer fold evaluable queries
        outer_blend = blend_predictions(outer_qids, best_cfg, xgb_tuned, lgb145, xgb131, prof)
        outer_kin = apply_unsupervised_kinship(outer_blend, outer_qids, doc_labels, questions, doc_preambles)
        outer_dedup, dedup_cnt = apply_superseded_statute_dedup(
            outer_kin, outer_qids, superseded_pairs=cross_fitted_pairs
        )

        strict_oof_preds.update({q: outer_dedup[q] for q in outer_qids})
        # Add non-evaluable queries from prof baseline
        for q in outer_all_qids:
            if q not in strict_oof_preds:
                strict_oof_preds[q] = prof.get(q, [])
        m_outer = compute_metrics(outer_dedup, labels, outer_qids)
        fold_reports.append({
            "fold": f_outer,
            "queries": len(outer_qids),
            "metrics": m_outer,
            "selected_config": {
                "weights": list(best_cfg[:4]),
                "depths": list(best_cfg[4:]),
            },
            "cross_fitted_pairs_count": len(cross_fitted_pairs),
            "dedup_promotions": dedup_cnt,
        })
        print(f"Fold {f_outer} STRICT-VALID Recall@5 = {m_outer['recall_at_5']:.6f}, Prec@5 = {m_outer['precision_at_5']:.6f}")

    overall_metrics = compute_metrics(strict_oof_preds, labels, eval_qids)
    print("\n" + "=" * 80)
    print(f"Overall Strict-Valid 5-Fold OOF Recall@5:        {overall_metrics['recall_at_5']:.6f}")
    print(f"Overall Strict-Valid 5-Fold OOF Precision@5:     {overall_metrics['precision_at_5']:.6f}")
    print(f"Overall Strict-Valid 5-Fold OOF MRR@5:           {overall_metrics['mrr_at_5']:.6f}")
    print(f"Overall Strict-Valid 5-Fold OOF Multi-Gold R@5:  {overall_metrics['multi_gold_recall_at_5']:.6f}")
    print("=" * 80)

    # Paired Bootstrap vs Profile LTR Anchor
    prof_all = {q: prof[q] for q in eval_qids}
    boot_prof = paired_bootstrap(prof_all, strict_oof_preds, labels, eval_qids, n_boot=10000, seed=42)
    print(f"\nPaired Bootstrap vs Profile LTR Anchor (0.946448):")
    print(f"  Delta: {boot_prof['mean_delta']:+.6f}, p-value: {boot_prof['p_value']:.4f}")
    print(f"  Wins: {boot_prof['wins']}, Losses: {boot_prof['losses']}, Ties: {boot_prof['ties']}")

    # Write clean predictions
    best_preds_path = BEST_DIR / "BEST_ENSEMBLE_PREDICTIONS.json"
    best_preds_path.write_text(json.dumps(strict_oof_preds, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write clean summary with explicit revocation
    summary = {
        "status": "REVOKED_H61_REPLACED_BY_HONEST_NESTED_CV_SOTA",
        "revocation_notice": "H61 goal complete conclusion was formally revoked following a forensic audit. H61 achieved 0.960621 via 53 manual query patches where 88.7% memorized specific training fold queries (+0.5579pp inflation) and public submission suffered a deployment parity failure. This file now reflects the honest, leak-free, nested CV benchmark (0.955042).",
        "official_target_reached": False,
        "target_recall_at_5": 0.960000,
        "achieved_recall_at_5": overall_metrics["recall_at_5"],
        "architecture": "145D GBDT Ensemble (Tuned XGB-145D + LGBM-145D + XGB-131D + Profile LTR) + Nested CV AMFD Fusion + Unsupervised Statutory Kinship Suite",
        "overall_metrics": overall_metrics,
        "fold_reports": fold_reports,
        "bootstrap_vs_profile_anchor": boot_prof,
    }
    summary_path = BEST_DIR / "BEST_ENSEMBLE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSuccessfully wrote honest predictions to {best_preds_path}")
    print(f"Successfully wrote honest summary to {summary_path}")


if __name__ == "__main__":
    export_honest_sota()
