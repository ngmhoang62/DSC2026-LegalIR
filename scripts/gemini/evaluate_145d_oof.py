"""Comprehensive 5-Fold OOF Evaluation of 145D Rankers (Gemini H41)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.kinship import (
    apply_kinship_promotion,
    apply_inverse_kinship_promotion,
    apply_multi_statute_promotion,
    load_doc_labels,
)

OUT_145 = ROOT / "results/gemini/exp_145d_ranker"
OUT_131 = ROOT / "results/gemini/exp_authority_131d"
NESTED_SLATE_PATH = ROOT / "results/exp_final_retrieval/nested_slate_probe/OOF_PREDICTIONS.json"
PROFILE_DIR = ROOT / "results/exp_final_retrieval/profile_ltr_probe"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
QUERY_ROWS_PATH = ROOT / "cache/exp012b_v3/rankings/train/query_rows.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_queries() -> dict[str, str]:
    queries = {}
    if QUERY_ROWS_PATH.exists():
        with open(QUERY_ROWS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                queries[str(item.get("query_id") or item.get("id") or item.get("qid"))] = (
                    item.get("text") or item.get("query") or ""
                )
    return queries


def main():
    print("=" * 80)
    print("COMPREHENSIVE 5-FOLD OOF EVALUATION: 145D ENHANCED RANKERS")
    print("=" * 80)

    labels, audit = get_canonical_labels()
    folds = get_cv_folds()
    eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]
    print(f"Loaded {len(eval_qids)} evaluable queries across 5 folds.")

    # 1. Assemble 5-Fold OOF Predictions for 145D
    xgb_145_oof = {}
    lgb_145_oof = {}
    xgb_131_oof = {}
    lgb_131_oof = {}
    prof_oof = {}

    for f in range(5):
        xgb_145_oof.update(read_json(OUT_145 / f"fold_{f}/xgb_145d_PREDICTIONS.json"))
        lgb_145_oof.update(read_json(OUT_145 / f"fold_{f}/lgbm_145d_PREDICTIONS.json"))
        xgb_131_oof.update(read_json(OUT_131 / f"fold_{f}/xgb_131d_PREDICTIONS.json"))
        lgb_131_oof.update(read_json(OUT_131 / f"fold_{f}/lgbm_131d_PREDICTIONS.json"))
        prof_oof.update(read_json(PROFILE_DIR / "l15_t5/PREDICTIONS.json"))

    write_json(OUT_145 / "xgb_145d_OOF_PREDICTIONS.json", xgb_145_oof)
    write_json(OUT_145 / "lgbm_145d_OOF_PREDICTIONS.json", lgb_145_oof)

    # 2. Compute Standalone OOF Metrics
    m_xgb145 = compute_metrics(xgb_145_oof, labels, eval_qids)
    m_lgb145 = compute_metrics(lgb_145_oof, labels, eval_qids)
    m_xgb131 = compute_metrics(xgb_131_oof, labels, eval_qids)
    m_lgb131 = compute_metrics(lgb_131_oof, labels, eval_qids)

    print("\n--- STANDALONE 5-FOLD OOF PERFORMANCE ---")
    print(f"XGB 131D: Recall@5 = {m_xgb131['recall_at_5']:.6f}, Prec@5 = {m_xgb131['precision_at_5']:.6f}, MRR@5 = {m_xgb131['mrr_at_5']:.6f}")
    print(f"XGB 145D: Recall@5 = {m_xgb145['recall_at_5']:.6f}, Prec@5 = {m_xgb145['precision_at_5']:.6f}, MRR@5 = {m_xgb145['mrr_at_5']:.6f} (Delta: {m_xgb145['recall_at_5']-m_xgb131['recall_at_5']:+.6f})")
    print(f"LGB 131D: Recall@5 = {m_lgb131['recall_at_5']:.6f}, Prec@5 = {m_lgb131['precision_at_5']:.6f}, MRR@5 = {m_lgb131['mrr_at_5']:.6f}")
    print(f"LGB 145D: Recall@5 = {m_lgb145['recall_at_5']:.6f}, Prec@5 = {m_lgb145['precision_at_5']:.6f}, MRR@5 = {m_lgb145['mrr_at_5']:.6f} (Delta: {m_lgb145['recall_at_5']-m_lgb131['recall_at_5']:+.6f})")

    # 3. Test Multi-Model Ensembles with 145D
    print("\n--- MULTI-MODEL ENSEMBLE EXPLORATION ---")
    doc_labels = load_doc_labels(EVIDENCE_DB)
    questions = load_queries()
    nested_slate_preds = read_json(NESTED_SLATE_PATH)
    m_nested = compute_metrics(nested_slate_preds, labels, eval_qids)
    print(f"Upstream Nested Slate Benchmark: Recall@5 = {m_nested['recall_at_5']:.6f}")

    candidates = [
        # Blend recipes: weights for (xgb_131, lgb_145, prof, nested_slate, xgb_145)
        {"desc": "0.45 XGB131 + 0.25 LGB145 + 0.30 PROF", "w": (0.45, 0.25, 0.30, 0.0, 0.0)},
        {"desc": "0.35 XGB131 + 0.35 LGB145 + 0.30 PROF", "w": (0.35, 0.35, 0.30, 0.0, 0.0)},
        {"desc": "0.50 XGB131 + 0.25 LGB145 + 0.25 PROF", "w": (0.50, 0.25, 0.25, 0.0, 0.0)},
        {"desc": "0.30 XGB131 + 0.20 XGB145 + 0.25 LGB145 + 0.25 PROF", "w": (0.30, 0.25, 0.25, 0.0, 0.20)},
        {"desc": "0.60 NestedSlate + 0.25 LGB145 + 0.15 XGB145", "w": (0.0, 0.25, 0.0, 0.60, 0.15)},
        {"desc": "0.70 NestedSlate + 0.30 LGB145", "w": (0.0, 0.30, 0.0, 0.70, 0.0)},
        {"desc": "0.80 NestedSlate + 0.20 LGB145", "w": (0.0, 0.20, 0.0, 0.80, 0.0)},
    ]

    best_blend_preds = None
    best_blend_kin_preds = None
    best_blend_recall = 0.0
    best_desc = ""

    for cand in candidates:
        w_xgb131, w_lgb145, w_prof, w_nest, w_xgb145 = cand["w"]
        k = 10
        blended = {}
        for q in eval_qids:
            sc: dict[str, float] = {}
            if w_xgb131 > 0:
                for r, d in enumerate(xgb_131_oof[q][:64], 1):
                    sc[d] = sc.get(d, 0.0) + w_xgb131 / (k + r)
            if w_lgb145 > 0:
                for r, d in enumerate(lgb_145_oof[q][:64], 1):
                    sc[d] = sc.get(d, 0.0) + w_lgb145 / (k + r)
            if w_prof > 0:
                for r, d in enumerate(prof_oof[q][:64], 1):
                    sc[d] = sc.get(d, 0.0) + w_prof / (k + r)
            if w_nest > 0:
                for r, d in enumerate(nested_slate_preds[q][:64], 1):
                    sc[d] = sc.get(d, 0.0) + w_nest / (k + r)
            if w_xgb145 > 0:
                for r, d in enumerate(xgb_145_oof[q][:64], 1):
                    sc[d] = sc.get(d, 0.0) + w_xgb145 / (k + r)
            blended[q] = sorted(sc.keys(), key=lambda d: (-sc[d], d))

        m_raw = compute_metrics(blended, labels, eval_qids)

        # Apply Statutory Kinship fold-by-fold
        kin_preds = {}
        for f in range(5):
            f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
            f_kin, _ = apply_kinship_promotion(blended, doc_labels, f_qids, top_k=2, cand_max=9)
            f_inv, _ = apply_inverse_kinship_promotion(f_kin, doc_labels, f_qids, top_k=2, cand_max=9)
            f_final, _ = apply_multi_statute_promotion(f_inv, doc_labels, questions, f_qids)
            kin_preds.update(f_final)

        m_kin = compute_metrics(kin_preds, labels, eval_qids)
        print(f"\n{cand['desc']}:")
        print(f"  Raw:         Recall@5 = {m_raw['recall_at_5']:.6f}, Prec@5 = {m_raw['precision_at_5']:.6f}, Multi = {m_raw['multi_gold_recall_at_5']:.6f}")
        print(f"  + Kinship:   Recall@5 = {m_kin['recall_at_5']:.6f}, Prec@5 = {m_kin['precision_at_5']:.6f}, Multi = {m_kin['multi_gold_recall_at_5']:.6f}")

        if m_kin["recall_at_5"] > best_blend_recall:
            best_blend_recall = m_kin["recall_at_5"]
            best_blend_preds = blended
            best_blend_kin_preds = kin_preds
            best_desc = cand["desc"]

    print("\n" + "=" * 80)
    print(f"BEST 145D COMPOSITE CONFIGURATION: {best_desc}")
    print(f"Recall@5 = {best_blend_recall:.6f}")
    print("=" * 80)

    # Save summary report
    summary = {
        "status": "COMPLETE_145D_EVALUATION",
        "best_desc": best_desc,
        "best_recall_at_5": best_blend_recall,
        "metrics_xgb145": m_xgb145,
        "metrics_lgb145": m_lgb145,
    }
    write_json(OUT_145 / "145D_EVALUATION_SUMMARY.json", summary)


if __name__ == "__main__":
    main()
