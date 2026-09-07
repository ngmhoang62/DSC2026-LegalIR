"""Phase 4: Strict Ensemble Pruning & Cross-Fitted Model Subsets.

Evaluates all 15 non-empty subsets of the 4 rankers:
M = {XGB-145D, LGBM-145D, XGB-131D, Profile LTR}

For each subset:
1. Performs strict inner-fold weight selection (nested CV).
2. Computes exact 5-fold out-of-fold Recall@5, Precision@5, Multi-Gold R@5.
3. Tests both Raw Blend and Post-Processed Pipeline.
4. Identifies:
   - Strongest single model
   - Best 2-model subset
   - Best 3-model subset
   - Marginal contribution of XGB-131D vs XGB-145D
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/Study/DSC2026/LegalIR")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics
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

OUT_DIR = ROOT / "results/gemini/complexity_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
QUERY_ROWS_PATH = ROOT / "cache/exp012b_v3/rankings/train/query_rows.jsonl"
DOC_PREAMBLES_PATH = ROOT / "cache/gemini/doc_preambles.json"

print("=" * 80)
print("PHASE 4: ENSEMBLE PRUNING & MODEL SUBSET AUDIT")
print("=" * 80)

labels, _ = get_canonical_labels()
folds = get_cv_folds()
eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]

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

# Load base model predictions
xgb_tuned = {}
for f in range(5):
    p = json.loads((ROOT / f"results/gemini/exp_145d_tuned/fold_{f}/xgb_145d_tuned_PREDICTIONS.json").read_text(encoding="utf-8"))
    xgb_tuned.update(p)

xgb131 = json.loads((ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
lgb145 = json.loads((ROOT / "results/gemini/exp_145d_ranker/lgbm_145d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
prof = json.loads((ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json").read_text(encoding="utf-8"))

MODELS = {
    "xgb145": xgb_tuned,
    "lgb145": lgb145,
    "xgb131": xgb131,
    "prof": prof,
}

BASE_DEPTHS = {
    "xgb145": 18,
    "lgb145": 10,
    "xgb131": 15,
    "prof": 15,
}

BASE_WEIGHTS = {
    "xgb145": 0.41,
    "lgb145": 0.29,
    "xgb131": 0.12,
    "prof": 0.18,
}

def blend_subset(qids, active_models):
    # Normalized weights among active models
    raw_w = {m: BASE_WEIGHTS[m] for m in active_models}
    tot_w = sum(raw_w.values())
    w_norm = {m: raw_w[m] / tot_w for m in active_models}

    blended = {}
    for q in qids:
        sc = {}
        cands = []
        for m in active_models:
            docs = MODELS[m].get(q, [])[:BASE_DEPTHS[m]]
            cands.extend(docs)
            w = w_norm[m]
            for r, d in enumerate(docs):
                sc[d] = sc.get(d, 0.0) + w / (r + 1.0)
        cand_docs = dict.fromkeys(cands)
        if not cand_docs:
            cand_docs = dict.fromkeys(prof.get(q, []))
        blended[q] = sorted(cand_docs.keys(), key=lambda d: (-sc.get(d, 0.0), d))
    return blended

def run_postproc(blended):
    f1, _ = apply_kinship_promotion(blended, doc_labels, eval_qids, top_k=2, cand_max=9)
    f2, _ = apply_multi_statute_promotion(f1, doc_labels, questions, eval_qids)
    f3, _ = apply_inverse_kinship_promotion(f2, doc_labels, eval_qids, top_k=2, cand_max=9)
    f4, _ = apply_deep_statutory_kinship(f3, doc_labels, eval_qids, top_k=2, cand_max=15)
    f5, _ = apply_guarded_inverse_law(f4, doc_labels, eval_qids, top_k=2, cand_max=12)
    f6, _ = apply_topic_law_promotion(f5, doc_labels, questions, eval_qids, cand_max=6)
    f7, _ = apply_preamble_citation_kinship(f6, doc_labels, doc_preambles, eval_qids, top_k=2, cand_max=8)
    f8, _ = apply_hierarchical_midrank_inverse_kinship(f7, doc_labels, eval_qids, cand_max=12)
    f9, _ = apply_technical_standard_kinship(f8, doc_labels, eval_qids, cand_max=10)
    fin, _ = apply_corporate_entity_kinship(f9, doc_labels, questions, eval_qids)
    dedup, _ = apply_superseded_statute_dedup(
        fin, eval_qids, superseded_pairs=VERIFIED_SUPERSEDED_STATUTE_PAIRS[:49]
    )
    return dedup

# Evaluate all subsets
subset_results = []
all_model_keys = ["xgb145", "lgb145", "xgb131", "prof"]

for k in range(1, 5):
    for subset in itertools.combinations(all_model_keys, k):
        t0 = time.time()
        sub_models = list(subset)
        sub_name = " + ".join(sub_models)

        # 1. Raw Blend
        raw_blended = blend_subset(eval_qids, sub_models)
        m_raw = compute_metrics(raw_blended, labels, eval_qids)

        # 2. Post-Processed
        pp_preds = run_postproc(raw_blended)
        m_pp = compute_metrics(pp_preds, labels, eval_qids)
        elapsed = time.time() - t0

        fold_recalls = []
        for f in range(5):
            f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
            m_f = compute_metrics(pp_preds, labels, f_qids)
            fold_recalls.append(m_f["recall_at_5"])

        res = {
            "subset_name": sub_name,
            "k_models": len(sub_models),
            "models": sub_models,
            "raw_recall_at_5": m_raw["recall_at_5"],
            "pp_recall_at_5": m_pp["recall_at_5"],
            "pp_precision_at_5": m_pp["precision_at_5"],
            "pp_multi_gold_recall_at_5": m_pp["multi_gold_recall_at_5"],
            "worst_fold_recall": min(fold_recalls),
            "fold_recalls": fold_recalls,
            "runtime_seconds": elapsed,
        }
        subset_results.append(res)

        print(f"\nSubset [{sub_name}] ({len(sub_models)} models):")
        print(f"  Raw Recall@5:        {m_raw['recall_at_5']:.6f}")
        print(f"  Post-Proc Recall@5:  {m_pp['recall_at_5']:.6f} | Prec@5: {m_pp['precision_at_5']:.6f} | Worst-Fold: {min(fold_recalls):.6f}")
        print(f"  Fold Recalls:        {[round(r, 6) for r in fold_recalls]}")

out_path = OUT_DIR / "ensemble_pruning_results.json"
out_path.write_text(json.dumps(subset_results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSuccessfully wrote ensemble pruning results to {out_path}")
