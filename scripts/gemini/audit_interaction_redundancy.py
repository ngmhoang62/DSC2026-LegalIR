"""Phase 2: Interaction & Redundancy Audit across Rule Subgroups.

Investigates redundancy and overlapping mechanisms:
1. Subgroup A (Amendment Kinship): Forward, Inverse, Deep, Mid-rank Inverse
2. Subgroup B (Niche / Micro Rules): Multi-statute, Corporate Entity, Topic Law
3. Subgroup C (Document Metadata): Preamble Citation, Technical Standard
4. Subgroup D (Statutory Cleansing): Superseded Statute Dedup

Evaluates:
- Full 11 rules (Baseline)
- Zero rules (Raw Ensemble)
- Dedup ONLY (Superseded Statute Dedup, no heuristic swaps)
- Consolidated Core: Deep Kinship + Guarded Inverse Law + Preamble + Technical Standard + Dedup (5 rules)
- Strongest Kinship Only: Deep Kinship + Dedup (2 rules)
- Ablate entire Amendment Kinship family
- Ablate entire Niche / Micro Rule family
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
print("PHASE 2: INTERACTION & REDUNDANCY AUDIT")
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

# Load 4 base model predictions
xgb_tuned = {}
for f in range(5):
    p = json.loads((ROOT / f"results/gemini/exp_145d_tuned/fold_{f}/xgb_145d_tuned_PREDICTIONS.json").read_text(encoding="utf-8"))
    xgb_tuned.update(p)

xgb131 = json.loads((ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
lgb145 = json.loads((ROOT / "results/gemini/exp_145d_ranker/lgbm_145d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
prof = json.loads((ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json").read_text(encoding="utf-8"))

# Blend predictions using the honest config
w_x, w_l, w_13, w_p = 0.41, 0.29, 0.12, 0.18
k_x, k_l, k_13, k_p = 18, 10, 15, 15

base_blended = {}
for q in eval_qids:
    sc = {}
    for r, d in enumerate(xgb_tuned.get(q, [])[:k_x]):
        sc[d] = sc.get(d, 0.0) + w_x / (r + 1.0)
    for r, d in enumerate(lgb145.get(q, [])[:k_l]):
        sc[d] = sc.get(d, 0.0) + w_l / (r + 1.0)
    for r, d in enumerate(xgb131.get(q, [])[:k_13]):
        sc[d] = sc.get(d, 0.0) + w_13 / (r + 1.0)
    for r, d in enumerate(prof.get(q, [])[:k_p]):
        sc[d] = sc.get(d, 0.0) + w_p / (r + 1.0)
    cand_docs = dict.fromkeys(
        xgb_tuned.get(q, [])[:k_x]
        + lgb145.get(q, [])[:k_l]
        + xgb131.get(q, [])[:k_13]
        + prof.get(q, [])[:k_p]
    )
    if not cand_docs:
        cand_docs = dict.fromkeys(prof.get(q, []))
    base_blended[q] = sorted(cand_docs.keys(), key=lambda d: (-sc.get(d, 0.0), d))

CONFIGURATIONS = [
    {
        "name": "Full 11-Rule Pipeline (Authoritative Baseline)",
        "rules": ["forward", "multi_statute", "inverse", "deep", "guarded_inverse_law", "topic_law", "preamble", "midrank", "technical_standard", "corporate_entity", "superseded_dedup"],
    },
    {
        "name": "Zero Post-Processing (Raw 4-Model Ensemble Blend)",
        "rules": [],
    },
    {
        "name": "Dedup ONLY (Superseded Statute Dedup, 1 rule)",
        "rules": ["superseded_dedup"],
    },
    {
        "name": "Consolidated Core (Deep + Guarded Law + Preamble + Tech Standard + Dedup, 5 rules)",
        "rules": ["deep", "guarded_inverse_law", "preamble", "technical_standard", "superseded_dedup"],
    },
    {
        "name": "Strongest Kinship + Dedup (Deep Kinship + Dedup, 2 rules)",
        "rules": ["deep", "superseded_dedup"],
    },
    {
        "name": "Drop Micro-Rules (Drop Multi-Statute, Corporate Entity, Midrank Inverse)",
        "rules": ["forward", "inverse", "deep", "guarded_inverse_law", "topic_law", "preamble", "technical_standard", "superseded_dedup"],
    },
    {
        "name": "Drop All Amendment Kinship (Keep only Law, Preamble, Tech, Dedup)",
        "rules": ["guarded_inverse_law", "topic_law", "preamble", "technical_standard", "superseded_dedup"],
    },
]

def run_selective_pipeline(rankings, active_rules):
    cur = rankings
    if "forward" in active_rules:
        cur, _ = apply_kinship_promotion(cur, doc_labels, eval_qids, top_k=2, cand_max=9)
    if "multi_statute" in active_rules:
        cur, _ = apply_multi_statute_promotion(cur, doc_labels, questions, eval_qids)
    if "inverse" in active_rules:
        cur, _ = apply_inverse_kinship_promotion(cur, doc_labels, eval_qids, top_k=2, cand_max=9)
    if "deep" in active_rules:
        cur, _ = apply_deep_statutory_kinship(cur, doc_labels, eval_qids, top_k=2, cand_max=15)
    if "guarded_inverse_law" in active_rules:
        cur, _ = apply_guarded_inverse_law(cur, doc_labels, eval_qids, top_k=2, cand_max=12)
    if "topic_law" in active_rules:
        cur, _ = apply_topic_law_promotion(cur, doc_labels, questions, eval_qids, cand_max=6)
    if "preamble" in active_rules:
        cur, _ = apply_preamble_citation_kinship(cur, doc_labels, doc_preambles, eval_qids, top_k=2, cand_max=8)
    if "midrank" in active_rules:
        cur, _ = apply_hierarchical_midrank_inverse_kinship(cur, doc_labels, eval_qids, cand_max=12)
    if "technical_standard" in active_rules:
        cur, _ = apply_technical_standard_kinship(cur, doc_labels, eval_qids, cand_max=10)
    if "corporate_entity" in active_rules:
        cur, _ = apply_corporate_entity_kinship(cur, doc_labels, questions, eval_qids)
    if "superseded_dedup" in active_rules:
        cur, _ = apply_superseded_statute_dedup(
            cur, eval_qids, superseded_pairs=VERIFIED_SUPERSEDED_STATUTE_PAIRS[:49]
        )
    return cur

base_m = compute_metrics(run_selective_pipeline(base_blended, CONFIGURATIONS[0]["rules"]), labels, eval_qids)

redundancy_results = []

for cfg in CONFIGURATIONS:
    t0 = time.time()
    preds = run_selective_pipeline(base_blended, cfg["rules"])
    elapsed = time.time() - t0
    m = compute_metrics(preds, labels, eval_qids)
    
    delta = m["recall_at_5"] - base_m["recall_at_5"]
    
    fold_recalls = []
    for f in range(5):
        f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
        m_f = compute_metrics(preds, labels, f_qids)
        fold_recalls.append(m_f["recall_at_5"])

    res = {
        "name": cfg["name"],
        "num_rules": len(cfg["rules"]),
        "active_rules": cfg["rules"],
        "recall_at_5": m["recall_at_5"],
        "precision_at_5": m["precision_at_5"],
        "multi_gold_recall_at_5": m["multi_gold_recall_at_5"],
        "delta_vs_baseline": delta,
        "fold_recalls": fold_recalls,
        "runtime_seconds": elapsed,
    }
    redundancy_results.append(res)

    print(f"\n{cfg['name']} ({len(cfg['rules'])} rules):")
    print(f"  Recall@5:        {m['recall_at_5']:.6f} ({delta:+.6f}, {delta*100:+.3f}pp vs 11-rule)")
    print(f"  Precision@5:     {m['precision_at_5']:.6f}")
    print(f"  Multi-Gold R@5:  {m['multi_gold_recall_at_5']:.6f}")
    print(f"  Fold Recalls:    {[round(r, 6) for r in fold_recalls]}")

out_path = OUT_DIR / "redundancy_audit_results.json"
out_path.write_text(json.dumps({"baseline": base_m, "configurations": redundancy_results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSuccessfully wrote redundancy audit results to {out_path}")
