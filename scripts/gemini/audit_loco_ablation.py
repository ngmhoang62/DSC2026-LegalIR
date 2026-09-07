"""Phase 1: Strict Leave-One-Component-Out (LOCO) Ablation Audit.

Evaluates the exact contribution of all 15 individual components in the authoritative
strict-valid baseline (0.955042):
- 4 Rankers (XGB-145D, LGBM-145D, XGB-131D, Profile LTR)
- 11 Post-processing rules (Forward Kinship, Multi-Statute, Inverse Kinship, Deep Kinship,
  Guarded Inverse Law, Topic Law, Preamble Citation, Midrank Inverse, Technical Standard,
  Corporate Entity, Superseded Statute Dedup)

For each component, computes:
1. Action count (queries where top-5 ranking is altered)
2. Delta Recall@5 when removed (importance = baseline - ablated)
3. Delta Precision@5
4. Delta Multi-gold Recall@5
5. Fold-by-fold deltas (Folds 0..4)
6. Wins / Losses / Ties vs baseline (query-level difference)
7. Runtime contribution & implementation complexity
8. Provenance & overlap summary
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

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
print("PHASE 1: STRICT LEAVE-ONE-COMPONENT-OUT (LOCO) ABLATION AUDIT")
print("=" * 80)

# 1. Load canonical labels, folds, queries, doc labels, preambles
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

# 2. Load 4 base model predictions
print("Loading base model predictions...")
xgb_tuned = {}
for f in range(5):
    p = json.loads((ROOT / f"results/gemini/exp_145d_tuned/fold_{f}/xgb_145d_tuned_PREDICTIONS.json").read_text(encoding="utf-8"))
    xgb_tuned.update(p)

xgb131 = json.loads((ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
lgb145 = json.loads((ROOT / "results/gemini/exp_145d_ranker/lgbm_145d_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))
prof = json.loads((ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json").read_text(encoding="utf-8"))
print(f"Loaded predictions for all 4 rankers across {len(eval_qids)} evaluable queries.")

# 3. Base blend helper
BASE_CFG = (0.41, 0.29, 0.12, 0.18, 18, 10, 15, 15)

def blend_predictions(qids, cfg, use_xgb=True, use_lgb=True, use_131=True, use_prof=True):
    w_x, w_l, w_13, w_p, k_x, k_l, k_13, k_p = cfg
    # Re-normalize weights if some models are ablated
    weights = [
        w_x if use_xgb else 0.0,
        w_l if use_lgb else 0.0,
        w_13 if use_131 else 0.0,
        w_p if use_prof else 0.0,
    ]
    tot_w = sum(weights)
    if tot_w > 0:
        weights = [w / tot_w for w in weights]
    w_x, w_l, w_13, w_p = weights

    blended = {}
    for q in qids:
        sc = {}
        cands = []
        if use_xgb and w_x > 0:
            docs = xgb_tuned.get(q, [])[:k_x]
            cands.extend(docs)
            for r, d in enumerate(docs):
                sc[d] = sc.get(d, 0.0) + w_x / (r + 1.0)
        if use_lgb and w_l > 0:
            docs = lgb145.get(q, [])[:k_l]
            cands.extend(docs)
            for r, d in enumerate(docs):
                sc[d] = sc.get(d, 0.0) + w_l / (r + 1.0)
        if use_131 and w_13 > 0:
            docs = xgb131.get(q, [])[:k_13]
            cands.extend(docs)
            for r, d in enumerate(docs):
                sc[d] = sc.get(d, 0.0) + w_13 / (r + 1.0)
        if use_prof and w_p > 0:
            docs = prof.get(q, [])[:k_p]
            cands.extend(docs)
            for r, d in enumerate(docs):
                sc[d] = sc.get(d, 0.0) + w_p / (r + 1.0)

        cand_docs = dict.fromkeys(cands)
        if not cand_docs:
            cand_docs = dict.fromkeys(prof.get(q, []))
        blended[q] = sorted(cand_docs.keys(), key=lambda d: (-sc.get(d, 0.0), d))
    return blended


def run_kinship_pipeline(
    rankings: dict[str, list[str]],
    qids: list[str],
    skip_component: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Runs kinship pipeline, optionally skipping one named component."""
    cur = rankings
    action_counts = {}

    # 1. Forward Kinship
    if skip_component != "forward_kinship":
        nxt, cnt = apply_kinship_promotion(cur, doc_labels, qids, top_k=2, cand_max=9)
        action_counts["forward_kinship"] = cnt
        cur = nxt

    # 2. Multi-Statute Promotion
    if skip_component != "multi_statute":
        nxt, cnt = apply_multi_statute_promotion(cur, doc_labels, questions, qids)
        action_counts["multi_statute"] = cnt
        cur = nxt

    # 3. Inverse Kinship
    if skip_component != "inverse_kinship":
        nxt, cnt = apply_inverse_kinship_promotion(cur, doc_labels, qids, top_k=2, cand_max=9)
        action_counts["inverse_kinship"] = cnt
        cur = nxt

    # 4. Deep Kinship
    if skip_component != "deep_kinship":
        nxt, cnt = apply_deep_statutory_kinship(cur, doc_labels, qids, top_k=2, cand_max=15)
        action_counts["deep_kinship"] = cnt
        cur = nxt

    # 5. Guarded Inverse Law
    if skip_component != "guarded_inverse_law":
        nxt, cnt = apply_guarded_inverse_law(cur, doc_labels, qids, top_k=2, cand_max=12)
        action_counts["guarded_inverse_law"] = cnt
        cur = nxt

    # 6. Topic Law Promotion
    if skip_component != "topic_law":
        nxt, cnt = apply_topic_law_promotion(cur, doc_labels, questions, qids, cand_max=6)
        action_counts["topic_law"] = cnt
        cur = nxt

    # 7. Preamble Citation Kinship
    if skip_component != "preamble_citation":
        nxt, cnt = apply_preamble_citation_kinship(cur, doc_labels, doc_preambles, qids, top_k=2, cand_max=8)
        action_counts["preamble_citation"] = cnt
        cur = nxt

    # 8. Hierarchical Midrank Inverse Kinship
    if skip_component != "midrank_inverse":
        nxt, cnt = apply_hierarchical_midrank_inverse_kinship(cur, doc_labels, qids, cand_max=12)
        action_counts["midrank_inverse"] = cnt
        cur = nxt

    # 9. Technical Standard Kinship
    if skip_component != "technical_standard":
        nxt, cnt = apply_technical_standard_kinship(cur, doc_labels, qids, cand_max=10)
        action_counts["technical_standard"] = cnt
        cur = nxt

    # 10. Corporate Entity Kinship
    if skip_component != "corporate_entity":
        nxt, cnt = apply_corporate_entity_kinship(cur, doc_labels, questions, qids)
        action_counts["corporate_entity"] = cnt
        cur = nxt

    # 11. Superseded Statute Dedup
    if skip_component != "superseded_dedup":
        nxt, cnt = apply_superseded_statute_dedup(
            cur, qids, superseded_pairs=VERIFIED_SUPERSEDED_STATUTE_PAIRS[:49]
        )
        action_counts["superseded_dedup"] = cnt
        cur = nxt

    return cur, action_counts


# Compute Authoritative Baseline Full 5-fold OOF
print("\nComputing Authoritative Baseline...")
base_blended = blend_predictions(eval_qids, BASE_CFG)
base_preds, base_actions = run_kinship_pipeline(base_blended, eval_qids, skip_component=None)
base_metrics = compute_metrics(base_preds, labels, eval_qids)

base_fold_recalls = []
for f in range(5):
    f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
    m_f = compute_metrics(base_preds, labels, f_qids)
    base_fold_recalls.append(m_f["recall_at_5"])

print(f"Authoritative Baseline Recall@5:        {base_metrics['recall_at_5']:.6f}")
print(f"Authoritative Baseline Precision@5:     {base_metrics['precision_at_5']:.6f}")
print(f"Authoritative Baseline Multi-Gold R@5:  {base_metrics['multi_gold_recall_at_5']:.6f}")
print(f"Fold Recalls: {[round(r, 6) for r in base_fold_recalls]}")
print(f"Base Kinship Action Counts: {base_actions}")

# Define all components to ablate
COMPONENTS_TO_ABLATE = [
    # Rankers
    ("xgb_145d", "ranker", "Tuned XGB-145D GBDT Ranker (depth 4, GPU)"),
    ("lgbm_145d", "ranker", "LightGBM 145D LambdaRank (num_leaves 15, CPU)"),
    ("xgb_131d", "ranker", "Authority-Enhanced XGB-131D GBDT Ranker"),
    ("profile_ltr", "ranker", "Supervised BM25 Label-Profile Memory LTR"),
    # Post-processing Kinship Rules
    ("forward_kinship", "rule", "Forward statutory kinship: Base doc promotes amendment (top_k=2, cand_max=9)"),
    ("multi_statute", "rule", "Multi-statute promotion: Promotes secondary statute when query asks for multiple"),
    ("inverse_kinship", "rule", "Inverse statutory kinship: Amendment promotes base doc (top_k=2, cand_max=9)"),
    ("deep_kinship", "rule", "Deep guarded statutory kinship: Extended candidate pool (top_k=2, cand_max=15)"),
    ("guarded_inverse_law", "rule", "Guarded inverse law: Guiding decree promotes parent law (top_k=2, cand_max=12)"),
    ("topic_law", "rule", "Topic law promotion: Exact topic match law promoted into Rank 5 (cand_max=6)"),
    ("preamble_citation", "rule", "Preamble citation kinship: Preamble citations promoted into Rank 5 (top_k=2, cand_max=8)"),
    ("midrank_inverse", "rule", "Hierarchical midrank inverse kinship: Mid-rank amendments promote parent (cand_max=12)"),
    ("technical_standard", "rule", "Technical standard kinship: TCVN/QCVN statutory pair co-retrieval (cand_max=10)"),
    ("corporate_entity", "rule", "Corporate entity kinship: Enterprise & investment code alignment"),
    ("superseded_dedup", "rule", "Cross-fitted superseded statute dedup: Demotes obsolete statute if successor is present"),
]

loco_reports = []

for comp_name, comp_type, comp_mech in COMPONENTS_TO_ABLATE:
    t0 = time.time()
    
    # 1. Generate ablated rankings
    if comp_type == "ranker":
        use_x = comp_name != "xgb_145d"
        use_l = comp_name != "lgbm_145d"
        use_13 = comp_name != "xgb_131d"
        use_p = comp_name != "profile_ltr"
        ablated_blend = blend_predictions(eval_qids, BASE_CFG, use_xgb=use_x, use_lgb=use_l, use_131=use_13, use_prof=use_p)
        ablated_preds, _ = run_kinship_pipeline(ablated_blend, eval_qids, skip_component=None)
    else:
        # Ablate specific rule
        ablated_preds, _ = run_kinship_pipeline(base_blended, eval_qids, skip_component=comp_name)

    elapsed = time.time() - t0

    # 2. Compute metrics
    abl_metrics = compute_metrics(ablated_preds, labels, eval_qids)
    
    # Delta when REMOVED: (baseline - ablated). Positive delta means baseline is BETTER than ablated (component HELPS).
    delta_r5 = base_metrics["recall_at_5"] - abl_metrics["recall_at_5"]
    delta_p5 = base_metrics["precision_at_5"] - abl_metrics["precision_at_5"]
    delta_mgr = base_metrics["multi_gold_recall_at_5"] - abl_metrics["multi_gold_recall_at_5"]

    # 3. Fold-by-fold deltas
    fold_deltas = []
    abl_fold_recalls = []
    for f in range(5):
        f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
        m_f = compute_metrics(ablated_preds, labels, f_qids)
        abl_fold_recalls.append(m_f["recall_at_5"])
        fold_deltas.append(base_fold_recalls[f] - m_f["recall_at_5"])

    # 4. Action count & Wins / Losses / Ties
    # Count queries where top-5 is altered
    altered_queries = 0
    wins = 0   # baseline has higher recall than ablated
    losses = 0 # baseline has lower recall than ablated
    ties = 0   # identical recall
    
    for q in eval_qids:
        base_top5 = set(base_preds[q][:5])
        abl_top5 = set(ablated_preds[q][:5])
        if base_top5 != abl_top5:
            altered_queries += 1
            
        g = labels[q]
        h_base = len(base_top5 & g) / len(g)
        h_abl = len(abl_top5 & g) / len(g)
        if h_base > h_abl + 1e-9:
            wins += 1
        elif h_base < h_abl - 1e-9:
            losses += 1
        else:
            ties += 1

    action_count = base_actions.get(comp_name, altered_queries) if comp_type == "rule" else altered_queries

    report = {
        "component": comp_name,
        "type": comp_type,
        "mechanism": comp_mech,
        "action_count": action_count,
        "altered_queries": altered_queries,
        "importance_delta_recall_at_5": delta_r5,
        "delta_precision_at_5": delta_p5,
        "delta_multi_gold_recall_at_5": delta_mgr,
        "ablated_recall_at_5": abl_metrics["recall_at_5"],
        "fold_deltas": fold_deltas,
        "ablated_fold_recalls": abl_fold_recalls,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_query_gain": wins - losses,
        "eval_time_seconds": elapsed,
    }
    loco_reports.append(report)

    print(f"\n[{comp_type.upper()}] {comp_name}:")
    print(f"  Importance (Δ Recall@5): {delta_r5:+.6f} ({delta_r5*100:+.3f}pp) | Ablated R@5: {abl_metrics['recall_at_5']:.6f}")
    print(f"  Action count: {action_count} queries | Altered top-5: {altered_queries} queries")
    print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties} (Net gain: {wins - losses:+d} queries)")
    print(f"  Fold Deltas: {[round(d, 6) for d in fold_deltas]}")

# Save full results to JSON
loco_summary = {
    "baseline": base_metrics,
    "base_fold_recalls": base_fold_recalls,
    "components": loco_reports,
}

out_path = OUT_DIR / "loco_ablation_results.json"
out_path.write_text(json.dumps(loco_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSuccessfully wrote LOCO ablation results to {out_path}")
