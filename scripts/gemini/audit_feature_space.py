"""Phase 3: Feature Space Audit & Progressive Pruning (145D).

Analyzes the 145D feature matrix:
1. Per-fold feature importance stability for XGB-145D (Folds 0..4)
2. Identification of zero / near-zero importance and unstable features
3. Feature family drop ablation:
   - Full 145D (Base 72D + Memory 14D + Profile 12D + Authority 33D + Advanced 14D)
   - 131D (Drop 14D Advanced Features)
   - 98D (Drop 33D Authority Features)
   - 86D (Drop 12D Profile Features)
   - 72D (Base Features Only)
   - Compact Top-N Feature Subsets (e.g. Top 60, Top 40)
4. Evaluates exact 5-fold OOF standalone Recall@5 and precision for each feature set.
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
from gemini.metrics import compute_metrics
from exp_final.data import SourceStore

OUT_DIR = ROOT / "results/gemini/complexity_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / "cache/gemini/exp_145d_ranker"
SOURCES_DB = ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite"

labels, _ = get_canonical_labels()
folds = get_cv_folds()
eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]

store = SourceStore(SOURCES_DB)

print("=" * 80)
print("PHASE 3: FEATURE SPACE AUDIT & PRUNING (145D)")
print("=" * 80)

# Feature family boundaries:
# 0..71: Base EXP-112 (72D)
# 72..85: Memory Features (14D)
# 86..97: Profile Features (12D)
# 98..130: Authority Features (33D)
# 131..144: Advanced Features (14D)
FEATURE_FAMILIES = {
    "base_72d": list(range(0, 72)),
    "memory_14d": list(range(72, 86)),
    "profile_12d": list(range(86, 98)),
    "authority_33d": list(range(98, 131)),
    "advanced_14d": list(range(131, 145)),
}

# 1. Feature Importance Audit on Fold 0..4
print("\n--- 1. Evaluating Per-Fold Feature Importances for XGB-145D ---")
all_fold_importances = []
fold_models = {}

for f in range(5):
    t0 = time.time()
    marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
    train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
    
    train_docs_by_q = [list(dict.fromkeys(store.candidates(q) + sorted(labels[q]))) for q in train_qids]
    train_groups = [len(docs) for docs in train_docs_by_q]
    train_y = np.array([d in labels[q] for q, docs in zip(train_qids, train_docs_by_q) for d in docs], dtype=np.int8)

    X_train = np.load(CACHE_DIR / f"fold_{f}/train_145d.f32.npy")
    
    model = xgb.XGBRanker(
        n_estimators=350,
        learning_rate=0.08,
        max_depth=4,
        tree_method="hist",
        device="cuda",
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    model.fit(X_train, train_y, group=train_groups, verbose=False)
    imp = model.feature_importances_
    all_fold_importances.append(imp)
    fold_models[f] = model
    print(f"Fold {f} model fitted in {time.time()-t0:.1f}s | Non-zero features: {(imp > 0).sum()}/145")

mean_imp = np.mean(all_fold_importances, axis=0)
std_imp = np.std(all_fold_importances, axis=0)

# Identify zero / near-zero features across all folds
zero_imp_features = [i for i in range(145) if mean_imp[i] == 0]
low_imp_features = [i for i in range(145) if 0 < mean_imp[i] < 0.001]

print(f"\nTotal completely zero-importance features across all folds: {len(zero_imp_features)}/145")
print(f"Total low-importance features (mean imp < 0.001): {len(low_imp_features)}/145")

# Per-family importance summary
print("\nFeature Family Importance Breakdown:")
family_imp_summary = {}
for fam_name, col_indices in FEATURE_FAMILIES.items():
    fam_mean = float(mean_imp[col_indices].mean())
    fam_sum = float(mean_imp[col_indices].sum())
    fam_nonzeros = int((mean_imp[col_indices] > 0).sum())
    family_imp_summary[fam_name] = {
        "dim": len(col_indices),
        "total_importance": fam_sum,
        "mean_importance": fam_mean,
        "active_features": fam_nonzeros,
    }
    print(f"  {fam_name:15s}: dim={len(col_indices):2d}, total_imp={fam_sum:.4f} ({fam_sum*100:.1f}%), active={fam_nonzeros}/{len(col_indices)}")

# 2. Progressive Feature Family Ablation
# Evaluate exact 5-fold OOF standalone recall for each feature subset
FEATURE_SUBSETS = [
    ("Full 145D", list(range(145))),
    ("131D (Drop Advanced)", list(range(131))),
    ("117D (Drop Authority 33D, Keep Advanced)", list(range(98)) + list(range(131, 145))),
    ("98D (Drop Authority + Advanced)", list(range(98))),
    ("86D (Drop Profile + Authority + Advanced)", list(range(86))),
    ("72D (Base Features Only)", list(range(72))),
]

# Add Top-K compact feature sets based on mean importance
sorted_features_by_imp = np.argsort(-mean_imp).tolist()
FEATURE_SUBSETS.append(("Top-60 Features", sorted_features_by_imp[:60]))
FEATURE_SUBSETS.append(("Top-40 Features", sorted_features_by_imp[:40]))

print("\n--- 2. Progressive Feature Subset 5-Fold OOF Evaluation ---")
ablation_results = []

for subset_name, col_indices in FEATURE_SUBSETS:
    t0 = time.time()
    oof_preds = {}
    fold_recalls = []

    for f in range(5):
        marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
        train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
        test_qids = [str(q) for q in folds[f"fold_{f}"] if labels.get(str(q))]

        train_docs_by_q = [list(dict.fromkeys(store.candidates(q) + sorted(labels[q]))) for q in train_qids]
        train_groups = [len(docs) for docs in train_docs_by_q]
        train_y = np.array([d in labels[q] for q, docs in zip(train_qids, train_docs_by_q) for d in docs], dtype=np.int8)

        test_docs_by_q = [store.candidates(q) for q in test_qids]
        test_groups = [len(docs) for docs in test_docs_by_q]

        X_train = np.load(CACHE_DIR / f"fold_{f}/train_145d.f32.npy")[:, col_indices]
        X_test = np.load(CACHE_DIR / f"fold_{f}/test_145d.f32.npy")[:, col_indices]

        model = xgb.XGBRanker(
            n_estimators=350,
            learning_rate=0.08,
            max_depth=4,
            tree_method="hist",
            device="cuda",
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
        model.fit(X_train, train_y, group=train_groups, verbose=False)
        test_scores = model.predict(X_test)

        ends = np.cumsum([0] + test_groups)
        for qi, (q, docs) in enumerate(zip(test_qids, test_docs_by_q)):
            local = test_scores[ends[qi]:ends[qi+1]]
            oof_preds[q] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(local[i]), docs[i]))]

        m_f = compute_metrics({q: oof_preds[q] for q in test_qids}, labels, test_qids)
        fold_recalls.append(m_f["recall_at_5"])

    overall_m = compute_metrics(oof_preds, labels, eval_qids)
    elapsed = time.time() - t0

    res = {
        "subset_name": subset_name,
        "feature_dim": len(col_indices),
        "recall_at_5": overall_m["recall_at_5"],
        "precision_at_5": overall_m["precision_at_5"],
        "multi_gold_recall_at_5": overall_m["multi_gold_recall_at_5"],
        "fold_recalls": fold_recalls,
        "runtime_seconds": elapsed,
    }
    ablation_results.append(res)

    print(f"\n{subset_name} ({len(col_indices)}D):")
    print(f"  5-Fold Standalone Recall@5: {overall_m['recall_at_5']:.6f} | Prec@5: {overall_m['precision_at_5']:.6f}")
    print(f"  Fold Recalls: {[round(r, 6) for r in fold_recalls]} | Runtime: {elapsed:.1f}s")

# Save summary
feature_audit_summary = {
    "zero_importance_indices": zero_imp_features,
    "family_summary": family_imp_summary,
    "subsets": ablation_results,
}
out_path = OUT_DIR / "feature_audit_results.json"
out_path.write_text(json.dumps(feature_audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSuccessfully wrote feature audit results to {out_path}")
