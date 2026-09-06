"""Unified Multi-Specialist Feature Fusion via XGBRanker (Level-Wise Tree Regularization).

Hypothesis H4.1:
Level-wise depth-constrained gradient boosted trees via XGBRanker (objective="rank:ndcg",
eval_metric="ndcg@5", max_depth=4) on the 119D unified feature matrix will provide
superior regularization against feature dominance compared to greedy leaf-wise trees,
improving boundary candidate discrimination and setting a new 5-fold OOF Recall@5 state-of-the-art.

Hypothesis H4.2:
Combining level-wise XGBRanker with leaf-wise LGBMRanker and supervised BM25 Profile LTR
yields multi-architectural tree ensemble synergy.

Strict outer fold isolation:
Uses pre-aligned, zero-leakage training and test feature matrices from cache/gemini/exp_unified_ltr/.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from exp_final.data import SourceStore

CACHE_DIR = ROOT / "cache/gemini/exp_unified_ltr"
LGBM_RESULTS = ROOT / "results/gemini/exp_unified_ltr"
PROFILE_DIR = ROOT / "results/exp_final_retrieval/profile_ltr_probe"
MEMORY_DIR = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT_DIR = ROOT / "results/gemini/exp_unified_xgb"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_xgb_fold(outer: int, configs: dict[str, dict]) -> dict[str, Any]:
    print(f"\n=================== STARTING XGB FOLD {outer} ===================", flush=True)
    fold_cache = CACHE_DIR / f"fold_{outer}"
    fold_out = OUT_DIR / f"fold_{outer}"
    fold_out.mkdir(parents=True, exist_ok=True)

    if (fold_out / "FOLD_REPORT.json").exists():
        report = read_json(fold_out / "FOLD_REPORT.json")
        if all(cfg in report for cfg in configs):
            print(f"Fold {outer} already completed, reusing results.", flush=True)
            return report

    labels, _ = get_canonical_labels()
    folds = get_cv_folds()

    store = SourceStore(ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite")
    marker = read_json(ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{outer}/outer-ml.json")
    train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
    test_qids = [str(q) for q in folds[f"fold_{outer}"] if labels.get(str(q))]

    # Load pre-aligned 119D training unified matrix
    train_unified = np.load(fold_cache / "train_unified.f32.npy", mmap_mode="r")
    groups = []
    target = []
    for q in train_qids:
        docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
        groups.append(len(docs))
        target.extend(d in labels[q] for d in docs)
    y_train = np.asarray(target, dtype=np.int8)

    # Load pre-aligned 119D test matrix
    test_matrix = np.load(fold_cache / "test_matrix.f32.npy")
    test_docs = read_json(fold_cache / "test_docs.json")
    test_groups = read_json(fold_cache / "test_groups.json")
    test_ends = np.cumsum([0] + test_groups)

    store.close()

    print(f"Fold {outer}: {len(train_qids)} train queries ({train_unified.shape[0]} rows), "
          f"{len(test_qids)} test queries ({test_matrix.shape[0]} rows)", flush=True)

    fold_results = {}
    for cfg_name, cfg in configs.items():
        print(f"\nFitting XGBRanker [{cfg_name}] on Fold {outer} (GPU)...", flush=True)
        t0 = time.time()
        model = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            random_state=4200 + outer,
            n_jobs=4,
            tree_method="hist",
            device="cuda",
            **cfg,
        )
        model.fit(train_unified, y_train, group=groups)
        fit_time = time.time() - t0
        print(f"  Fitted in {fit_time:.1f}s", flush=True)

        scores = model.predict(test_matrix)
        preds = {}
        for index, (qid, docs) in enumerate(zip(test_qids, test_docs)):
            vals = scores[test_ends[index]:test_ends[index + 1]]
            order = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(vals[i]), docs[i]))]
            preds[qid] = order

        metrics = compute_metrics(preds, labels, test_qids, k=5)
        print(f"  Fold {outer} [{cfg_name}]: Recall@5 = {metrics['recall_at_5']:.6f}, "
              f"Precision@5 = {metrics['precision_at_5']:.6f}, MRR@5 = {metrics['mrr_at_5']:.6f}", flush=True)

        write_json(fold_out / f"{cfg_name}_PREDICTIONS.json", preds)
        fold_results[cfg_name] = {
            "config": cfg,
            "metrics": metrics,
            "fit_time_seconds": fit_time,
        }

    write_json(fold_out / "FOLD_REPORT.json", fold_results)
    return fold_results


def evaluate_oof(configs: dict[str, dict]) -> None:
    print("\n=================== COMPUTING FULL 5-FOLD OOF METRICS ===================", flush=True)
    labels, _ = get_canonical_labels()
    folds = get_cv_folds()
    all_eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]

    # Load reference baselines
    mem_preds = {}
    prof_preds = {}
    lgbm_preds = read_json(LGBM_RESULTS / "l15_t5_OOF_PREDICTIONS.json")
    for f in range(5):
        mem_preds.update(read_json(MEMORY_DIR / f"fold_{f}/PREDICTIONS.json"))
        prof_preds.update(read_json(PROFILE_DIR / "l15_t5/PREDICTIONS.json"))

    mem_metrics = compute_metrics(mem_preds, labels, all_eval_qids)
    prof_metrics = compute_metrics(prof_preds, labels, all_eval_qids)
    lgbm_metrics = compute_metrics(lgbm_preds, labels, all_eval_qids)

    summary = {
        "status": "COMPLETE_GEMINI_UNIFIED_XGB_OOF",
        "feature_count": 119,
        "baselines": {
            "memory_ltr": mem_metrics,
            "profile_ltr_l15_t5": prof_metrics,
            "unified_lgbm_l15_t5": lgbm_metrics,
        },
        "systems": {},
        "blends": {},
    }

    for cfg_name in configs:
        pooled_preds = {}
        fold_metrics_list = []
        for f in range(5):
            pred_f = read_json(OUT_DIR / f"fold_{f}/{cfg_name}_PREDICTIONS.json")
            pooled_preds.update(pred_f)
            f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
            fold_metrics_list.append(compute_metrics(pred_f, labels, f_qids))

        oof_metrics = compute_metrics(pooled_preds, labels, all_eval_qids)
        write_json(OUT_DIR / f"{cfg_name}_OOF_PREDICTIONS.json", pooled_preds)

        boot_vs_prof = paired_bootstrap(prof_preds, pooled_preds, labels, all_eval_qids)
        boot_vs_lgbm = paired_bootstrap(lgbm_preds, pooled_preds, labels, all_eval_qids)
        boot_vs_mem = paired_bootstrap(mem_preds, pooled_preds, labels, all_eval_qids)

        fold_deltas_vs_prof = [
            fold_metrics_list[f]["recall_at_5"] -
            compute_metrics(prof_preds, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall_at_5"]
            for f in range(5)
        ]
        fold_deltas_vs_lgbm = [
            fold_metrics_list[f]["recall_at_5"] -
            compute_metrics(lgbm_preds, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall_at_5"]
            for f in range(5)
        ]

        summary["systems"][cfg_name] = {
            "oof_metrics": oof_metrics,
            "fold_metrics": fold_metrics_list,
            "delta_vs_profile_ltr": oof_metrics["recall_at_5"] - prof_metrics["recall_at_5"],
            "delta_vs_unified_lgbm": oof_metrics["recall_at_5"] - lgbm_metrics["recall_at_5"],
            "fold_deltas_vs_profile": fold_deltas_vs_prof,
            "fold_deltas_vs_lgbm": fold_deltas_vs_lgbm,
            "nonnegative_folds_vs_profile": sum(d >= 0 for d in fold_deltas_vs_prof),
            "nonnegative_folds_vs_lgbm": sum(d >= 0 for d in fold_deltas_vs_lgbm),
            "bootstrap_vs_profile": boot_vs_prof,
            "bootstrap_vs_lgbm": boot_vs_lgbm,
            "bootstrap_vs_memory": boot_vs_mem,
        }

        print(f"\n=======================================================")
        print(f"SYSTEM: {cfg_name} 5-Fold OOF Results")
        print(f"  Recall@5:    {oof_metrics['recall_at_5']:.6f} "
              f"(LGBM: {lgbm_metrics['recall_at_5']:.6f}, Delta: {oof_metrics['recall_at_5'] - lgbm_metrics['recall_at_5']:+.6f}) "
              f"(Profile: {prof_metrics['recall_at_5']:.6f}, Delta: {oof_metrics['recall_at_5'] - prof_metrics['recall_at_5']:+.6f})")
        print(f"  Precision@5: {oof_metrics['precision_at_5']:.6f}")
        print(f"  MRR@5:       {oof_metrics['mrr_at_5']:.6f}")
        print(f"  Bootstrap vs LGBM: p={boot_vs_lgbm['p_value']:.4f}, W/L/T={boot_vs_lgbm['wins']}/{boot_vs_lgbm['losses']}/{boot_vs_lgbm['ties']}")
        print(f"  Fold deltas vs LGBM: {[round(d, 6) for d in fold_deltas_vs_lgbm]}")
        print(f"  Fold deltas vs Profile: {[round(d, 6) for d in fold_deltas_vs_prof]}")

    # Now evaluate Blends between Best XGB, Unified LGBM, and Profile LTR
    best_xgb_cfg = max(configs.keys(), key=lambda c: summary["systems"][c]["oof_metrics"]["recall_at_5"])
    best_xgb_preds = read_json(OUT_DIR / f"{best_xgb_cfg}_OOF_PREDICTIONS.json")

    print("\n--- Evaluating Multi-Architectural Blends ---", flush=True)
    blend_weights = [
        (0.50, 0.50, 0.00, "0.50_xgb + 0.50_lgbm"),
        (0.40, 0.40, 0.20, "0.40_xgb + 0.40_lgbm + 0.20_profile"),
        (0.45, 0.35, 0.20, "0.45_xgb + 0.35_lgbm + 0.20_profile"),
        (0.35, 0.45, 0.20, "0.35_xgb + 0.45_lgbm + 0.20_profile"),
        (0.60, 0.20, 0.20, "0.60_xgb + 0.20_lgbm + 0.20_profile"),
    ]

    for wx, wl, wp, bname in blend_weights:
        b_preds = {}
        for q in all_eval_qids:
            sc = {}
            for r, d in enumerate(best_xgb_preds[q][:64], 1):
                sc[d] = sc.get(d, 0.0) + wx / (32 + r)
            for r, d in enumerate(lgbm_preds[q][:64], 1):
                sc[d] = sc.get(d, 0.0) + wl / (32 + r)
            if wp > 0:
                for r, d in enumerate(prof_preds[q][:64], 1):
                    sc[d] = sc.get(d, 0.0) + wp / (32 + r)
            b_preds[q] = sorted(sc.keys(), key=lambda d: (-sc[d], d))

        b_metrics = compute_metrics(b_preds, labels, all_eval_qids)
        b_boot = paired_bootstrap(lgbm_preds, b_preds, labels, all_eval_qids)
        summary["blends"][bname] = {
            "metrics": b_metrics,
            "bootstrap_vs_lgbm": b_boot,
            "delta_vs_lgbm": b_metrics["recall_at_5"] - lgbm_metrics["recall_at_5"],
            "delta_vs_prof": b_metrics["recall_at_5"] - prof_metrics["recall_at_5"],
        }
        write_json(OUT_DIR / f"blend_{bname.replace(' ', '_').replace('+', '_')}.json", b_preds)
        print(f"  Blend [{bname}]: Recall@5 = {b_metrics['recall_at_5']:.6f} "
              f"(Delta vs LGBM: {b_metrics['recall_at_5'] - lgbm_metrics['recall_at_5']:+.6f}, "
              f"Delta vs Prof: {b_metrics['recall_at_5'] - prof_metrics['recall_at_5']:+.6f}, "
              f"MRR@5: {b_metrics['mrr_at_5']:.6f}, p={b_boot['p_value']:.4f}, W/L/T={b_boot['wins']}/{b_boot['losses']}/{b_boot['ties']})")

    write_json(OUT_DIR / "UNIFIED_XGB_SUMMARY.json", summary)
    print(f"\nSaved complete summary to {OUT_DIR / 'UNIFIED_XGB_SUMMARY.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run Gemini Unified XGBRanker OOF")
    parser.add_argument("--fold", default="all", choices=["0", "1", "2", "3", "4", "all"])
    args = parser.parse_args()

    configs = {
        "xgb_d4_n300": dict(max_depth=4, learning_rate=0.05, n_estimators=300),
        "xgb_d4_n400": dict(max_depth=4, learning_rate=0.05, n_estimators=400),
    }

    folds_to_run = range(5) if args.fold == "all" else [int(args.fold)]
    for f in folds_to_run:
        run_xgb_fold(f, configs)

    if len(folds_to_run) == 5:
        evaluate_oof(configs)


if __name__ == "__main__":
    main()
