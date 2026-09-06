"""Unified Multi-Specialist Feature Fusion (Joint 119D LambdaMART) for Gemini namespace.

Hypothesis H1:
Jointly training a single LambdaMART ranker with all 119 features
(72 Base + 14 Memory + 12 Profile + 21 Kernel)
allows tree splits to find optimal non-linear cross-specialist interactions,
surpassing separate rankers and post-hoc rank blends.

Strict outer fold isolation:
Zero label leakage into features, profiles, kernels, or models.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap

# Apply robust Data compatibility patches for permissions
import exp109b_encoder_complementarity as old
old.canonical_labels = get_canonical_labels

def _load_minimal_metadata():
    p = ROOT / "results/exp110p_semantic_label_prototype/colab_bundle/input/parent_metadata_minimal.jsonl"
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["doc_id"])] = row
    return out

old.build_parent_text_metadata = _load_minimal_metadata

from exp_final.data import Data, SourceStore
from exp_final.fusion import features as base_features
from exp_final_memory_ltr_probe import memory_features, normalize, support_index, MEMORY_NAMES
from exp_final_supervised_profile_probe import build_profiles
from exp_final_profile_ltr_probe import profile_features, PROFILE_CONFIGS
from exp_final_kernel_ltr_probe import (
    build_channel, top_indices, channel_evidence, kernel_features,
    KERNEL_NAMES, NEIGHBOR_DEPTH, POWER,
)

LEGACY_CACHE = ROOT / "cache/exp112_task_adaptive_retrieval"
MEMORY_PROBE_DIR = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
PROFILE_PROBE_DIR = ROOT / "results/exp_final_retrieval/profile_ltr_probe"
KERNEL_PROBE_DIR = ROOT / "results/exp_final_retrieval/kernel_ltr_probe"

OUT = ROOT / "results/gemini/exp_unified_ltr"
CACHE = ROOT / "cache/gemini/exp_unified_ltr"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_outer_fold(outer: int, configs: dict) -> dict:
    print(f"\n=================== STARTING OUTER FOLD {outer} ===================", flush=True)
    fold_cache = CACHE / f"fold_{outer}"
    fold_out = OUT / f"fold_{outer}"
    fold_cache.mkdir(parents=True, exist_ok=True)
    fold_out.mkdir(parents=True, exist_ok=True)
    
    if (fold_out / "FOLD_REPORT.json").exists():
        report = read_json(fold_out / "FOLD_REPORT.json")
        if all(cfg in report for cfg in configs):
            print(f"Fold {outer} already completed, reusing results.", flush=True)
            return report
    
    labels, audit = get_canonical_labels()
    folds = get_cv_folds()
    fold_of = {str(q): f for f in range(5) for q in folds[f"fold_{f}"]}
    
    data = Data()
    store = SourceStore(LEGACY_CACHE / "sources.sqlite")
    store.jina_enabled = True
    
    marker = read_json(LEGACY_CACHE / f"outer/fold_{outer}/outer-ml.json")
    train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
    test_qids = [str(q) for q in folds[f"fold_{outer}"] if labels.get(str(q))]
    
    print(f"Fold {outer}: {len(train_qids)} train queries, {len(test_qids)} test queries", flush=True)
    
    # 1. Training data: load pre-aligned blocks
    mem_block = np.load(MEMORY_PROBE_DIR / f"fold_{outer}/train_augmented.f32.npy", mmap_mode="r")
    prof_block = np.load(PROFILE_PROBE_DIR / f"fold_{outer}/train_profile.f32.npy", mmap_mode="r")
    kern_block = np.load(KERNEL_PROBE_DIR / f"fold_{outer}/train_kernel.f32.npy", mmap_mode="r")
    
    n_rows = mem_block.shape[0]
    total_cols = mem_block.shape[1] + prof_block.shape[1] + kern_block.shape[1]
    assert total_cols == 119, f"Expected 119 cols, got {total_cols}"
    
    unified_train_path = fold_cache / "train_unified.f32.npy"
    if not unified_train_path.exists() or unified_train_path.stat().st_size != n_rows * total_cols * 4 + 128:
        print(f"Assembling unified training matrix ({n_rows} rows x {total_cols} cols)...", flush=True)
        t0 = time.time()
        train_unified = np.lib.format.open_memmap(
            unified_train_path, mode="w+", dtype=np.float32, shape=(n_rows, total_cols)
        )
        train_unified[:, :mem_block.shape[1]] = mem_block
        train_unified[:, mem_block.shape[1]:mem_block.shape[1] + prof_block.shape[1]] = prof_block
        train_unified[:, mem_block.shape[1] + prof_block.shape[1]:] = kern_block
        train_unified.flush()
        print(f"Assembled training matrix in {time.time() - t0:.1f}s", flush=True)
    else:
        print("Reusing existing assembled training matrix.", flush=True)
        train_unified = np.load(unified_train_path, mmap_mode="r")
        
    # Build training target and groups
    groups = []
    target = []
    for q in train_qids:
        docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
        groups.append(len(docs))
        target.extend(d in labels[q] for d in docs)
    assert sum(groups) == n_rows, f"Group sum mismatch: {sum(groups)} vs {n_rows}"
    y_train = np.asarray(target, dtype=np.int8)
    
    # 2. Test data: assemble test feature matrix
    test_matrix_path = fold_cache / "test_matrix.f32.npy"
    test_docs_path = fold_cache / "test_docs.json"
    test_groups_path = fold_cache / "test_groups.json"
    
    if test_matrix_path.exists() and test_docs_path.exists() and test_groups_path.exists():
        print("Reusing pre-computed test feature matrix.", flush=True)
        test_matrix = np.load(test_matrix_path)
        test_docs = read_json(test_docs_path)
        test_groups = read_json(test_groups_path)
    else:
        print(f"Computing test feature matrix for {len(test_qids)} queries...", flush=True)
        t0 = time.time()
        
        # Load LAL query vectors for memory features
        with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz", allow_pickle=False) as z:
            vector_ids = list(map(str, z["query_ids"].tolist()))
            lal_vectors = normalize(z["vectors"])
        qrow = {qid: row for row, qid in enumerate(vector_ids)}
        
        by_doc, frequency = support_index(labels, train_qids)
        lal_support = lal_vectors[[qrow[q] for q in train_qids]]
        lal_test = np.asarray(lal_vectors[[qrow[q] for q in test_qids]] @ lal_support.T, dtype=np.float32)
        
        # Build Profile model
        test_profile_model = build_profiles(data.questions, labels, train_qids)
        
        # Build Kernel TF-IDF channels
        from sklearn.feature_extraction.text import TfidfVectorizer
        train_text = [data.questions[q] for q in train_qids]
        test_text = [data.questions[q] for q in test_qids]
        
        word_support, word_test = build_channel(
            TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 3),
                            min_df=2, max_df=.995, max_features=180_000, sublinear_tf=True,
                            dtype=np.float32), train_text, test_text
        )
        char_support, char_test = build_channel(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                            max_features=160_000, sublinear_tf=True, dtype=np.float32),
            train_text, test_text
        )
        all_allowed = np.ones(len(train_qids), dtype=bool)
        
        test_rows = []
        test_docs = []
        test_groups = []
        
        for begin in range(0, len(test_qids), 64):
            end = min(len(test_qids), begin + 64)
            w_sim = (word_test[begin:end] @ word_support.T).tocsr()
            c_sim = (char_test[begin:end] @ char_support.T).tocsr()
            for local, test_index in enumerate(range(begin, end)):
                qid = test_qids[test_index]
                docs = store.candidates(qid)
                
                # 1. Base features (72)
                xb = np.asarray(base_features(data, store, qid, docs, 2))
                
                # 2. Memory features (14)
                xm = memory_features(lal_test[test_index], docs, train_qids, labels, by_doc, frequency)
                
                # 3. Profile features (12)
                xp = profile_features(data.questions[qid], docs, test_profile_model)
                
                # 4. Kernel features (21)
                wc, wv = top_indices(w_sim.getrow(local), all_allowed)
                cc, cv = top_indices(c_sim.getrow(local), all_allowed)
                word_ev = channel_evidence(wc, wv, train_qids, labels)
                char_ev = channel_evidence(cc, cv, train_qids, labels)
                xk, _ = kernel_features(word_ev, char_ev, docs, frequency)
                
                row_block = np.concatenate([xb, xm, xp, xk], axis=1)
                test_rows.append(row_block)
                test_docs.append(docs)
                test_groups.append(len(docs))
                
            if (end % 256 == 0) or (end == len(test_qids)):
                print(f"  test features: {end}/{len(test_qids)} elapsed={time.time()-t0:.1f}s", flush=True)
                
        test_matrix = np.concatenate(test_rows, axis=0)
        np.save(test_matrix_path, test_matrix)
        write_json(test_docs_path, test_docs)
        write_json(test_groups_path, test_groups)
        print(f"Saved test matrix shape {test_matrix.shape} in {time.time() - t0:.1f}s", flush=True)
        
    test_ends = np.cumsum([0] + test_groups)
    
    # 3. Train models and evaluate
    import lightgbm as lgb
    fold_results = {}
    
    for cfg_name, cfg in configs.items():
        print(f"\nFitting LightGBM [{cfg_name}] on Fold {outer}...", flush=True)
        t0 = time.time()
        model = lgb.LGBMRanker(
            objective="lambdarank",
            learning_rate=0.05,
            n_estimators=300,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            deterministic=True,
            force_col_wise=True,
            n_jobs=4,
            random_state=4200 + outer,
            verbosity=-1,
            **cfg,
        )
        model.fit(train_unified, y_train, group=groups, eval_at=[5])
        train_time = time.time() - t0
        print(f"  Fitted in {train_time:.1f}s", flush=True)
        
        scores = model.predict(test_matrix)
        preds = {}
        for index, (qid, docs) in enumerate(zip(test_qids, test_docs)):
            vals = scores[test_ends[index]:test_ends[index + 1]]
            order = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(vals[i]), docs[i]))]
            preds[qid] = order
            
        metrics = compute_metrics(preds, labels, test_qids, k=5)
        print(f"  Fold {outer} [{cfg_name}]: Recall@5 = {metrics['recall_at_5']:.6f}, Precision@5 = {metrics['precision_at_5']:.6f}, MRR@5 = {metrics['mrr_at_5']:.6f}", flush=True)
        
        write_json(fold_out / f"{cfg_name}_PREDICTIONS.json", preds)
        fold_results[cfg_name] = {
            "config": cfg,
            "metrics": metrics,
            "train_time_seconds": train_time,
        }
        
    store.close()
    write_json(fold_out / "FOLD_REPORT.json", fold_results)
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Run Gemini Unified LTR Feature Fusion")
    parser.add_argument("--fold", default="all", choices=["0", "1", "2", "3", "4", "all"])
    args = parser.parse_args()
    
    configs = {
        "l15_t5": dict(num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5),
        "l7_t30": dict(num_leaves=7, min_child_samples=50, lambdarank_truncation_level=30),
        "l23_t5": dict(num_leaves=23, min_child_samples=50, lambdarank_truncation_level=5),
    }
    
    folds_to_run = range(5) if args.fold == "all" else [int(args.fold)]
    all_fold_results = {}
    for f in folds_to_run:
        all_fold_results[f] = run_outer_fold(f, configs)
        
    if len(folds_to_run) == 5:
        print("\n=================== COMPUTING FULL 5-FOLD OOF METRICS ===================", flush=True)
        labels, _ = get_canonical_labels()
        folds = get_cv_folds()
        
        # Load baseline predictions for comparison
        mem_preds = {}
        prof_preds = {}
        for f in range(5):
            mem_preds.update(read_json(MEMORY_PROBE_DIR / f"fold_{f}/PREDICTIONS.json"))
            prof_preds.update(read_json(PROFILE_PROBE_DIR / f"l15_t5/PREDICTIONS.json"))
            
        summary = {
            "status": "COMPLETE_GEMINI_UNIFIED_LTR_OOF",
            "feature_count": 119,
            "feature_composition": {
                "base_dense_sparse_jina": 72,
                "semantic_case_memory": 14,
                "supervised_bm25_profile": 12,
                "tfidf_kernel_posterior": 21,
            },
            "systems": {},
        }
        
        all_eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]
        mem_metrics = compute_metrics(mem_preds, labels, all_eval_qids)
        prof_metrics = compute_metrics(prof_preds, labels, all_eval_qids)
        summary["baselines"] = {
            "memory_ltr": mem_metrics,
            "profile_ltr_l15_t5": prof_metrics,
        }
        
        for cfg_name in configs:
            pooled_preds = {}
            fold_metrics_list = []
            for f in range(5):
                pred_f = read_json(OUT / f"fold_{f}/{cfg_name}_PREDICTIONS.json")
                pooled_preds.update(pred_f)
                f_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
                fold_metrics_list.append(compute_metrics(pred_f, labels, f_qids))
                
            oof_metrics = compute_metrics(pooled_preds, labels, all_eval_qids)
            write_json(OUT / f"{cfg_name}_OOF_PREDICTIONS.json", pooled_preds)
            
            # Paired bootstrap vs Profile LTR
            boot_vs_prof = paired_bootstrap(prof_preds, pooled_preds, labels, all_eval_qids)
            boot_vs_mem = paired_bootstrap(mem_preds, pooled_preds, labels, all_eval_qids)
            
            fold_deltas_vs_prof = [
                fold_metrics_list[f]["recall_at_5"] - compute_metrics(prof_preds, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall_at_5"]
                for f in range(5)
            ]
            
            summary["systems"][cfg_name] = {
                "oof_metrics": oof_metrics,
                "fold_metrics": fold_metrics_list,
                "delta_vs_profile_ltr": oof_metrics["recall_at_5"] - prof_metrics["recall_at_5"],
                "delta_vs_memory_ltr": oof_metrics["recall_at_5"] - mem_metrics["recall_at_5"],
                "fold_deltas_vs_profile": fold_deltas_vs_prof,
                "nonnegative_folds_vs_profile": sum(d >= 0 for d in fold_deltas_vs_prof),
                "bootstrap_vs_profile": boot_vs_prof,
                "bootstrap_vs_memory": boot_vs_mem,
            }
            
            print(f"\n--- {cfg_name} OOF RESULTS ---")
            print(f"Recall@5:    {oof_metrics['recall_at_5']:.6f} (Profile LTR: {prof_metrics['recall_at_5']:.6f}, Delta: {oof_metrics['recall_at_5'] - prof_metrics['recall_at_5']:+.6f})")
            print(f"Precision@5: {oof_metrics['precision_at_5']:.6f}")
            print(f"MRR@5:       {oof_metrics['mrr_at_5']:.6f}")
            print(f"Bootstrap 95% CI vs Profile: [{boot_vs_prof['ci_95_lower']:+.6f}, {boot_vs_prof['ci_95_upper']:+.6f}], p={boot_vs_prof['p_value']:.4f}, W/L/T={boot_vs_prof['wins']}/{boot_vs_prof['losses']}/{boot_vs_prof['ties']}")
            print(f"Fold deltas vs Profile: {[round(d, 6) for d in fold_deltas_vs_prof]}")
            
        write_json(OUT / "UNIFIED_LTR_SUMMARY.json", summary)
        print(f"\nSaved full summary to {OUT / 'UNIFIED_LTR_SUMMARY.json'}")


if __name__ == "__main__":
    main()
