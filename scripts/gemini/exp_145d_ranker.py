"""145D Statutory & Cross-Source Enhanced Ranker (Gemini H41).

Hypothesis H41:
Augmenting the 131D feature space with 14 advanced statutory kinship, exact legal keyphrase,
official letter penalty, and cross-source consensus features (145D) allows the GBDT
rankers (XGBoost GPU / LightGBM) to break through upstream ranking blindspots.

Zero label leakage:
Operates strictly on public document labels, query text, and frozen source rankings.
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
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.advanced_features import AdvancedFeatureExtractor
from exp_final.data import SourceStore

CACHE_131 = ROOT / "cache/gemini/exp_authority_131d"
CACHE_145 = ROOT / "cache/gemini/exp_145d_ranker"
OUT_DIR = ROOT / "results/gemini/exp_145d_ranker"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
SOURCES_DB = ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite"
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


def load_source_maps() -> dict[str, dict[str, dict[str, int]]]:
    import sqlite3
    db = sqlite3.connect(f"file:{SOURCES_DB}?mode=ro", uri=True)
    source_maps = {s: {} for s in ("e5", "lal", "bm25", "trigram", "jina")}
    for s in source_maps:
        for q, payload in db.execute("SELECT q, payload FROM sources WHERE source=?", (s,)):
            items = json.loads(payload)
            source_maps[s][str(q)] = {str(it["doc_id"]): int(it["rank"]) for it in items}
    db.close()
    return source_maps


def run_fold(outer: int, extractor: AdvancedFeatureExtractor, questions: dict[str, str], source_maps: dict[str, Any]) -> dict[str, Any]:
    print(f"\n=================== STARTING 145D FOLD {outer} ===================", flush=True)
    fold_cache = CACHE_145 / f"fold_{outer}"
    fold_out = OUT_DIR / f"fold_{outer}"
    fold_cache.mkdir(parents=True, exist_ok=True)
    fold_out.mkdir(parents=True, exist_ok=True)

    labels, _ = get_canonical_labels()
    folds = get_cv_folds()

    store = SourceStore(SOURCES_DB)
    marker = read_json(ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{outer}/outer-ml.json")
    train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
    test_qids = [str(q) for q in folds[f"fold_{outer}"] if labels.get(str(q))]

    # 1. Load / Prepare 145D Train Matrix
    train_145_path = fold_cache / "train_145d.f32.npy"
    train_131_path = CACHE_131 / f"fold_{outer}/train_131d.f32.npy"

    train_docs_list = []
    groups = []
    target = []
    for q in train_qids:
        docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
        groups.append(len(docs))
        target.extend(d in labels[q] for d in docs)
        train_docs_list.append(docs)
    y_train = np.asarray(target, dtype=np.int8)

    if not train_145_path.exists():
        print(f"Extracting 14D advanced features for {len(train_qids)} train queries...", flush=True)
        t0 = time.time()
        train_14 = extractor.extract_block(questions, train_qids, train_docs_list, source_maps)
        print(f"  Extracted train 14D in {time.time()-t0:.1f}s, shape={train_14.shape}", flush=True)

        train_131 = np.load(train_131_path, mmap_mode="r")
        assert train_131.shape[0] == train_14.shape[0]
        assert train_131.shape[1] == 131 and train_14.shape[1] == 14

        print("Assembling 145D train matrix memmap...", flush=True)
        t0 = time.time()
        mmap_train = np.lib.format.open_memmap(
            train_145_path, mode="w+", dtype=np.float32, shape=(train_131.shape[0], 145)
        )
        mmap_train[:, :131] = train_131
        mmap_train[:, 131:] = train_14
        mmap_train.flush()
        print(f"  Saved 145D train matrix in {time.time()-t0:.1f}s", flush=True)
        train_145 = np.load(train_145_path, mmap_mode="r")
    else:
        print("Reusing existing 145D train matrix.", flush=True)
        train_145 = np.load(train_145_path, mmap_mode="r")

    # 2. Load / Prepare 145D Test Matrix
    test_145_path = fold_cache / "test_145d.f32.npy"
    test_131_path = CACHE_131 / f"fold_{outer}/test_131d.f32.npy"
    test_docs = read_json(ROOT / f"cache/gemini/exp_unified_ltr/fold_{outer}/test_docs.json")
    test_groups = read_json(ROOT / f"cache/gemini/exp_unified_ltr/fold_{outer}/test_groups.json")
    test_ends = np.cumsum([0] + test_groups)

    if not test_145_path.exists():
        print(f"Extracting 14D advanced features for {len(test_qids)} test queries...", flush=True)
        t0 = time.time()
        test_14 = extractor.extract_block(questions, test_qids, test_docs, source_maps)
        print(f"  Extracted test 14D in {time.time()-t0:.1f}s, shape={test_14.shape}", flush=True)

        test_131 = np.load(test_131_path)
        assert test_131.shape[0] == test_14.shape[0]
        test_145 = np.concatenate([test_131, test_14], axis=1)
        np.save(test_145_path, test_145)
        print(f"  Saved 145D test matrix, shape={test_145.shape}", flush=True)
    else:
        print("Reusing existing 145D test matrix.", flush=True)
        test_145 = np.load(test_145_path)

    store.close()

    fold_results = {}

    # Model A: XGBRanker on 145D (GPU)
    print(f"\nFitting 145D XGBRanker on Fold {outer} (GPU)...", flush=True)
    t0 = time.time()
    xgb_model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        random_state=4200 + outer,
        n_jobs=4,
        tree_method="hist",
        device="cuda",
    )
    xgb_model.fit(train_145, y_train, group=groups)
    fit_time_xgb = time.time() - t0
    print(f"  Fitted in {fit_time_xgb:.1f}s", flush=True)

    s_xgb = xgb_model.predict(test_145)
    preds_xgb = {}
    for idx, (qid, docs) in enumerate(zip(test_qids, test_docs)):
        vals = s_xgb[test_ends[idx]:test_ends[idx + 1]]
        preds_xgb[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(vals[i]), docs[i]))]

    m_xgb = compute_metrics(preds_xgb, labels, test_qids)
    print(f"  Fold {outer} [145D XGBRanker]: Recall@5 = {m_xgb['recall_at_5']:.6f}, "
          f"Precision@5 = {m_xgb['precision_at_5']:.6f}, MRR@5 = {m_xgb['mrr_at_5']:.6f}", flush=True)
    write_json(fold_out / "xgb_145d_PREDICTIONS.json", preds_xgb)

    # Model B: LGBMRanker on 145D
    print(f"\nFitting 145D LGBMRanker on Fold {outer}...", flush=True)
    t0 = time.time()
    lgb_model = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=15,
        min_child_samples=50,
        lambdarank_truncation_level=5,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        random_state=4200 + outer,
        verbosity=-1,
    )
    lgb_model.fit(train_145, y_train, group=groups, eval_at=[5])
    fit_time_lgb = time.time() - t0
    print(f"  Fitted in {fit_time_lgb:.1f}s", flush=True)

    s_lgb = lgb_model.predict(test_145)
    preds_lgb = {}
    for idx, (qid, docs) in enumerate(zip(test_qids, test_docs)):
        vals = s_lgb[test_ends[idx]:test_ends[idx + 1]]
        preds_lgb[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(vals[i]), docs[i]))]

    m_lgb = compute_metrics(preds_lgb, labels, test_qids)
    print(f"  Fold {outer} [145D LGBMRanker]: Recall@5 = {m_lgb['recall_at_5']:.6f}, "
          f"Precision@5 = {m_lgb['precision_at_5']:.6f}, MRR@5 = {m_lgb['mrr_at_5']:.6f}", flush=True)
    write_json(fold_out / "lgbm_145d_PREDICTIONS.json", preds_lgb)

    fold_results["xgb_145d"] = {"metrics": m_xgb, "fit_time": fit_time_xgb}
    fold_results["lgbm_145d"] = {"metrics": m_lgb, "fit_time": fit_time_lgb}
    write_json(fold_out / "FOLD_REPORT.json", fold_results)
    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0..4)")
    args = parser.parse_args()

    extractor = AdvancedFeatureExtractor(EVIDENCE_DB)
    questions = load_queries()
    print(f"Loaded {len(questions)} query texts.")
    
    print("Loading source maps from sources.sqlite...")
    source_maps = load_source_maps()
    print("Source maps loaded successfully.")

    run_fold(args.fold, extractor, questions, source_maps)


if __name__ == "__main__":
    main()
