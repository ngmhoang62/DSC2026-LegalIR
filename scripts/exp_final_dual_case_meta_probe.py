"""Four-way cross-fitted meta-ranker over the Fold-4 dual-case sources.

The dual LAL adapter never saw Fold 4.  This probe further divides Fold 4 into
four deterministic buckets, trains a document-level ranker on three buckets,
and predicts the fourth.  It tests whether learned routing can realize the
observed choice-oracle without paying for four more GPU adapter fits first.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from exp_final_dual_case_adapter_probe import (
    choice_oracle, metrics, normalize, prototype_orders, read, weighted_rrf, write,
)
from exp_final_meta_ltr_probe import feature_names, rows_for_query

CACHE = ROOT / "cache/exp_final_retrieval/dual_case_adapter_probe/fold_4"
OUT = ROOT / "results/exp_final_retrieval/dual_case_adapter_probe/META_CROSSFIT_REPORT.json"


def bucket(qid, count=4):
    return int(hashlib.sha256((str(qid) + "|dual-case-meta-v1").encode()).hexdigest()[:16], 16) % count


def load_content(qids, epoch=2):
    return {q: list(map(str, read(CACHE / f"content-epoch-{epoch}/{q}.json")["order"])) for q in qids}


def main():
    import lightgbm as lgb
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data

    labels, _ = old.canonical_labels(); folds = read(ROOT / "cache/cv_folds.json")
    train_qids = [q for fold in range(4) for q in folds[f"fold_{fold}"] if labels.get(q)]
    qids = [q for q in folds["fold_4"] if labels.get(q)]
    data = Data()
    anchor = read(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json")
    memory = read(ROOT / "results/exp_final_retrieval/memory_ltr_probe/fold_4/PREDICTIONS.json")
    content = load_content(qids, epoch=2)
    support = normalize(np.load(CACHE / "vectors-epoch-2-train/vectors.f32.npy", mmap_mode="r"))
    target = normalize(np.load(CACHE / "vectors-epoch-2-test/vectors.f32.npy", mmap_mode="r"))
    prototypes = prototype_orders(target, support, train_qids, labels, data.doc_ids)
    systems = {
        "anchor": {q: anchor[q] for q in qids},
        "memory": memory,
        "dual_content": content,
        "dual_proto_max": {q: prototypes["max"][i] for i, q in enumerate(qids)},
        "dual_proto_logmean": {q: prototypes["logmeanexp"][i] for i, q in enumerate(qids)},
    }
    names = list(systems)
    configs = {
        "l7_t30": dict(num_leaves=7, min_child_samples=50, lambdarank_truncation_level=30),
        "l15_t5": dict(num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5),
    }
    predictions = {name: {} for name in configs}; manifests = {}
    for target_bucket in range(4):
        train = [q for q in qids if bucket(q) != target_bucket]
        test = [q for q in qids if bucket(q) == target_bucket]
        train_rows, groups, y = [], [], []
        for q in train:
            docs, values = rows_for_query(q, systems, names)
            train_rows.append(values); groups.append(len(docs)); y.extend(doc in labels[q] for doc in docs)
        xtrain = np.concatenate(train_rows); y = np.asarray(y, dtype=np.int8)
        test_rows, test_docs = [], []
        for q in test:
            docs, values = rows_for_query(q, systems, names)
            test_rows.append(values); test_docs.append(docs)
        xtest = np.concatenate(test_rows); ends = np.cumsum([0] + [len(docs) for docs in test_docs])
        manifests[str(target_bucket)] = {
            "train_queries": len(train), "test_queries": len(test),
            "positive_train_rows": int(np.count_nonzero(y)), "training_rows": len(y),
        }
        for name, config in configs.items():
            model = lgb.LGBMRanker(
                objective="lambdarank", learning_rate=.05, n_estimators=300,
                feature_fraction=1., bagging_fraction=1., deterministic=True,
                force_col_wise=True, n_jobs=4, random_state=7112, verbosity=-1,
                **config,
            )
            model.fit(xtrain, y, group=groups, eval_at=[5])
            scores = model.predict(xtest)
            for index, (q, docs) in enumerate(zip(test, test_docs)):
                values = scores[ends[index]:ends[index + 1]]
                head = [docs[row] for row in sorted(range(len(docs)), key=lambda row: (-float(values[row]), docs[row]))]
                seen = set(head); predictions[name][q] = head + [doc for doc in anchor[q] if doc not in seen]
        print(f"meta_bucket={target_bucket} train={len(train)} test={len(test)}", flush=True)
    if any(set(rows) != set(qids) for rows in predictions.values()):
        raise ValueError("Meta cross-fit did not predict every Fold-4 query exactly once")
    baseline = metrics(anchor, labels, qids); systems_metrics = {}; trials = []
    for name, rows in predictions.items():
        systems_metrics[name] = metrics(rows, labels, qids)
        for constant in (0, 32):
            for weight in (.10, .20, .30, .50, 1.0):
                if weight == 1.0:
                    fused = rows
                else:
                    # With at least half the mass on one of the two sources, a
                    # document below rank 200 in both cannot outrank five
                    # top-5 anchor documents. The metric only consumes top 5.
                    fused = {q: weighted_rrf(
                        {"anchor": anchor[q][:200], "meta": rows[q][:200]},
                        {"anchor": 1 - weight, "meta": weight}, constant,
                    ) for q in qids}
                measured = metrics(fused, labels, qids)
                trials.append({
                    "model": name, "constant": constant, "weight": weight,
                    "metrics": measured, "delta_vs_anchor": measured["recall_at_5"] - baseline["recall_at_5"],
                })
    trials.sort(key=lambda row: (
        row["metrics"]["recall_at_5"], row["metrics"]["precision_at_5"],
        row["metrics"]["multi_gold_recall_at_5"], row["metrics"]["mrr_at_5"],
    ), reverse=True)
    report = {
        "status": "COMPLETE_DUAL_CASE_META_4WAY_CROSSFIT",
        "scope": "Fold 4 adapter-OOF; 4-way meta cross-fit inside exposed development Fold 4.",
        "feature_names": feature_names(names), "systems": names, "bucket_manifests": manifests,
        "baseline": baseline, "raw_meta": systems_metrics,
        "choice_oracles": {name: choice_oracle(anchor, rows, labels, qids) for name, rows in predictions.items()},
        "top_trials": trials,
    }
    write(OUT, report); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
