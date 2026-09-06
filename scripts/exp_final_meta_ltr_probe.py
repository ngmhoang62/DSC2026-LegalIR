"""Leakage-controlled level-2 document ranker over EXP-FINAL OOF systems.

Every base prediction is out-of-fold.  For each target fold, this script fits
the meta ranker only on the other four OOF folds and evaluates on the target.
It learns document-level agreement and specialist rank patterns rather than a
single global RRF weight.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exp_final_retrieval/meta_ltr_probe"
DEPTH = 64


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_systems():
    memory = {}
    for fold in range(5):
        memory.update(read(ROOT / f"results/exp_final_retrieval/memory_ltr_probe/fold_{fold}/PREDICTIONS.json"))
    systems = {
        "memory": memory,
        "profile_l7": read(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l7_t30/PREDICTIONS.json"),
        "profile_l15": read(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json"),
        "capacity_l15": read(ROOT / "results/exp_final_retrieval/memory_capacity_probe/l15_m50_t5/PREDICTIONS.json"),
        "capacity_l31": read(ROOT / "results/exp_final_retrieval/memory_capacity_probe/l31_m50_t10/PREDICTIONS.json"),
        "kernel_l15": read(ROOT / "results/exp_final_retrieval/kernel_ltr_probe/l15_t5/PREDICTIONS.json"),
    }
    expected = set(memory)
    # EXP-112 produced these rankings with an adapter trained on the other four
    # folds even though calibration selected beta=0 for the submitted system.
    # Read only the required head instead of deserializing ~4.5 GB of full-rank
    # JSON artifacts.
    adapted_cache = OUT / f"adapted_e5_top{DEPTH}.json"
    if adapted_cache.exists():
        adapted = read(adapted_cache)
    else:
        adapted = {}
        folds = read(ROOT / "cache/cv_folds.json")
        for fold in range(5):
            folder = ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{fold}/test-query"
            for qid in folds[f"fold_{fold}"]:
                qid = str(qid)
                if qid not in expected:
                    continue
                order = []
                inside = False
                with (folder / f"{qid}.json").open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if '"order"' in line:
                            inside = True
                        elif inside:
                            match = re.search(r'"([^"]+)"', line)
                            if match:
                                order.append(match.group(1))
                            if len(order) == DEPTH:
                                break
                if len(order) != DEPTH:
                    raise ValueError(f"Incomplete adapted E5 head: {qid}")
                adapted[qid] = order
        write(adapted_cache, adapted)
    systems["adapted_e5"] = adapted
    for name, rows in systems.items():
        if set(rows) != expected:
            raise ValueError(f"Prediction qid mismatch: {name}")
    return systems


def feature_names(names):
    result = []
    for name in names:
        result += [f"{name}_present", f"{name}_rr0", f"{name}_rr32",
                   f"{name}_top1", f"{name}_top3", f"{name}_top5",
                   f"{name}_top10", f"{name}_top20"]
    result += ["present_count", "top1_votes", "top3_votes", "top5_votes", "top10_votes",
               "rr0_mean", "rr0_max", "rr0_std", "rr32_sum", "rr32_std",
               "best_rank", "mean_observed_rank", "rank_std"]
    return result


def rows_for_query(qid, systems, names):
    ranks = {name: {doc: rank for rank, doc in enumerate(systems[name][qid][:DEPTH], 1)} for name in names}
    docs = sorted(set().union(*(set(local) for local in ranks.values())))
    matrix = np.zeros((len(docs), len(feature_names(names))), dtype=np.float32)
    for row, doc in enumerate(docs):
        observed = []
        rr0, rr32 = [], []
        column = 0
        for name in names:
            rank = ranks[name].get(doc)
            present = float(rank is not None)
            a = 0.0 if rank is None else 1.0 / rank
            b = 0.0 if rank is None else 1.0 / (32 + rank)
            matrix[row, column:column+8] = (
                present, a, b, float(rank == 1), float(rank is not None and rank <= 3),
                float(rank is not None and rank <= 5), float(rank is not None and rank <= 10),
                float(rank is not None and rank <= 20),
            )
            column += 8
            if rank is not None:
                observed.append(rank)
            rr0.append(a); rr32.append(b)
        matrix[row, column:] = (
            len(observed), sum(rank == 1 for rank in observed), sum(rank <= 3 for rank in observed),
            sum(rank <= 5 for rank in observed), sum(rank <= 10 for rank in observed),
            np.mean(rr0), np.max(rr0), np.std(rr0), np.sum(rr32), np.std(rr32),
            min(observed), np.mean(observed), np.std(observed),
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite meta feature")
    return docs, matrix


def metrics(rankings, labels, qids):
    recall, precision, multi, mrr = [], [], [], []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = rankings[qid][:5]; hits = len(set(top) & gold); value = hits / len(gold)
        recall.append(value); precision.append(hits / 5)
        if len(gold) > 1: multi.append(value)
        first = next((rank for rank, doc in enumerate(top, 1) if doc in gold), None)
        mrr.append(0 if first is None else 1 / first)
    return {"recall_at_5": float(np.mean(recall)), "precision_at_5": float(np.mean(precision)),
            "multi_gold_recall_at_5": float(np.mean(multi)), "mrr_at_5": float(np.mean(mrr)),
            "queries": len(recall)}


def main():
    sys.path.insert(0, str(ROOT / "src"))
    import lightgbm as lgb
    import exp109b_encoder_complementarity as old
    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    systems = load_systems(); names = list(systems)
    configs = {
        "l7_t30": dict(num_leaves=7, min_child_samples=50, lambdarank_truncation_level=30),
        "l15_t5": dict(num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5),
    }
    all_predictions = {name: {} for name in configs}
    fold_reports = {}
    for outer in range(5):
        train_qids = [q for fold in range(5) if fold != outer for q in folds[f"fold_{fold}"] if labels.get(q)]
        test_qids = [q for q in folds[f"fold_{outer}"] if labels.get(q)]
        train_rows, train_groups, target = [], [], []
        for index, qid in enumerate(train_qids):
            docs, values = rows_for_query(qid, systems, names)
            train_rows.append(values); train_groups.append(len(docs)); target.extend(doc in labels[qid] for doc in docs)
            if (index + 1) % 1000 == 0:
                print(f"outer={outer} meta_train_rows={index+1}/{len(train_qids)}", flush=True)
        xtrain = np.concatenate(train_rows)
        test_rows, test_groups, test_docs = [], [], []
        for qid in test_qids:
            docs, values = rows_for_query(qid, systems, names)
            test_rows.append(values); test_groups.append(len(docs)); test_docs.append(docs)
        xtest = np.concatenate(test_rows); ends = np.cumsum([0] + test_groups)
        fold_reports[f"fold_{outer}"] = {}
        for name, config in configs.items():
            model = lgb.LGBMRanker(objective="lambdarank", learning_rate=.05, n_estimators=300,
                                   feature_fraction=1., bagging_fraction=1., deterministic=True,
                                   force_col_wise=True, n_jobs=4, random_state=6112, verbosity=-1,
                                   **config)
            model.fit(xtrain, np.asarray(target, dtype=np.int8), group=train_groups, eval_at=[5])
            scores = model.predict(xtest); local = {}
            for index, (qid, docs) in enumerate(zip(test_qids, test_docs)):
                values = scores[ends[index]:ends[index+1]]
                head = [docs[row] for row in sorted(range(len(docs)), key=lambda row: (-float(values[row]), docs[row]))]
                head_set = set(head)
                local[qid] = head + [doc for doc in systems["memory"][qid] if doc not in head_set]
            all_predictions[name].update(local)
            fold_reports[f"fold_{outer}"][name] = metrics(local, labels, test_qids)
            print(f"outer={outer} meta_ltr={name} recall={fold_reports[f'fold_{outer}'][name]['recall_at_5']:.9f}", flush=True)
        write(OUT / f"fold_{outer}/MANIFEST.json", {
            "outer": outer, "meta_train_folds": [f for f in range(5) if f != outer],
            "test_fold": outer, "base_predictions": "strict OOF", "depth": DEPTH,
            "feature_names": feature_names(names), "systems": names,
        })
    qids = [q for q in systems["memory"] if labels.get(q)]
    baseline = metrics(systems["memory"], labels, qids)
    aggregate = []
    for name, predictions in all_predictions.items():
        value = metrics(predictions, labels, qids)
        fold_deltas = []
        for fold in range(5):
            ids = [q for q in folds[f"fold_{fold}"] if labels.get(q)]
            fold_deltas.append(fold_reports[f"fold_{fold}"][name]["recall_at_5"] - metrics(systems["memory"], labels, ids)["recall_at_5"])
        aggregate.append({"system": name, "config": configs[name], "metrics": value,
                          "delta": value["recall_at_5"] - baseline["recall_at_5"],
                          "fold_deltas": fold_deltas, "nonnegative_folds": sum(delta >= 0 for delta in fold_deltas)})
        write(OUT / name / "PREDICTIONS.json", predictions)
    aggregate.sort(key=lambda item: (item["metrics"]["recall_at_5"], item["metrics"]["precision_at_5"], item["metrics"]["mrr_at_5"]), reverse=True)
    report = {"status": "COMPLETE_STRICT_META_LTR_PROBE",
              "scope_warning": "Each meta model trains only on other folds' OOF base predictions.",
              "baseline": baseline, "depth": DEPTH, "systems": names,
              "feature_names": feature_names(names), "folds": fold_reports, "aggregate": aggregate}
    write(OUT / "META_LTR_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
