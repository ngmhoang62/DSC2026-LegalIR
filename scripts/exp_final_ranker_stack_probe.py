"""Cross-fitted rank-level stacking of completed EXP-FINAL specialists.

The recipe used on each target fold is selected only from the other four OOF
folds.  Input systems are themselves outer-fold predictions.  This is a small
second-level generalization test, not a post-hoc best-per-fold oracle.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exp_final_retrieval/ranker_stack_probe"


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
    paths = {
        "memory": None,
        "profile_l7": ROOT / "results/exp_final_retrieval/profile_ltr_probe/l7_t30/PREDICTIONS.json",
        "profile_l15": ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json",
        "capacity_l15": ROOT / "results/exp_final_retrieval/memory_capacity_probe/l15_m50_t5/PREDICTIONS.json",
        "capacity_l31": ROOT / "results/exp_final_retrieval/memory_capacity_probe/l31_m50_t10/PREDICTIONS.json",
        "kernel_l15": ROOT / "results/exp_final_retrieval/kernel_ltr_probe/l15_t5/PREDICTIONS.json",
    }
    systems = {"memory": memory}
    for name, path in paths.items():
        if path is not None:
            systems[name] = read(path)
    qids = set(memory)
    for name, rows in systems.items():
        if set(rows) != qids:
            raise ValueError(f"Prediction qid mismatch: {name}")
    return systems


def recipes(names):
    output = []
    for name in names:
        output.append({"weights": {name: 1.0}, "k": 32})
    specialists = [name for name in names if name != "memory"]
    for name in specialists:
        for anchor_weight in (.50, .65, .80, .90, .95):
            for k in (0, 32, 80):
                output.append({"weights": {"memory": anchor_weight, name: 1-anchor_weight}, "k": k})
    for left, right in itertools.combinations(specialists, 2):
        for left_weight in (.05, .10, .20):
            for right_weight in (.05, .10, .20):
                if left_weight + right_weight > .40:
                    continue
                for k in (0, 32):
                    output.append({"weights": {"memory": 1-left_weight-right_weight,
                                                left: left_weight, right: right_weight}, "k": k})
    unique = {}
    for recipe in output:
        key = json.dumps(recipe, sort_keys=True)
        unique[key] = recipe
    return list(unique.values())


def fuse(qid, systems, recipe, depth=64):
    scores = {}
    for name, weight in recipe["weights"].items():
        for rank, doc in enumerate(systems[name][qid][:depth], 1):
            scores[doc] = scores.get(doc, 0.0) + weight / (recipe["k"] + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def contribution(top, gold):
    hits = len(set(top[:5]) & gold)
    first = next((rank for rank, doc in enumerate(top[:5], 1) if doc in gold), None)
    return np.asarray((hits / len(gold), hits / 5, (hits / len(gold)) if len(gold) > 1 else 0.0,
                       float(len(gold) > 1), 0.0 if first is None else 1.0 / first), dtype=np.float64)


def metric(values):
    count = values[5]
    multi_count = values[3]
    return {"recall_at_5": values[0]/count, "precision_at_5": values[1]/count,
            "multi_gold_recall_at_5": values[2]/max(1, multi_count),
            "mrr_at_5": values[4]/count, "queries": int(count)}


def key(values):
    row = metric(values)
    return (row["recall_at_5"], row["precision_at_5"], row["multi_gold_recall_at_5"], row["mrr_at_5"])


def main():
    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old
    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    fold_of = {str(q): fold for fold in range(5) for q in folds[f"fold_{fold}"]}
    systems = load_systems()
    qids = sorted(q for q in systems["memory"] if labels.get(q))
    candidates = recipes(list(systems))
    print(f"systems={len(systems)} recipes={len(candidates)} queries={len(qids)}", flush=True)
    # Per recipe and per fold: recall sum, precision sum, multi recall sum,
    # multi count, MRR sum, query count.
    fold_values = np.zeros((len(candidates), 5, 6), dtype=np.float64)
    for recipe_index, recipe in enumerate(candidates):
        for qid in qids:
            fold = fold_of[qid]
            fold_values[recipe_index, fold, :5] += contribution(fuse(qid, systems, recipe), labels[qid])
            fold_values[recipe_index, fold, 5] += 1
        if (recipe_index + 1) % 25 == 0:
            print(f"screen={recipe_index+1}/{len(candidates)}", flush=True)
    selected = {}
    stacked = {}
    for outer in range(5):
        training = fold_values.sum(axis=1) - fold_values[:, outer]
        winner = max(range(len(candidates)), key=lambda index: (key(training[index]), -index))
        recipe = candidates[winner]
        selected[f"fold_{outer}"] = {
            "recipe_index": winner, "recipe": recipe,
            "meta_train_metrics": metric(training[winner]),
            "target_metrics": metric(fold_values[winner, outer]),
        }
        for qid in folds[f"fold_{outer}"]:
            if labels.get(qid):
                stacked[qid] = fuse(qid, systems, recipe)
    total = fold_values.sum(axis=1)
    diagnostic_best = max(range(len(candidates)), key=lambda index: (key(total[index]), -index))
    # Choice oracle is diagnostic only and never used to select the output.
    oracle = {}
    for qid in qids:
        oracle[qid] = max((systems[name][qid] for name in systems),
                          key=lambda order: (len(set(order[:5]) & labels[qid]),
                                             -next((rank for rank, doc in enumerate(order[:5], 1) if doc in labels[qid]), 99)))
    stacked_values = np.zeros(6)
    oracle_values = np.zeros(6)
    system_values = {name: np.zeros(6) for name in systems}
    for qid in qids:
        stacked_values[:5] += contribution(stacked[qid], labels[qid]); stacked_values[5] += 1
        oracle_values[:5] += contribution(oracle[qid], labels[qid]); oracle_values[5] += 1
        for name in systems:
            system_values[name][:5] += contribution(systems[name][qid], labels[qid]); system_values[name][5] += 1
    report = {
        "status": "COMPLETE_CROSSFITTED_RANKER_STACK",
        "scope_warning": "Level-2 recipes selected on the other four OOF folds; historical architecture development has seen all folds.",
        "systems": {name: metric(values) for name, values in system_values.items()},
        "candidate_recipes": len(candidates), "selected": selected,
        "crossfitted_stack": metric(stacked_values),
        "choice_oracle_diagnostic": metric(oracle_values),
        "global_posthoc_best_diagnostic": {"recipe": candidates[diagnostic_best], "metrics": metric(total[diagnostic_best])},
    }
    write(OUT / "PREDICTIONS.json", stacked)
    write(OUT / "RANKER_STACK_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
