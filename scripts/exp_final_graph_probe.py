"""Cross-fitted co-relevance graph probe over immutable EXP-112 OOF rankings.

The graph for an outer fold is built from the other four folds only. This is a
development diagnostic: its parameter grid is inspected on already-exposed OOF
folds and therefore cannot be described as a pristine new held-out result.
"""
from __future__ import annotations

import json
import math
import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "exp112_task_adaptive_retrieval"
OUT = ROOT / "results" / "exp_final_retrieval" / "graph_probe"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def graph(labels: dict[str, set[str]], qids: list[str]):
    frequency = Counter()
    edges = Counter()
    for qid in qids:
        docs = sorted(labels.get(qid, ()))
        frequency.update(docs)
        for i, left in enumerate(docs):
            for right in docs[i + 1 :]:
                edges[left, right] += 1
                edges[right, left] += 1
    adjacency = defaultdict(list)
    for (left, right), count in edges.items():
        adjacency[left].append((right, count))
    return frequency, adjacency, edges


def strength(left, right, count, frequency, mode):
    fl, fr = frequency[left], frequency[right]
    if mode == "conditional":
        return count / (fl + 1.0)
    if mode == "cosine":
        return count / math.sqrt((fl + 0.5) * (fr + 0.5))
    if mode == "lift":
        return count / ((fl + 0.5) * (fr + 0.5)) ** 0.35
    raise ValueError(mode)


def rerank(base, frequency, adjacency, seed_depth, seed_power, mode, min_edge, alpha, base_k):
    propagated = defaultdict(float)
    for rank, seed in enumerate(base[:seed_depth], 1):
        for target, count in adjacency.get(seed, ()):
            if count >= min_edge:
                propagated[target] += rank ** (-seed_power) * strength(seed, target, count, frequency, mode)
    maximum = max(propagated.values(), default=0.0)
    base_pos = {doc: rank for rank, doc in enumerate(base, 1)}
    docs = set(base_pos) | set(propagated)
    scores = {}
    for doc in docs:
        b = 1.0 / (base_k + base_pos.get(doc, 100000))
        g = propagated.get(doc, 0.0) / maximum if maximum else 0.0
        scores[doc] = b + alpha * g / (base_k + 1.0)
    return sorted(docs, key=lambda d: (-scores[d], d))


def macro(rankings, labels, qids):
    values = []
    multi = []
    precision = []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        hit = len(set(rankings[qid][:5]) & gold)
        value = hit / len(gold)
        values.append(value)
        precision.append(hit / 5)
        if len(gold) > 1:
            multi.append(value)
    return {"recall@5": float(np.mean(values)), "precision@5": float(np.mean(precision)),
            "multi_gold_recall@5": float(np.mean(multi)), "queries": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=("exp112", "memory_ltr"), default="exp112")
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache" / "cv_folds.json")
    base = {}
    for fold in range(5):
        if args.base == "exp112":
            rows = read(OLD / "outer" / f"fold_{fold}" / "PREDICTIONS.json")
            base.update({qid: list(map(str, row["order"])) for qid, row in rows.items()})
        else:
            rows = read(ROOT / "results/exp_final_retrieval/memory_ltr_probe" / f"fold_{fold}" / "PREDICTIONS.json")
            base.update({qid: list(map(str, row)) for qid, row in rows.items()})
    evaluable = [q for q in base if labels.get(q)]
    baseline = macro(base, labels, evaluable)

    fold_graphs = {}
    reach = {}
    for fold in range(5):
        held = [q for q in folds[f"fold_{fold}"] if labels.get(q)]
        train = [q for other in range(5) if other != fold for q in folds[f"fold_{other}"] if labels.get(q)]
        frequency, adjacency, edges = graph(labels, train)
        fold_graphs[fold] = (frequency, adjacency)
        multi = [q for q in held if len(labels[q]) > 1]
        pair_seen = sum(any(edges[a, b] for i, a in enumerate(sorted(labels[q])) for b in sorted(labels[q])[i + 1 :]) for q in multi)
        row = {"multi_queries": len(multi), "gold_pair_seen_in_outer_train": pair_seen}
        for depth in (1, 3, 5, 10):
            values = []
            recovered = 0
            for q in held:
                gold = labels[q]
                current = set(base[q][:5]) & gold
                targets = {t for seed in base[q][:depth] for t, _ in adjacency.get(seed, ())}
                newly = (gold - current) & targets
                recovered += len(newly)
                values.append(min(5, len(current) + len(newly)) / len(gold))
            row[f"reachable_oracle_seed{depth}"] = float(np.mean(values))
            row[f"reachable_missing_gold_assignments_seed{depth}"] = recovered
        reach[f"fold_{fold}"] = row

    # Bounded hypothesis screen around the teammate repository's fixed graph
    # recipe. A broad six-dimensional search is both wasteful and especially
    # prone to development-OOF overfitting.
    configs = list(product((1, 3, 5), (0.5,), ("conditional", "cosine"), (1, 2, 3),
                           (0.10, 0.20, 0.40), (0, 20)))
    trials = []
    for ci, config in enumerate(configs):
        ranked = {}
        per_fold = {}
        for fold in range(5):
            frequency, adjacency = fold_graphs[fold]
            ids = [q for q in folds[f"fold_{fold}"] if labels.get(q)]
            for q in ids:
                ranked[q] = rerank(base[q], frequency, adjacency, *config)
            per_fold[f"fold_{fold}"] = macro(ranked, labels, ids)
        score = macro(ranked, labels, evaluable)
        trials.append({"params": dict(zip(("seed_depth", "seed_power", "mode", "min_edge", "alpha", "base_k"), config)),
                       "metrics": score, "delta": score["recall@5"] - baseline["recall@5"],
                       "multi_delta": score["multi_gold_recall@5"] - baseline["multi_gold_recall@5"],
                       "nonnegative_folds": sum(per_fold[f"fold_{f}"]["recall@5"] >= macro(base, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall@5"] for f in range(5)),
                       "per_fold": per_fold})
    trials.sort(key=lambda x: (x["metrics"]["recall@5"], x["metrics"]["precision@5"], x["multi_delta"]), reverse=True)
    report = {"status": "COMPLETE_DEVELOPMENT_GRAPH_PROBE", "base": args.base,
              "warning": "Grid inspected on previously exposed OOF folds; diagnostic evidence only.",
              "baseline": baseline, "reachability": reach, "configs": len(configs), "top_trials": trials[:50]}
    write(OUT / f"GRAPH_PROBE_{args.base}.json", report)
    print(json.dumps({"status": report["status"], "baseline": baseline, "reachability": reach, "configs": len(configs), "top5": trials[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
