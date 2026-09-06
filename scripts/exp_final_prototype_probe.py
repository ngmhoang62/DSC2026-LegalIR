"""Efficient fold-isolated semantic legal-case memory probe.

Unlike the cancelled Colab run, this implementation computes only top-64 query
neighbours from small 7k x 1024 matrices, then evaluates a bounded set of
predeclared voting/fusion hypotheses against immutable EXP-112 OOF rankings.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "exp112_task_adaptive_retrieval"
OUT = ROOT / "results" / "exp_final_retrieval" / "prototype_probe"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def top_neighbours(test, support, support_ids, n=64):
    similarities = np.asarray(test @ support.T, dtype=np.float32)
    part = np.argpartition(-similarities, min(n, similarities.shape[1]) - 1, axis=1)[:, :n]
    result = []
    for row, indices in zip(similarities, part):
        ordered = sorted(indices.tolist(), key=lambda i: (-float(row[i]), str(support_ids[i])))
        result.append([(str(support_ids[i]), float(row[i])) for i in ordered])
    return result


def prototype_rank(neighbours, labels, frequency, n, temperature, gamma, cardinality=True):
    votes = defaultdict(float)
    for qid, similarity in neighbours[:n]:
        docs = labels.get(qid, ())
        affinity = math.exp(temperature * (similarity - 1.0))
        if cardinality:
            affinity /= max(1, len(docs))
        for doc in docs:
            votes[doc] += affinity / max(1.0, frequency[doc] ** gamma)
    return sorted(votes, key=lambda d: (-votes[d], d)), votes


def fuse(base, prototype, weight, k=32):
    if weight == 0:
        return list(base)
    br = {d: i for i, d in enumerate(base, 1)}
    pr = {d: i for i, d in enumerate(prototype, 1)}
    docs = set(br) | set(pr)
    score = {d: (1 - weight) / (k + br[d]) if d in br else 0.0 for d in docs}
    for d in pr:
        score[d] += weight / (k + pr[d])
    return sorted(docs, key=lambda d: (-score[d], d))


def metrics(rankings, labels, qids):
    all_values, multi, precision = [], [], []
    wins = losses = 0
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        hit = len(set(rankings[qid][:5]) & gold)
        value = hit / len(gold)
        all_values.append(value)
        precision.append(hit / 5)
        if len(gold) > 1:
            multi.append(value)
    return {"recall@5": float(np.mean(all_values)), "precision@5": float(np.mean(precision)),
            "multi_gold_recall@5": float(np.mean(multi)), "queries": len(all_values)}


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache" / "cv_folds.json")
    base = {}
    for fold in range(5):
        rows = read(OLD / "outer" / f"fold_{fold}" / "PREDICTIONS.json")
        base.update({qid: list(map(str, row["order"])) for qid, row in rows.items()})
    evaluable = [q for q in base if labels.get(q)]
    baseline = metrics(base, labels, evaluable)

    e5_dir = ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"
    e5_ids = list(map(str, read(e5_dir / "train_query_ids.json")))
    e5 = normalize(np.load(e5_dir / "train_queries.f32.npy", mmap_mode="r"))
    with np.load(ROOT / "cache" / "exp109b_encoder_complementarity" / "embeddings" / "vnlegal_lal" / "queries.npz", allow_pickle=False) as z:
        lal_ids = list(map(str, z["query_ids"].tolist()))
        lal = normalize(z["vectors"])
    if set(e5_ids) != set(lal_ids) or set(e5_ids) != set(base):
        raise ValueError("Query vector/prediction ID mismatch")
    lal_map = {q: i for i, q in enumerate(lal_ids)}
    lal = lal[[lal_map[q] for q in e5_ids]]
    combo = normalize(e5 + lal)
    qrow = {q: i for i, q in enumerate(e5_ids)}
    representations = {"e5": e5, "lal": lal, "mean_e5_lal": combo}

    neighbour_cache = {name: {} for name in representations}
    reachability = {name: {} for name in representations}
    for fold in range(5):
        test_ids = [q for q in folds[f"fold_{fold}"] if labels.get(q)]
        support_ids = [q for f in range(5) if f != fold for q in folds[f"fold_{f}"] if labels.get(q)]
        ti = [qrow[q] for q in test_ids]
        si = [qrow[q] for q in support_ids]
        for name, matrix in representations.items():
            rows = top_neighbours(matrix[ti], matrix[si], support_ids, 64)
            neighbour_cache[name].update(dict(zip(test_ids, rows)))
            oracle = []
            recovered = 0
            for qid, neighbours in zip(test_ids, rows):
                reachable = set().union(*(labels[nqid] for nqid, _ in neighbours))
                current = set(base[qid][:5]) & labels[qid]
                new = (labels[qid] - current) & reachable
                recovered += len(new)
                oracle.append(min(5, len(current) + len(new)) / len(labels[qid]))
            reachability[name][f"fold_{fold}"] = {"oracle_recall@5": float(np.mean(oracle)),
                                                   "reachable_missing_gold_assignments": recovered,
                                                   "nearest_similarity_mean": float(np.mean([r[0][1] for r in rows])),
                                                   "nearest_similarity_p90": float(np.quantile([r[0][1] for r in rows], .9))}

    policies = [(1, 10.0, 0.5), (4, 10.0, 0.5), (16, 10.0, 0.5), (64, 10.0, 0.5),
                (16, 5.0, 0.0), (16, 20.0, 1.0), (64, 5.0, 1.0)]
    weights = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    trials = []
    for representation, cache in neighbour_cache.items():
        for policy in policies:
            proto = {}
            for fold in range(5):
                support_ids = [q for f in range(5) if f != fold for q in folds[f"fold_{f}"] if labels.get(q)]
                frequency = Counter(d for q in support_ids for d in labels[q])
                for qid in folds[f"fold_{fold}"]:
                    if labels.get(qid):
                        proto[qid] = prototype_rank(cache[qid], labels, frequency, *policy)[0]
            standalone = metrics(proto, labels, evaluable)
            for weight in weights:
                ranked = {q: fuse(base[q], proto[q], weight) for q in evaluable}
                score = metrics(ranked, labels, evaluable)
                per_fold = {f"fold_{f}": metrics(ranked, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)]) for f in range(5)}
                trials.append({"representation": representation, "policy": {"neighbours": policy[0], "temperature": policy[1], "frequency_power": policy[2]},
                               "weight": weight, "prototype_standalone": standalone, "metrics": score,
                               "delta": score["recall@5"] - baseline["recall@5"],
                               "multi_delta": score["multi_gold_recall@5"] - baseline["multi_gold_recall@5"],
                               "nonnegative_folds": sum(per_fold[f"fold_{f}"]["recall@5"] >= metrics(base, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall@5"] for f in range(5)),
                               "per_fold": per_fold})
    trials.sort(key=lambda x: (x["metrics"]["recall@5"], x["metrics"]["precision@5"], x["multi_delta"]), reverse=True)
    report = {"status": "COMPLETE_DEVELOPMENT_PROTOTYPE_PROBE",
              "warning": "Fold-isolated label memory, but bounded recipes were inspected on previously exposed development OOF.",
              "baseline": baseline, "reachability": reachability, "trials": len(trials), "top_trials": trials[:50]}
    write(OUT / "PROTOTYPE_PROBE.json", report)
    print(json.dumps({"status": report["status"], "baseline": baseline, "reachability": reachability, "trials": len(trials), "top5": trials[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
