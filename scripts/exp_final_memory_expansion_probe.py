"""Bounded OOF probe: use semantic case memory as a candidate generator.

The incumbent case-memory LambdaMART can only rank documents already present in
the frozen content union.  This probe asks the orthogonal question: can labels
attached to fold-isolated nearest training queries safely introduce missing
parents into the final top five?  It is a development diagnostic, not an
unbiased final selection.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT = ROOT / "results/exp_final_retrieval/memory_expansion_probe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def metrics(rankings, labels, qids):
    recall, precision, multi, mrr = [], [], [], []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = rankings[qid][:5]
        hits = len(set(top) & gold)
        value = hits / len(gold)
        recall.append(value)
        precision.append(hits / 5)
        if len(gold) > 1:
            multi.append(value)
        rank = next((i for i, doc in enumerate(top, 1) if doc in gold), None)
        mrr.append(0.0 if rank is None else 1.0 / rank)
    return {
        "recall@5": float(np.mean(recall)),
        "precision@5": float(np.mean(precision)),
        "multi_gold_recall@5": float(np.mean(multi)),
        "mrr@5": float(np.mean(mrr)),
        "queries": len(recall),
    }


def prototype(similarities, support_qids, labels, frequency, neighbours, temperature, gamma):
    similarities = np.asarray(similarities, dtype=np.float32)
    take = min(neighbours, len(similarities))
    part = np.argpartition(-similarities, take - 1)[:take]
    order = sorted(part.tolist(), key=lambda i: (-float(similarities[i]), support_qids[i]))
    votes = defaultdict(float)
    for i in order:
        gold = labels.get(support_qids[i], ())
        affinity = math.exp(temperature * (float(similarities[i]) - 1.0)) / max(1, len(gold))
        for doc in gold:
            votes[doc] += affinity / max(1.0, frequency[doc] ** gamma)
    ranking = sorted(votes, key=lambda doc: (-votes[doc], doc))
    return ranking, votes, float(similarities[order[0]])


def rrf(base, memory, weight, k=32):
    if weight == 0:
        return list(base)
    br = {doc: i for i, doc in enumerate(base, 1)}
    mr = {doc: i for i, doc in enumerate(memory, 1)}
    docs = set(br) | set(mr)
    scores = {
        doc: ((1.0 - weight) / (k + br[doc]) if doc in br else 0.0)
        + (weight / (k + mr[doc]) if doc in mr else 0.0)
        for doc in docs
    }
    return sorted(docs, key=lambda doc: (-scores[doc], doc))


def replace_tail(base, memory, count):
    additions = [doc for doc in memory if doc not in set(base[:5])][:count]
    return list(base[: max(1, 5 - count)]) + additions + list(base[5:])


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    base = {}
    for fold in range(5):
        path = MEMORY / f"fold_{fold}" / "PREDICTIONS.json"
        base.update({str(q): list(map(str, docs)) for q, docs in read(path).items()})
    evaluable = [q for q in base if labels.get(q)]

    with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz", allow_pickle=False) as z:
        ids = list(map(str, z["query_ids"].tolist()))
        vectors = normalize(z["vectors"])
    qrow = {q: i for i, q in enumerate(ids)}

    # Keep the recipe family intentionally small.  These are the strongest
    # policies from the earlier bounded prototype screen plus two nearby
    # cardinality/frequency alternatives.
    policies = (
        (16, 20.0, 1.0),
        (16, 10.0, 0.5),
        (16, 5.0, 0.0),
        (64, 5.0, 1.0),
    )
    memories = {policy: {} for policy in policies}
    diagnostics = {policy: {} for policy in policies}
    for fold in range(5):
        test = [q for q in folds[f"fold_{fold}"] if labels.get(q)]
        support = [q for f in range(5) if f != fold for q in folds[f"fold_{f}"] if labels.get(q)]
        frequency = Counter(doc for q in support for doc in labels[q])
        similarities = np.asarray(vectors[[qrow[q] for q in test]] @ vectors[[qrow[q] for q in support]].T, dtype=np.float32)
        for policy in policies:
            for row, q in zip(similarities, test):
                ranking, votes, nearest = prototype(row, support, labels, frequency, *policy)
                memories[policy][q] = ranking
                diagnostics[policy][q] = {"nearest": nearest, "winner_vote": max(votes.values(), default=0.0)}

    baseline = metrics(base, labels, evaluable)
    trials = []
    for policy in policies:
        memory = memories[policy]
        for weight in (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15):
            ranking = {q: rrf(base[q], memory[q], weight) for q in evaluable}
            score = metrics(ranking, labels, evaluable)
            per_fold = {f"fold_{f}": metrics(ranking, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)]) for f in range(5)}
            trials.append({
                "method": "rrf",
                "policy": {"neighbours": policy[0], "temperature": policy[1], "frequency_power": policy[2]},
                "weight": weight,
                "metrics": score,
                "delta": score["recall@5"] - baseline["recall@5"],
                "multi_delta": score["multi_gold_recall@5"] - baseline["multi_gold_recall@5"],
                "nonnegative_folds": sum(per_fold[f"fold_{f}"]["recall@5"] >= metrics(base, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall@5"] for f in range(5)),
                "per_fold": per_fold,
            })
        for count in (1, 2):
            ranking = {q: replace_tail(base[q], memory[q], count) for q in evaluable}
            score = metrics(ranking, labels, evaluable)
            per_fold = {f"fold_{f}": metrics(ranking, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)]) for f in range(5)}
            trials.append({
                "method": "replace_tail",
                "policy": {"neighbours": policy[0], "temperature": policy[1], "frequency_power": policy[2]},
                "count": count,
                "metrics": score,
                "delta": score["recall@5"] - baseline["recall@5"],
                "multi_delta": score["multi_gold_recall@5"] - baseline["multi_gold_recall@5"],
                "nonnegative_folds": sum(per_fold[f"fold_{f}"]["recall@5"] >= metrics(base, labels, [q for q in folds[f"fold_{f}"] if labels.get(q)])["recall@5"] for f in range(5)),
                "per_fold": per_fold,
            })

    trials.sort(key=lambda x: (x["metrics"]["recall@5"], x["metrics"]["precision@5"], x["metrics"]["mrr@5"]), reverse=True)
    report = {
        "status": "COMPLETE_MEMORY_CANDIDATE_EXPANSION_PROBE",
        "scope_warning": "Fold-isolated memory, but bounded policies were compared on previously exposed development OOF.",
        "baseline": baseline,
        "trials": len(trials),
        "top_trials": trials[:30],
    }
    write(OUT / "MEMORY_EXPANSION_PROBE.json", report)
    print(json.dumps({"status": report["status"], "baseline": baseline, "trials": len(trials), "top5": trials[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
