"""Repair the duplicated semantic-memory vote features and rerun strict OOF.

The incumbent memory LTR accidentally writes the frequency-normalized vote to
both ``memory_soft_vote`` and ``memory_frequency_vote``.  This probe preserves
the exact candidate, fold, model and feature contracts while separating:

* soft vote: similarity mass, normalized only by neighbour label cardinality;
* frequency vote: the same mass divided by fold-local document frequency.

All target-fold labels remain excluded by ``run_outer`` from the reused probe.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import exp_final_memory_ltr_probe as base


def repaired_memory_features(similarities, docs, support_qids, labels, by_doc,
                             frequency, self_qid=None, self_index=None):
    similarities = np.asarray(similarities, dtype=np.float32)
    if self_index is not None:
        similarities = similarities.copy()
        similarities[self_index] = -np.inf
    take = min(16, len(similarities))
    part = np.argpartition(-similarities, take - 1)[:take]
    neighbours = sorted(part.tolist(), key=lambda i: (-float(similarities[i]), support_qids[i]))
    soft_votes = defaultdict(float)
    frequency_votes = defaultdict(float)
    nearest = float(similarities[neighbours[0]])
    second_near = float(similarities[neighbours[1]]) if len(neighbours) > 1 else nearest
    neighbour_labels = []
    self_gold = labels.get(self_qid, set()) if self_qid else set()
    for index in neighbours:
        qid = support_qids[index]
        gold = labels.get(qid, ())
        neighbour_labels.append(set(gold))
        mass = math.exp(20 * (float(similarities[index]) - 1)) / max(1, len(gold))
        for doc_id in gold:
            soft_votes[doc_id] += mass
            adjusted_frequency = frequency[doc_id] - (1 if self_qid and doc_id in self_gold else 0)
            frequency_votes[doc_id] += mass / max(1, adjusted_frequency)
    vote_order = sorted(frequency_votes, key=lambda d: (-frequency_votes[d], d))
    vote_rank = {doc_id: rank for rank, doc_id in enumerate(vote_order, 1)}
    positive = np.asarray([value for value in frequency_votes.values() if value > 0], dtype=np.float64)
    entropy = 0.0
    if len(positive) > 1:
        probabilities = positive / positive.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(len(probabilities)))
    ordered = sorted(frequency_votes.values(), reverse=True)
    winner_margin = float(ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) if ordered else 0.0
    counts = Counter(doc_id for gold in neighbour_labels for doc_id in gold)
    agreement = max(counts.values(), default=0) / max(1, len(neighbours))
    output = np.empty((len(docs), len(base.MEMORY_NAMES)), dtype=np.float32)
    for row, doc_id in enumerate(docs):
        indices = [i for i in by_doc.get(doc_id, ()) if support_qids[i] != self_qid]
        local = sorted((float(similarities[i]) for i in indices if np.isfinite(similarities[i])), reverse=True)
        count = max(0, frequency[doc_id] - (1 if doc_id in self_gold else 0))
        maximum = local[0] if local else -1.0
        second = local[1] if len(local) > 1 else -1.0
        top2 = float(np.mean(local[:2])) if local else -1.0
        output[row] = [
            float(bool(local)), count, math.log1p(count), maximum, second, top2,
            float(soft_votes.get(doc_id, 0.0)), float(frequency_votes.get(doc_id, 0.0)),
            1 / vote_rank[doc_id] if doc_id in vote_rank else 0.0,
            nearest, nearest - second_near, entropy, winner_margin, agreement,
        ]
    return output


def main():
    base.OUT = base.ROOT / "results/exp_final_retrieval/memory_vote_fix_probe"
    base.memory_features = repaired_memory_features
    reports = [base.run_outer(fold, "lal") for fold in range(5)]
    summary = {
        "status": "COMPLETE_MEMORY_VOTE_FIX_5FOLD",
        "hypothesis": "Distinct cardinality and frequency-normalized votes improve the incumbent memory LTR.",
        "baseline_incumbent_recall_at_5": 0.9444690792924235,
        "mean_delta_vs_exp112_anchor": float(np.mean([row["delta"] for row in reports])),
        "reports": reports,
    }
    base.write(base.OUT / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
