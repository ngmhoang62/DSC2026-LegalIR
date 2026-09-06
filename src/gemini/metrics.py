"""Evaluation metrics and significance tests for Gemini research namespace.

Contract:
- Primary metric: Recall@5.
- Secondary metric: Precision@5 (tie-breaker).
- Diagnostic metrics: MRR@5, Multi-gold Recall@5.
- Fold-level and pooled bootstrap statistical significance testing.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import numpy as np


def compute_metrics(
    rankings: Mapping[str, Sequence[str]],
    labels: Mapping[str, set[str]],
    qids: Iterable[str] | None = None,
    k: int = 5,
) -> dict[str, float | int]:
    if qids is None:
        qids = sorted(labels.keys())
    
    recalls = []
    precisions = []
    multi_recalls = []
    mrrs = []
    
    for q in qids:
        gold = labels.get(q, set())
        if not gold:
            continue
        preds = rankings.get(q, [])[:k]
        hits = len(set(preds) & gold)
        r = hits / len(gold)
        p = hits / k
        recalls.append(r)
        precisions.append(p)
        if len(gold) > 1:
            multi_recalls.append(r)
        
        first_rank = next((i for i, d in enumerate(preds, 1) if d in gold), None)
        mrrs.append(0.0 if first_rank is None else 1.0 / first_rank)
        
    return {
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"multi_gold_recall_at_{k}": float(np.mean(multi_recalls)) if multi_recalls else 0.0,
        f"mrr_at_{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        "queries": len(recalls),
    }


def paired_bootstrap(
    base_rankings: Mapping[str, Sequence[str]],
    candidate_rankings: Mapping[str, Sequence[str]],
    labels: Mapping[str, set[str]],
    qids: Iterable[str] | None = None,
    k: int = 5,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    if qids is None:
        qids = sorted(labels.keys())
        
    eval_qids = [q for q in qids if labels.get(q)]
    base_r = []
    cand_r = []
    
    for q in eval_qids:
        gold = labels[q]
        b_hits = len(set(base_rankings.get(q, [])[:k]) & gold)
        c_hits = len(set(candidate_rankings.get(q, [])[:k]) & gold)
        base_r.append(b_hits / len(gold))
        cand_r.append(c_hits / len(gold))
        
    base_arr = np.asarray(base_r, dtype=np.float64)
    cand_arr = np.asarray(cand_r, dtype=np.float64)
    deltas = cand_arr - base_arr
    mean_delta = float(np.mean(deltas))
    
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
        
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    p_value = float(np.mean(np.asarray(boot_means) <= 0.0)) if mean_delta > 0 else float(np.mean(np.asarray(boot_means) >= 0.0))
    
    return {
        "mean_delta": mean_delta,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "sample_size": n,
        "wins": int(np.sum(deltas > 0)),
        "losses": int(np.sum(deltas < 0)),
        "ties": int(np.sum(deltas == 0)),
    }
