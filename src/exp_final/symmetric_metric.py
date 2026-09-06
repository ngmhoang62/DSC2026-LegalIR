"""Symmetric low-rank metric adaptation over frozen E5 embeddings."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMetricTower(nn.Module):
    """Untied query/document low-rank residual transforms with identity init."""

    def __init__(self, dimension: int = 1024, rank: int = 32):
        super().__init__()
        self.query_down = nn.Linear(dimension, rank, bias=False)
        self.query_up = nn.Linear(rank, dimension, bias=False)
        self.document_down = nn.Linear(dimension, rank, bias=False)
        self.document_up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.query_down.weight, std=0.02)
        nn.init.normal_(self.document_down.weight, std=0.02)
        nn.init.zeros_(self.query_up.weight)
        nn.init.zeros_(self.document_up.weight)

    @staticmethod
    def _transform(value, down, up):
        value = F.normalize(value.float(), dim=-1)
        return F.normalize(value + up(F.gelu(down(value))), dim=-1)

    def query(self, value):
        return self._transform(value, self.query_down, self.query_up)

    def document(self, value):
        return self._transform(value, self.document_down, self.document_up)


class ProjectedParentBank:
    """One projected full-corpus bank with deterministic parent top-two pooling."""

    def __init__(self, shape, parent_indices, *, device="cuda"):
        self.vectors = torch.empty(shape, dtype=torch.float32, device=device)
        self.parent = torch.as_tensor(parent_indices, dtype=torch.long, device=device)
        self.count = int(self.parent.max()) + 1
        self.counts = torch.bincount(self.parent, minlength=self.count)
        self.chunk_ids = torch.arange(len(self.parent), device=device)

    @torch.no_grad()
    def refresh(self, raw_vectors, model: ResidualMetricTower, block_size=8192):
        model.eval()
        for start in range(0, len(self.vectors), block_size):
            stop = min(len(self.vectors), start + block_size)
            raw = torch.as_tensor(
                np.array(raw_vectors[start:stop], dtype=np.float32, copy=True),
                device=self.vectors.device,
            )
            self.vectors[start:stop].copy_(model.document(raw))

    @torch.no_grad()
    def mine(self, query):
        query = query.detach().float()
        if query.ndim == 1:
            query = query[None]
        scores = query @ self.vectors.T
        batch = len(scores)
        ids = self.parent.expand(batch, -1)
        first = torch.full((batch, self.count), -torch.inf, device=scores.device)
        first.scatter_reduce_(1, ids, scores, reduce="amax", include_self=True)
        sentinel = len(self.parent)
        arg1 = torch.full((batch, self.count), sentinel, device=scores.device, dtype=torch.long)
        eligible = torch.where(scores == first.gather(1, ids), self.chunk_ids, sentinel)
        arg1.scatter_reduce_(1, ids, eligible, reduce="amin", include_self=True)
        rest = scores.masked_fill(self.chunk_ids[None] == arg1.gather(1, ids), -torch.inf)
        second = torch.full_like(first, -torch.inf)
        second.scatter_reduce_(1, ids, rest, reduce="amax", include_self=True)
        arg2 = torch.full_like(arg1, sentinel)
        eligible2 = torch.where(rest == second.gather(1, ids), self.chunk_ids, sentinel)
        arg2.scatter_reduce_(1, ids, eligible2, reduce="amin", include_self=True)
        singleton = self.counts == 1
        arg2[:, singleton] = arg1[:, singleton]
        parent_scores = torch.where(singleton, first, 0.5 * (first + second))
        return parent_scores, torch.stack((arg1, arg2), dim=-1)


def symmetric_multi_loss(positive, negative, temperature=0.05):
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("Symmetric metric loss needs positives and negatives")
    p, n = positive / temperature, negative / temperature
    return (torch.logaddexp(p, torch.logsumexp(n, dim=0)) - p).mean()


def top5_boundary_loss(positive, negative, gold_count, temperature=0.05):
    """Push every positive above the negative capacity boundary."""
    boundary_rank = max(1, 6 - int(gold_count))
    boundary_rank = min(boundary_rank, int(negative.numel()))
    boundary = torch.topk(negative, k=boundary_rank).values[-1]
    return F.softplus((boundary - positive) / temperature).mean()


def deterministic_negatives(current, sources, gold, universe, qid, epoch, count=64):
    rng = random.Random(int(qid) * 1009 + epoch * 9176 + 112)
    blocked, selected = set(gold), []

    def add(values, limit, shuffle=False):
        values = list(values)
        if shuffle:
            rng.shuffle(values)
        before = len(selected)
        for doc in values:
            if doc not in blocked:
                blocked.add(doc)
                selected.append(doc)
                if len(selected) - before >= limit:
                    break

    add(current[:32], 16)
    add(current[16:100], 16, shuffle=True)
    disagreement = []
    for source in ("lal", "bm25", "trigram"):
        disagreement.extend(sources.get(source, [])[:100])
    add(disagreement, 24, shuffle=True)
    add(universe, 8, shuffle=True)
    if len(selected) < count:
        add(current, count - len(selected))
    if len(selected) < count:
        add(universe, count - len(selected), shuffle=True)
    if len(selected) != count:
        raise ValueError("Could not construct 64 unique symmetric-metric negatives")
    return selected


@dataclass(frozen=True)
class SymmetricTrainConfig:
    rank: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 2
    drift_weight: float = 0.05
    boundary_weight: float = 0.25
    seed: int = 112
