"""Trainable extreme multi-label primitives for EXP-final retrieval probes.

The module deliberately operates on precomputed query vectors.  It is a cheap
way to test whether the remaining retrieval error is better modelled as
query-to-label prediction than as another text-retrieval fusion tweak.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def l2_normalize(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def initial_label_prototypes(
    vectors: np.ndarray,
    label_rows: Sequence[Sequence[int]],
    label_count: int,
) -> np.ndarray:
    """Return normalized mean query prototypes for every observed label."""
    vectors = l2_normalize(vectors)
    sums = np.zeros((label_count, vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(label_count, dtype=np.int64)
    for row, labels in enumerate(label_rows):
        for label in labels:
            sums[label] += vectors[row]
            counts[label] += 1
    if np.any(counts == 0):
        raise ValueError("Every XMC label must have at least one training query")
    return l2_normalize(sums / counts[:, None])


class TrainablePrototypeXMC(nn.Module):
    """Cosine label classifier initialized from fold-scoped query centroids."""

    def __init__(self, prototypes: np.ndarray, *, query_rank: int = 0, scale: float = 20.0):
        super().__init__()
        base = torch.as_tensor(l2_normalize(prototypes), dtype=torch.float32)
        self.register_buffer("base_prototypes", base)
        self.prototype_residual = nn.Parameter(torch.zeros_like(base))
        self.label_bias = nn.Parameter(torch.zeros(base.shape[0], dtype=torch.float32))
        self.query_rank = int(query_rank)
        self.scale = float(scale)
        if self.query_rank:
            self.query_down = nn.Linear(base.shape[1], self.query_rank, bias=False)
            self.query_up = nn.Linear(self.query_rank, base.shape[1], bias=False)
            nn.init.normal_(self.query_down.weight, std=0.02)
            nn.init.zeros_(self.query_up.weight)

    def representations(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frozen = F.normalize(query.float(), dim=-1)
        if self.query_rank:
            delta = self.query_up(F.gelu(self.query_down(frozen)))
            adapted = F.normalize(frozen + 0.1 * delta, dim=-1)
        else:
            adapted = frozen
        labels = F.normalize(self.base_prototypes + self.prototype_residual, dim=-1)
        return adapted, labels

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adapted, labels = self.representations(query)
        logits = self.scale * (adapted @ labels.T) + self.label_bias
        diagnostics = {
            "prototype_drift": (1.0 - (labels * self.base_prototypes).sum(-1)).mean(),
            "query_drift": (1.0 - (adapted * F.normalize(query.float(), dim=-1)).sum(-1)).mean(),
            "bias_l2": self.label_bias.square().mean(),
        }
        return logits, diagnostics


def decoupled_top5_loss(
    logits: torch.Tensor,
    positive_rows: Sequence[Sequence[int]],
    *,
    boundary_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-positive full-softmax plus a direct top-five boundary surrogate.

    Other positives of the same query are excluded from each positive's
    denominator.  For a query with ``g`` positives, each positive is compared
    with the ``(6-g)``-th highest negative, the boundary required for all golds
    to fit in a five-document output.
    """
    if logits.ndim != 2 or len(positive_rows) != logits.shape[0]:
        raise ValueError("Logit batch and positive rows do not align")
    multi_losses: list[torch.Tensor] = []
    boundary_losses: list[torch.Tensor] = []
    for row, positives in zip(logits, positive_rows):
        unique = sorted(set(map(int, positives)))
        if not unique:
            raise ValueError("XMC training query has no positive labels")
        mask = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
        mask[unique] = False
        negatives = row[mask]
        if negatives.numel() == 0:
            raise ValueError("XMC query has no negative labels")
        negative_lse = torch.logsumexp(negatives, dim=0)
        positive = row[torch.as_tensor(unique, device=row.device)]
        multi_losses.append((torch.logaddexp(positive, negative_lse) - positive).mean())
        boundary_rank = max(1, 6 - len(unique))
        boundary_rank = min(boundary_rank, int(negatives.numel()))
        boundary = torch.topk(negatives, k=boundary_rank).values[-1]
        boundary_losses.append(F.softplus(boundary - positive).mean())
    multi = torch.stack(multi_losses).mean()
    boundary = torch.stack(boundary_losses).mean()
    return multi + float(boundary_weight) * boundary, {
        "multi": multi.detach(),
        "boundary": boundary.detach(),
    }


@dataclass(frozen=True)
class XMCTrainConfig:
    query_rank: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    boundary_weight: float = 0.25
    prototype_drift_weight: float = 0.05
    query_drift_weight: float = 0.05
    bias_weight: float = 1e-4
    batch_size: int = 128
    seed: int = 112


def train_epoch(
    model: TrainablePrototypeXMC,
    optimizer: torch.optim.Optimizer,
    vectors: np.ndarray,
    positives: Sequence[Sequence[int]],
    config: XMCTrainConfig,
    epoch: int,
    *,
    device: str,
) -> dict[str, float]:
    generator = np.random.default_rng(config.seed + int(epoch))
    order = generator.permutation(len(vectors))
    totals = {"loss": 0.0, "multi": 0.0, "boundary": 0.0, "batches": 0}
    model.train()
    for start in range(0, len(order), config.batch_size):
        rows = order[start:start + config.batch_size]
        query = torch.as_tensor(np.asarray(vectors[rows]), device=device, dtype=torch.float32)
        logits, diagnostic = model(query)
        classification, parts = decoupled_top5_loss(
            logits, [positives[int(i)] for i in rows], boundary_weight=config.boundary_weight,
        )
        loss = (
            classification
            + config.prototype_drift_weight * diagnostic["prototype_drift"]
            + config.query_drift_weight * diagnostic["query_drift"]
            + config.bias_weight * diagnostic["bias_l2"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["multi"] += float(parts["multi"])
        totals["boundary"] += float(parts["boundary"])
        totals["batches"] += 1
    count = max(1, totals.pop("batches"))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def score_topk(
    model: TrainablePrototypeXMC,
    vectors: np.ndarray,
    *,
    topk: int = 200,
    batch_size: int = 256,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores, all_indices = [], []
    for start in range(0, len(vectors), batch_size):
        query = torch.as_tensor(np.asarray(vectors[start:start + batch_size]), device=device, dtype=torch.float32)
        logits, _ = model(query)
        values, indices = torch.topk(logits, k=min(topk, logits.shape[1]), dim=1, sorted=True)
        all_scores.append(values.cpu().numpy())
        all_indices.append(indices.cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_indices)
