"""Small residual set encoder for jointly scoring a legal candidate slate in the Gemini namespace."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualSetRanker(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        query_dim: int,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        depth: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(candidate_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.query = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(depth, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden,
            heads,
            hidden * 2,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.output = nn.Linear(hidden, 1)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 3 or query.ndim != 2 or len(candidates) != len(query):
            raise ValueError("Invalid set-ranker batch")
        if candidates.shape[1] > len(self.position):
            raise ValueError("Slate exceeds positional capacity")
        hidden = (
            self.candidate(candidates)
            + self.query(query)[:, None, :]
            + self.position[: candidates.shape[1]]
        )
        return self.output(self.encoder(hidden)).squeeze(-1)


def multi_positive_set_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    multi_weight: float = 2.0,
) -> torch.Tensor:
    """Independent-positive contrastive loss with query-balanced BCE.
    
    Multiple positives do not compete with each other in the denominator;
    each positive is contrasted against the log-sum-exp of negative scores.
    """
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("Set loss shape mismatch")
    losses = []
    for score, target in zip(logits, targets.bool()):
        positives = score[target]
        negatives = score[~target]
        if not len(positives):
            continue
        if not len(negatives):
            contrastive = F.softplus(-positives).mean()
            balanced = contrastive
        else:
            contrastive = (
                torch.logaddexp(positives, torch.logsumexp(negatives, dim=0)) - positives
            ).mean()
            balanced = 0.5 * (F.softplus(-positives).mean() + F.softplus(negatives).mean())
        weight = multi_weight if len(positives) > 1 else 1.0
        losses.append(weight * (contrastive + 0.1 * balanced))
    if not losses:
        raise ValueError("Batch contains no positive slate")
    denom = sum(
        multi_weight if int(target.sum()) > 1 else 1.0
        for target in targets
        if target.any()
    )
    return torch.stack(losses).sum() / denom
