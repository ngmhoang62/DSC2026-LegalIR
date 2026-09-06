import numpy as np
import torch

from exp_final.symmetric_metric import (
    ProjectedParentBank,
    ResidualMetricTower,
    deterministic_negatives,
    symmetric_multi_loss,
    top5_boundary_loss,
)


def test_identity_initialization_for_both_towers():
    model = ResidualMetricTower(dimension=4, rank=2)
    value = torch.tensor([[1.0, 2.0, 0.0, -1.0]])
    expected = torch.nn.functional.normalize(value, dim=-1)
    assert torch.allclose(model.query(value), expected)
    assert torch.allclose(model.document(value), expected)


def test_projected_parent_bank_top_two_and_singleton():
    parent = np.array([0, 0, 0, 1], dtype=np.int64)
    model = ResidualMetricTower(dimension=2, rank=1)
    raw = np.array([[1, 0], [.8, .2], [0, 1], [1, 1]], dtype=np.float32)
    bank = ProjectedParentBank(raw.shape, parent, device="cpu")
    bank.refresh(raw, model, block_size=2)
    score, indices = bank.mine(torch.tensor([1.0, 0.0]))
    expected0 = (1.0 + float(.8 / np.sqrt(.68))) / 2
    expected1 = 1 / np.sqrt(2)
    assert torch.allclose(score[0], torch.tensor([expected0, expected1], dtype=torch.float32), atol=1e-6)
    assert indices[0, 1, 0] == indices[0, 1, 1] == 3


def test_metric_loss_and_boundary_prefer_positive_margin():
    bad_p, good_p = torch.tensor([0.0]), torch.tensor([1.0])
    negative = torch.tensor([.5, .4, .3, .2, .1])
    assert symmetric_multi_loss(good_p, negative) < symmetric_multi_loss(bad_p, negative)
    assert top5_boundary_loss(good_p, negative, 1) < top5_boundary_loss(bad_p, negative, 1)


def test_negative_sampler_is_unique_deterministic_and_gold_free():
    current = [str(i) for i in range(100)]
    sources = {"lal": list(reversed(current)), "bm25": current, "trigram": current[::2]}
    universe = [str(i) for i in range(200)]
    first = deterministic_negatives(current, sources, {"0", "1"}, universe, "42", 0)
    second = deterministic_negatives(current, sources, {"0", "1"}, universe, "42", 0)
    assert first == second
    assert len(first) == len(set(first)) == 64
    assert not ({"0", "1"} & set(first))
