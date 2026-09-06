from __future__ import annotations

import numpy as np
import torch

from exp_final.learning import CaseSupportSampler, case_multi_loss


class FakeData:
    def __init__(self):
        self.gold = {
            "q1": {"a"},
            "q2": {"a", "x"},
            "q3": {"b"},
            "q4": {"c"},
            "q5": {"d"},
            "q6": {"e"},
        }
        self.vectors = {
            "q1": [1.0, 0.0],
            "q2": [0.9, 0.1],
            "q3": [0.8, 0.2],
            "q4": [0.0, 1.0],
            "q5": [-0.5, 0.5],
            "q6": [-1.0, 0.0],
        }

    def query_vector(self, qid, source):
        assert source == "lal"
        value = np.asarray(self.vectors[qid], dtype=np.float32)
        return value / np.linalg.norm(value)


def test_case_sampler_uses_shared_labels_only_for_positives():
    data = FakeData()
    sampler = CaseSupportSampler(data, list(data.gold), hard_pool=4)
    positives, negatives = sampler.sample("q1", 0, positive_count=1, negative_count=4)
    assert positives == ["q2"]
    assert len(negatives) == 4
    assert len(set(negatives)) == 4
    assert all(not (data.gold["q1"] & data.gold[q]) for q in negatives)


def test_case_sampler_skips_queries_without_positive_peer():
    data = FakeData()
    sampler = CaseSupportSampler(data, list(data.gold), hard_pool=4)
    assert sampler.sample("q3", 0) == ([], [])


def test_case_loss_is_finite_and_backpropagates_to_all_roles():
    anchor = torch.tensor([1.0, 0.0], requires_grad=True)
    positives = torch.tensor([[0.9, 0.1]], requires_grad=True)
    negatives = torch.tensor([[0.1, 0.9], [-0.5, 0.5]], requires_grad=True)
    loss = case_multi_loss(anchor, positives, negatives)
    loss.backward()
    assert torch.isfinite(loss)
    assert anchor.grad is not None and torch.count_nonzero(anchor.grad)
    assert positives.grad is not None and torch.count_nonzero(positives.grad)
    assert negatives.grad is not None and torch.count_nonzero(negatives.grad)

