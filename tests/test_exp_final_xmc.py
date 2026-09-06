import numpy as np
import torch

from exp_final.xmc import (
    TrainablePrototypeXMC,
    decoupled_top5_loss,
    initial_label_prototypes,
)


def test_initial_prototypes_are_normalized_means():
    vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    value = initial_label_prototypes(vectors, [[0], [1], [0, 1]], 2)
    assert np.allclose(np.linalg.norm(value, axis=1), 1)
    assert value[0, 0] > value[0, 1]
    assert value[1, 1] > value[1, 0]


def test_decoupled_loss_excludes_other_gold_from_denominator():
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]], requires_grad=True)
    loss, _ = decoupled_top5_loss(logits, [[0, 1]], boundary_weight=0.0)
    expected0 = torch.logsumexp(torch.tensor([4.0, 1.0, 0.0]), 0) - 4.0
    expected1 = torch.logsumexp(torch.tensor([3.0, 1.0, 0.0]), 0) - 3.0
    assert torch.allclose(loss, (expected0 + expected1) / 2)


def test_boundary_pushes_each_positive_above_capacity_boundary():
    poor = torch.tensor([[0.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    good = torch.tensor([[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    poor_loss, poor_parts = decoupled_top5_loss(poor, [[0, 1]], boundary_weight=1.0)
    good_loss, good_parts = decoupled_top5_loss(good, [[0, 1]], boundary_weight=1.0)
    assert good_parts["boundary"] < poor_parts["boundary"]
    assert good_loss < poor_loss


def test_identity_query_path_and_gradients():
    prototypes = np.eye(3, dtype=np.float32)
    model = TrainablePrototypeXMC(prototypes, query_rank=0)
    logits, diagnostic = model(torch.tensor([[1.0, 0.0, 0.0]]))
    assert logits.argmax(1).item() == 0
    assert diagnostic["query_drift"].item() == 0
    loss, _ = decoupled_top5_loss(logits, [[0]])
    loss.backward()
    assert model.prototype_residual.grad is not None
    assert torch.isfinite(model.prototype_residual.grad).all()
