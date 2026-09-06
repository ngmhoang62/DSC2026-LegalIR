from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(__file__.replace("\\tests\\test_exp109a_softtop5_retrieval.py", "\\src")))

import exp109a_softtop5_retrieval as exp  # noqa: E402


def test_residual_projection_is_identity_at_initialization() -> None:
    exp.set_seed()
    model = exp.ResidualProjection(dimension=8, rank=3)
    query = torch.randn(5, 8)
    expected = torch.nn.functional.normalize(query, p=2, dim=-1)
    assert torch.allclose(model(query), expected, atol=1e-7, rtol=1e-7)
    assert model.down.weight.shape == (3, 8)
    assert model.up.weight.shape == (8, 3)


@pytest.mark.parametrize("length", [6, 8, 17, 64])
@pytest.mark.parametrize("alpha", [1.0, 2.0, 5.0, 10.0])
def test_soft_top5_has_finite_membership_and_sum(length: int, alpha: float) -> None:
    scores = torch.linspace(-3.0, 3.0, length, dtype=torch.float64)
    membership = exp.soft_top_k(scores, k=5, alpha=alpha)
    assert torch.isfinite(membership).all()
    assert float(membership.min()) >= 0.0
    assert float(membership.max()) <= 1.0
    assert float(membership.sum()) == pytest.approx(5.0, abs=2e-8)


def test_soft_top5_is_permutation_equivariant_and_standardized_shift_invariant() -> None:
    scores = torch.tensor([-2.0, -0.2, 0.1, 0.9, 1.4, 2.8, 3.2, -1.3], dtype=torch.float64)
    permutation = torch.tensor([3, 0, 7, 4, 1, 6, 2, 5])
    permuted = exp.soft_top5_membership(scores.index_select(0, permutation), alpha=2.0)
    restored = torch.empty_like(permuted)
    restored[permutation] = permuted
    assert torch.allclose(restored, exp.soft_top5_membership(scores, alpha=2.0), atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        exp.soft_top5_membership(scores + 100.0, alpha=2.0),
        exp.soft_top5_membership(scores, alpha=2.0),
        atol=1e-10,
        rtol=1e-10,
    )


def test_soft_top5_implicit_gradient_matches_finite_difference() -> None:
    scores = torch.tensor([-2.0, -0.2, 0.1, 0.9, 1.4, 2.8, 3.2, -1.3], dtype=torch.float64)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return -torch.log(exp.soft_top_k(value, k=5, alpha=2.0)).mean()

    analytic, numeric = exp._finite_difference_gradient(objective, scores, epsilon=1e-6)
    assert torch.allclose(analytic, numeric, atol=2e-4, rtol=2e-4)
    assert torch.isfinite(analytic).all()


def test_decoupled_loss_excludes_other_positives() -> None:
    scores = torch.tensor([2.0, 1.0, 0.4, 0.1, -0.3, -0.9], dtype=torch.float64)
    mask = torch.tensor([True, True, False, False, False, False])
    observed = exp.decoupled_terms(scores, mask, tau=0.5)
    expected_denominator = torch.logsumexp(scores[~mask] / 0.5, dim=0)
    expected = expected_denominator - scores[mask] / 0.5
    assert torch.allclose(observed, expected)
    wrong_denominator = torch.logsumexp(scores[1:] / 0.5, dim=0)
    assert not torch.isclose(observed[0], wrong_denominator - scores[0] / 0.5)


def test_hybrid_loss_keeps_gradients_from_both_terms() -> None:
    scores = torch.tensor([2.0, 1.0, 0.4, 0.1, -0.3, -0.9], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, True, False, False, False, False])
    total, decoupled, top5 = exp.hybrid_loss(scores, mask, [0, 1], alpha=2.0, lambda_top5=0.25)
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(decoupled)
    assert float(top5.detach()) > 0.0
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_parent_score_blockwise_matches_reference_for_one_and_two_best_chunks() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "a", "b", "b", "b", "c"])
    documents = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query = torch.tensor([[1.0, 0.0]])
    actual = exp.compute_parent_scores(query, documents, index, chunk_block_size=2)
    expected = exp.reference_parent_scores(query[0], documents, index).unsqueeze(0)
    assert torch.allclose(actual, expected)
    assert actual[0, index.doc_ids.index("a")] == pytest.approx(0.5)
    assert actual[0, index.doc_ids.index("b")] == pytest.approx(1.0)
    assert actual[0, index.doc_ids.index("c")] == pytest.approx(0.0)


def test_parent_score_noncontiguous_mapping_is_exact_and_differentiable() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "b", "a", "c", "b"])
    assert index.chunk_indices is not None
    documents = torch.randn(5, 7, dtype=torch.float64)
    query = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    actual = exp.compute_parent_scores(query, documents, index, chunk_block_size=2)
    expected = torch.stack([exp.reference_parent_scores(row, documents, index) for row in query])
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    actual.sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_document_block_re_normalizes_fp16_after_fp32_dequantization() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "a", "b"])
    original = torch.nn.functional.normalize(torch.randn(3, 9), p=2, dim=-1)
    stored_fp16 = original.numpy().astype(np.float16)
    block, _positions = exp._document_block(
        stored_fp16,
        index,
        0,
        2,
        torch.device("cpu"),
        torch.float32,
    )
    assert torch.allclose(torch.linalg.vector_norm(block, dim=-1), torch.ones(3), atol=1e-6, rtol=0.0)


def test_numpy_parent_reference_uses_cosine_after_fp16_dequantization() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "a", "b"])
    documents = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float16)
    values = exp.numpy_top2_parent_scores(np.asarray([1.0, 0.0], dtype=np.float32), documents, index, [0, 1])
    assert values[0] == pytest.approx((1.0 + 0.8) / 2.0, abs=2e-3)
    assert values[1] == pytest.approx(0.0, abs=1e-6)


def test_weighted_rrf_uses_scores_for_full_dense_ranking_and_zero_missing_source() -> None:
    dense = ["a", "b", "c", "d"]
    sparse = ["c", "x"]
    result = exp.weighted_rrf_ranking(dense, sparse, dense_weight=0.5, rrf_k=10, limit=6)
    assert set(dense).issubset(result)
    assert "x" in result
    # x is ranked by its sparse score, not appended after the dense list.
    assert result.index("x") < len(result) - 1
    with pytest.raises(ValueError):
        exp.weighted_rrf_ranking(["a", "a"], [], dense_weight=0.5, rrf_k=10)


def test_nested_partitions_exclude_outer_fold_and_are_disjoint() -> None:
    folds = {f"fold_{i}": [f"q{i}_{j}" for j in range(3)] for i in range(5)}
    parts = exp.nested_partitions(folds, "fold_0")
    assert len(parts) == 4
    outer = set(folds["fold_0"])
    seen_validation: set[str] = set()
    for part in parts:
        train = set(part["train_qids"])
        validation = set(part["validation_qids"])
        assert not train & validation
        assert not train & outer
        assert not validation & outer
        seen_validation |= validation
    assert seen_validation == set().union(*(set(folds[name]) for name in folds if name != "fold_0"))


def test_selection_tie_policy_prefers_a_then_c_smaller_lambda_then_b() -> None:
    base = {
        "recall@5": 0.8,
        "multi_gold_recall@5": 0.7,
        "precision@5": 0.2,
        "recall@16": 0.9,
        "mrr@5": 0.6,
    }
    records = [
        {**base, "config": {"arm": "B_softtop5", "alpha": 1.0, "lambda_top5": 0.0}},
        {**base, "config": {"arm": "C_hybrid", "alpha": 5.0, "lambda_top5": 0.5}},
        {**base, "config": {"arm": "C_hybrid", "alpha": 5.0, "lambda_top5": 0.1}},
        {**base, "config": {"arm": "A_decoupled", "alpha": None, "lambda_top5": 0.0}},
    ]
    assert exp.select_candidate(records)["config"]["arm"] == "A_decoupled"
    records = records[:3]
    chosen = exp.select_candidate(records)
    assert chosen["config"]["arm"] == "C_hybrid"
    assert chosen["config"]["lambda_top5"] == 0.1


def test_metrics_report_multi_gold_and_ceiling_scope() -> None:
    answers = {"q1": {"a"}, "q2": {"a", "b"}}
    predictions = {"q1": ["a"], "q2": ["b", "x", "a"]}
    metrics = exp.aggregate_metrics(predictions, answers, ["q1", "q2"])
    assert metrics["evaluable_queries"] == 2
    assert metrics["cardinality"]["single_gold"]["recall@5"] == pytest.approx(1.0)
    assert metrics["cardinality"]["multi_gold"]["recall@5"] == pytest.approx(1.0)
    assert metrics["precision@5"] == pytest.approx(3.0 / 10.0)


def test_canonical_labels_match_frozen_policy() -> None:
    answers, stats = exp.canonical_labels()
    assert len(answers) == 7000
    assert stats["evaluable_queries"] == 6991
    assert stats["non_evaluable_queries"] == 9
    assert stats["affected_queries"] == 13
    assert stats["canonicalized_duplicate_occurrences"] == 2
    assert stats["dropped_empty_occurrences"] == 11
    assert stats["label_fingerprint"] == "9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9"


def test_success_marker_fingerprint_is_fail_closed(tmp_path) -> None:
    directory = tmp_path / "stage"
    exp.write_success(directory, stage="test", fingerprint="abc")
    assert exp.require_success(directory, "abc")["status"] == "PASS"
    with pytest.raises(RuntimeError):
        exp.require_success(directory, "different")


def test_soft_top5_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        exp.soft_top_k(torch.randn(5), k=5, alpha=1.0)
