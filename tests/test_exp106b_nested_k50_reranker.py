import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp106b_nested_k50_reranker import _outer_contexts, build_training_group, choose_alpha, multi_positive_hard_negative_loss, validate_candidate_rows


def _rows():
    return [{"doc_id": value, "rank": index + 1, "stage1_score": 1.0 / (index + 2)} for index, value in enumerate(("g1", "n1", "n2", "g2", "n3"))]


def test_multi_gold_group_never_uses_gold_as_negative():
    rows = _rows(); group = build_training_group(rows, {"g1", "g2"}, {"n1": .8, "n2": .7, "n3": .9})
    assert {row["doc_id"] for row in group[:2]} == {"g1", "g2"}
    assert len({row["doc_id"] for row in group}) == len(group)


def test_listwise_loss_has_gradient():
    logits = torch.tensor([.2, .9, .1, .7], requires_grad=True); labels = torch.tensor([1., 0., 0., 1.])
    loss = multi_positive_hard_negative_loss(logits, labels); loss.backward()
    assert float(loss.detach()) > 0 and logits.grad is not None


def test_alpha_selection_prefers_complementary_ce():
    rows = {"q": [
        {"doc_id": "g1", "rank": 1, "stage1_score": 1.0},
        {"doc_id": "n1", "rank": 2, "stage1_score": .8},
        {"doc_id": "n2", "rank": 3, "stage1_score": .6},
        {"doc_id": "n3", "rank": 4, "stage1_score": .4},
        {"doc_id": "n4", "rank": 5, "stage1_score": .2},
        {"doc_id": "g2", "rank": 6, "stage1_score": .1},
    ]}
    scores = {"q": {"g1": .9, "n1": .1, "n2": .0, "n3": -.1, "n4": -.2, "g2": .8}}
    selected = choose_alpha(rows, scores, {"q": {"g1", "g2"}})["selected"]
    assert selected["alpha"] > 0


def test_candidate_exact_score_and_shape_validation():
    rows = [{"qid": "q", "candidates": [{"doc_id": str(i), "rank": i + 1, "stage1_score": float(50-i)} for i in range(50)]}]
    audit = validate_candidate_rows(rows, {"q": {"0"}})
    assert audit["malformed"] == 0 and audit["metrics"]["recall@50"] == 1.0


def test_nested_contexts_never_train_on_their_target_or_outer_holdout():
    folds = {f"fold_{i}": [f"q{i}"] for i in range(5)}
    contexts = _outer_contexts(folds)
    assert len(contexts) == 25
    for context in contexts:
        assert not (set(context["target_qids"]) & set(context["train_qids"]))
        if context["kind"] == "inner":
            assert f"q{context['outer'].split('_')[1]}" not in set(context["train_qids"])
