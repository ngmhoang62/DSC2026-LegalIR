import numpy as np
import pytest

from exp035_retrieval_error_adjudication import TAGS, _tag
from exp036_coverage_aware_fusion import _calibrated_rrf, _fold_name, _guarded, _orders, _stable
from exp038_multihead_e5_retrieval import MultiHeadProjection, aggregate_heads, multi_positive_pairwise_loss


def test_exp035_tags_are_closed_and_diagnostic_only():
    row={"gold_count":1,"rrf_rank":40,"e5_rank_exp022":12,"bm25_rank_exp022":100,"parse_mode":"structured","question":"abc"}
    assert _tag(row)=="fusion_displacement"
    assert "fusion_displacement" in TAGS
    assert set(TAGS)==set(("valid_direct_gold","multi_gold_secondary_or_background","underspecified_query","suspected_label_semantic_mismatch","source_generation_miss","fusion_displacement","structural_or_evidence_failure","unresolved"))


def test_exp036_deterministic_ties_and_guardrail_only_heads():
    assert _stable((("z",1.),("a",1.),("b",.5)))==["a","z","b"]
    rec={"qid":"q","candidates":[{"doc_id":"z","e5_rank":10,"bm25_rank":10},{"doc_id":"a","e5_rank":1,"bm25_rank":None},{"doc_id":"b","e5_rank":None,"bm25_rank":1}]}
    # Heads get protected, all others remain in learned-score ordering.
    guarded=_guarded(np.asarray([.9,.1,.2]),rec)
    assert guarded[1]>1e5 and guarded[2]>1e5 and guarded[0]<1
    assert _orders([rec],{"q":np.asarray([.5,.5,.1])})["q"]==["a","z","b"]


def test_exp036_guardrail_does_not_change_candidate_membership():
    rec={"qid":"q","candidates":[{"doc_id":"x","e5_rank":None,"bm25_rank":None},{"doc_id":"y","e5_rank":2,"bm25_rank":99}]}
    order=_orders([rec],{"q":np.asarray([1.,0.])},True)["q"]
    assert sorted(order)==["x","y"]


def test_exp036_normalizes_cli_fold_name():
    assert _fold_name("0")=="fold_0"
    assert _fold_name("fold_4")=="fold_4"


def test_exp036_rrf_never_parses_aggregation_metadata_as_coefficients():
    rec={"fold":"fold_2","candidates":[{"e5_rank":1,"bm25_rank":None}]}
    assert np.isclose(_calibrated_rrf(rec)[0], .65/33 + .35/(32+10**6))


def test_exp038_identity_initialization_and_head_normalization():
    torch = pytest.importorskip("torch")
    model=MultiHeadProjection(dim=4,rank=2)
    q=torch.tensor([[1.,2.,3.,4.]])
    heads=model(q)
    assert torch.allclose(heads[:,0],heads[:,1]) and torch.allclose(heads[:,0],heads[:,2])
    # Identical three-head score is unchanged by normalized log-sum-exp.
    score=torch.tensor([[2.,2.,2.]])
    assert torch.allclose(aggregate_heads(score),torch.tensor([2.]))


def test_exp038_multi_positive_loss_has_positive_gradients():
    torch=pytest.importorskip("torch")
    s=torch.tensor([.4,.2,.1,.0],requires_grad=True)
    loss=multi_positive_pairwise_loss(s,torch.tensor([True,True,False,False]),torch.tensor([False,False,True,True]))
    loss.backward()
    assert s.grad[0] != 0 and s.grad[1] != 0
