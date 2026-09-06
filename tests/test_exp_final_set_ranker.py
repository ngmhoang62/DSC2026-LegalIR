import torch

from exp_final.set_ranker import ResidualSetRanker,multi_positive_set_loss


def test_set_ranker_shapes_and_cross_candidate_gradient():
    torch.manual_seed(1);model=ResidualSetRanker(7,5,hidden=16,heads=4,layers=1,depth=10,dropout=0.)
    candidate=torch.randn(3,10,7,requires_grad=True);query=torch.randn(3,5)
    logits=model(candidate,query);assert logits.shape==(3,10)
    targets=torch.zeros(3,10);targets[:,[0,6]]=1
    loss=multi_positive_set_loss(logits,targets);loss.backward()
    assert torch.isfinite(loss) and candidate.grad is not None and torch.isfinite(candidate.grad).all()


def test_set_loss_includes_all_positives_without_competition():
    good=torch.tensor([[3.,3.,-2.,-2.]])
    bad=torch.tensor([[3.,-3.,-2.,-2.]])
    target=torch.tensor([[1,1,0,0]],dtype=torch.float32)
    assert multi_positive_set_loss(good,target)<multi_positive_set_loss(bad,target)


def test_set_loss_rejects_all_negative_batch():
    try:multi_positive_set_loss(torch.zeros(1,3),torch.zeros(1,3))
    except ValueError:return
    raise AssertionError('Expected no-positive rejection')
