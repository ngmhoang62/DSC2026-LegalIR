import math
import random

import numpy as np
import pytest
import torch

from exp_final.contracts import (correct, deployment_recipe, export_submission, metrics, prefix, rrf, splits, union)
from exp_final.learning import ParentBank, boundary_loss, ce_loss, multi_loss, replay_backward, select_negatives


def test_folds():
    folds = {f"fold_{i}": [str(i)] for i in range(5)}
    for i in range(5):
        train, cal, test = splits(folds, f"fold_{i}")
        assert len(train) == 3 and test == [str(i)]
        assert cal == [str((i-1) % 5)]


def test_original_and_canonical_metrics():
    p = {"a": ["1", "2"], "b": ["3"]}
    m = metrics(p, {"a": {"1"}, "b": {"3", "4"}}, output_count=True)
    assert m["recall@5"] == .75
    assert m["precision@5"] == .75
    assert metrics(p, {"a": set(), "b": {"3"}}, exclude_empty=True)["queries"] == 1
    with pytest.raises(ValueError):
        metrics({"a": ["1", "1"]}, {"a": {"1"}})


def test_empty_predictions_are_diagnostic_only():
    gold = {"a": {"1"}}
    with pytest.raises(ValueError, match="Invalid official prediction count"):
        metrics({"a": []}, gold)
    diagnostic = metrics({"a": []}, gold, allow_empty_predictions=True)
    assert diagnostic["recall@5"] == 0.0
    assert diagnostic["precision@5"] == 0.0
    assert diagnostic["queries"] == 1


def test_no_pool_truncation():
    assert len(union([list(range(100)), list(range(90, 190)), list(range(180, 280))])) == 280


def test_parent_scores_and_gradient():
    torch.manual_seed(3)
    documents = torch.randn(9, 7)
    parent = [0, 1, 0, 2, 1, 2, 2, 3, 2]
    bank = ParentBank(documents, parent, device="cpu")
    q = torch.randn(7, requires_grad=True)
    values, selected = bank.mine(q)
    rescored = bank.rescore(q, selected[0])
    reference = torch.stack([(bank.vectors[torch.tensor(parent) == p] @ q).topk(min(2, parent.count(p))).values.mean() for p in range(4)])
    assert torch.allclose(values[0], reference, atol=1e-6)
    assert torch.allclose(rescored, reference, atol=1e-6)
    g1, = torch.autograd.grad(rescored.sum(), q, retain_graph=True)
    g2, = torch.autograd.grad(reference.sum(), q)
    assert torch.allclose(g1, g2)


def test_negative_cosine_and_ties():
    b = ParentBank(np.array([[1., 0.], [1., 0.], [1., 0.]]), [0, 0, 1], device="cpu")
    s, ix = b.mine(torch.tensor([-1., 0.]))
    assert s.tolist() == [[-1., -1.]]
    assert ix.tolist() == [[[0, 1], [2, 2]]]


def test_independent_positive_loss():
    p, n = torch.tensor([1., 2.]), torch.tensor([-1., 0.])
    assert torch.allclose(multi_loss(p, n, 1), torch.stack([multi_loss(x[None], n, 1) for x in p]).mean())


def test_boundary_normalization():
    p = torch.tensor([0., 0.], requires_grad=True)
    n = torch.tensor([0., 0., 0.])
    value = boundary_loss(p, n, torch.tensor([6, 7]), torch.tensor([1, 2, 10]))
    assert float(value.detach()) == pytest.approx(math.log(2))
    assert boundary_loss(p, n, torch.tensor([6, 7]), torch.tensor([8, 9, 10])) == 0


@pytest.mark.parametrize("count", [6, 64])
def test_negative_policy(count):
    universe = list(map(str, range(200)))
    sources = {"e5": universe, "lal": universe[::-1], "bm25": universe[50:]}
    a = select_negatives(universe, {"0", "3"}, sources, universe, "q", 0, count)
    assert len(a) == len(set(a)) == count
    assert not set(a) & {"0", "3"}
    assert a == select_negatives(universe, {"0", "3"}, sources, universe, "q", 0, count)
    assert a != select_negatives(universe, {"0", "3"}, sources, universe, "q", 1, count)


def test_replay_dropout_gradients():
    torch.manual_seed(112)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.3), torch.nn.Linear(5, 1))
    batches = [torch.randn(2, 3), torch.randn(2, 3)]
    state = torch.get_rng_state()
    direct = ce_loss(torch.cat([model(x).flatten() for x in batches]), 2)
    direct.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(); torch.set_rng_state(state)
    replay_backward(model, batches, 2)
    for p, g in zip(model.parameters(), grads):
        assert torch.allclose(p.grad, g, atol=1e-6)


def test_noop_and_tail():
    order = list(map(str, range(60))); scores = list(range(60, 0, -1))
    assert correct(order, scores, {}, 0) == (order, scores)
    result, _ = correct(order, scores, {d: float(d) for d in order[:50]}, .25)
    assert result[50:] == order[50:]
    assert prefix(order, scores) == order[:5]
    assert set(prefix(order, scores, .001)) <= set(order[:5])


def test_complete_recipe_not_frankenstein():
    locks = [dict(epoch=1, family="lr", block=0, beta=.3, pool="expanded", alpha=.1, confidence=False, threshold=.7),
             dict(epoch=2, family="lm", block=1, beta=.5, pool="expanded", alpha=0, confidence=False, threshold=.8)]
    assert deployment_recipe(locks) in locks


def test_submission(tmp_path):
    result = export_submission(tmp_path, {"q": ["d"]}, ["q"], ["d"])
    assert result["status"] == "COMPLETE_SUBMISSION" and not result["uploaded"]
    with pytest.raises(ValueError):
        export_submission(tmp_path, {"q": ["bad"]}, ["q"], ["d"])


def test_profile_budget_never_drops_folds():
    from exp_final.pipeline import choose_profile
    costs = dict(frozen=1, query_epoch=100, query_scoring=1, ml=1, public=1,
                 jina=1, ce_inner_final=1e6, ce_refit=1e6, ce_scoring=1e6)
    chosen, _ = choose_profile(costs, 48, False)
    assert chosen["profile"] == "P4"
    costs["query_epoch"] = 1e9
    with pytest.raises(RuntimeError, match="BUDGET_EXTENSION_REQUIRED"):
        choose_profile(costs, 48, False)


def test_synthetic_fivefold_to_submission(tmp_path, monkeypatch):
    """Control-flow integration; fake learners are deliberately not metric evidence."""
    from exp_final import pipeline as p, contracts as c
    docs = [f"d{i}" for i in range(8)]
    class SyntheticData:
        def __init__(self):
            self.folds = {f"fold_{i}": [f"q{i}"] for i in range(5)}
            self.train = {f"q{i}": {} for i in range(5)}
            self.public = {"test": {}}
            self.original = self.gold = {q: {"d7"} for q in self.train}
            self.questions = {q: "synthetic" for q in [*self.train, "test"]}
            self.doc_ids = docs
            self.fingerprint = "SYNTHETIC_ONLY"
    class SyntheticStore:
        def close(self):
            pass
        def rankings(self, q):
            return {s: docs for s in ('e5','lal','bm25','trigram')}
    monkeypatch.setattr(p, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(p, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(c, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(p, "Data", SyntheticData)
    monkeypatch.setattr(p, "SourceStore", SyntheticStore)
    monkeypatch.setattr(p, "train_query", lambda *a, **k: {"synthetic": True})
    monkeypatch.setattr(p, "fit_ranker", lambda *a, **k: None)
    def fake_score(data, qids, checkpoint, output):
        for q in qids:
            c.write(output / f"{q}.json", {"order": docs, "scores": list(range(8, 0, -1)), "synthetic": True})
    monkeypatch.setattr(p, "score_queries", fake_score)
    monkeypatch.setattr(p, "upstream", lambda *a, **k: {"order": docs, "scores": list(range(8, 0, -1))})
    c.write(p.RESULTS / "RESOURCE_LOCK.json", {"epochs": 1, "ce": 0, "jina": False, "microbatch": 1})
    for i in range(5):
        result = p.run_fold(f"fold_{i}")
        assert result["metrics"]["official"]["recall@5"] == 0
        assert result["status"] == "COMPLETE_FOLD"
    oof = p.evaluate_oof()
    assert len(oof["folds"]) == 5
    assert oof["pooled"]["official"]["recall@5"] == 0
    p.fit_final()
    result = p.predict_public()
    assert result["status"] == "COMPLETE_SUBMISSION"
    assert (p.RESULTS / "public/submission.zip").exists()


def test_censored_dense_feature_is_not_missing_score(tmp_path, monkeypatch):
    from exp_final import fusion
    monkeypatch.setattr(fusion, "CACHE", tmp_path)
    class D:
        fingerprint = "test"
        metadata = {"outside": {"parent_chunk_count": 1, "parent_token_length": 5}}
        questions = {"q": "query"}
        def exact(self, q, docs, source):
            return [.25] * len(docs)
    class S:
        def get(self, q, source):
            return [{"doc_id": "inside", "rank": 1, "score": .5}]
    x = fusion.features(D(), S(), "q", ["outside"], 0)
    values = dict(zip(fusion.feature_names(0), x[0]))
    assert values["e5_score"] == .25
    assert values["e5_score_available"] == 1
    assert values["e5_rank_known"] == 0
    assert values["e5_rank"] == 501


def test_source_payload_and_scope_are_immutable(tmp_path):
    from exp_final.data import SourceStore
    s=SourceStore(tmp_path/'source.sqlite')
    s.put('q','e5',[dict(doc_id='a',rank=1,score=.1)],'frozen')
    with pytest.raises(ValueError,match='payload changed'):
        s.put('q','e5',[dict(doc_id='a',rank=1,score=.2)],'frozen')
    with pytest.raises(ValueError,match='provenance changed'):
        s.put('q','e5',[],'other')
    s.close()


@pytest.mark.parametrize('prediction',[[],['a']*2,['a','b','c','d','e','f'],['invalid']])
def test_invalid_submission_shapes(tmp_path,prediction):
    with pytest.raises(ValueError):
        export_submission(tmp_path,{'q':prediction},['q'],list('abcdef'))


def test_adaptive_recall_never_increases():
    order=list('abcdef');scores=[10,1,0,-1,-2,-3];gold={'b'}
    small=prefix(order,scores,.01)
    assert small==['a']
    assert len(set(small)&gold)<=len(set(order[:5])&gold)


def test_threshold_requires_exact_retained_gold_hits():
    from exp_final.contracts import select_threshold
    rows={'q':dict(order=list('abcde'),scores=[10,1,0,-1,-2])}
    assert select_threshold(rows,{'q':{'b'}}) is None
    assert select_threshold(rows,{'q':{'a'}}) is not None


def test_checkpoint_rng_scheduler_roundtrip(tmp_path):
    from exp_final.learning import checkpoint,rng_state,set_rng
    from exp_final.contracts import read,sha
    class Tiny(torch.nn.Linear):
        def adapter_state(self):
            return self.state_dict()
    model=Tiny(2,1);opt=torch.optim.AdamW(model.parameters());sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
    torch.manual_seed(112)
    model(torch.ones(1,2)).sum().backward();opt.step();sched.step()
    p=tmp_path/'resume.pt';checkpoint(p,model,opt,sched,epoch=1,position=16)
    assert read(p.with_suffix('.sha.json'))['sha256']==sha(p)
    state=torch.load(p,weights_only=False)
    expected=torch.rand(4);set_rng(state['rng'])
    assert torch.equal(torch.rand(4),expected)
    assert state['scheduler']['last_epoch']==sched.last_epoch
    assert state['position']==16


def test_global_boundary_not_sampled_position():
    p=torch.tensor([.2]);n=torch.tensor([.3,.1])
    assert boundary_loss(p,n,torch.tensor([600]),torch.tensor([40,100]))==0
    assert boundary_loss(p,n,torch.tensor([600]),torch.tensor([4,100]))>0


def test_alpha_zero_ties_exact():
    order=['b','a','c'];scores=[0.,0.,0.]
    assert correct(order,scores,{'b':-100,'a':100,'c':0},0)==(order,scores)


def test_expanded_pool_is_not_original_frozen_pool(tmp_path,monkeypatch):
    from exp_final import fusion
    class S:
        def candidates(self,q):return ['a','b']
    class M:
        def predict(self,x):return np.arange(len(x))
    monkeypatch.setattr(fusion,'features',lambda d,s,q,docs,b:np.zeros((len(docs),1)))
    r=dict(family='lm',block=0,pool='frozen',beta=0)
    frozen=fusion.upstream(None,S(),'q',M(),r)
    expanded=fusion.upstream(None,S(),'q',M(),r|{'pool':'expanded'},dict(order=['c','a','b']))
    assert frozen['order'][0]=='b' and expanded['order'][0]=='c'


def test_byte_bounded_mmap_keeps_live_borrow_valid(tmp_path):
    from exp_final.jina import BoundedMmapCache
    for i in range(4):np.save(tmp_path/f'{i}.npy',np.full(100,i,dtype=np.float32))
    c=BoundedMmapCache(max_bytes=600)
    borrowed=c.load(tmp_path/'0.npy')
    for i in range(1,4):c.load(tmp_path/f'{i}.npy')
    assert c.bytes<=600 and c.open_paths==1
    assert borrowed.sum()==0


def test_lazy_rows_do_not_retain_payloads(tmp_path):
    from exp_final.contracts import DiskRows,write
    write(tmp_path/'q.json',{'value':1})
    rows=DiskRows(tmp_path,['q'])
    assert rows.get('q')['value']==1 and rows.get('missing') is None
    write(tmp_path/'q.json',{'value':2})
    assert rows['q']['value']==2
    assert not hasattr(rows,'payloads')
