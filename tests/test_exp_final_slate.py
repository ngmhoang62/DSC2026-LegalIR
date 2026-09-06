import numpy as np

from exp_final.slate import FoldLabelStats,dense_relation_features,replacement_class,slate_features


def test_leave_one_query_out_label_stats():
    labels={"q1":{"a","b"},"q2":{"a"}}
    stats=FoldLabelStats(labels,["q1","q2"])
    assert stats.frequency_for("a",labels["q1"])==1
    assert stats.frequency_for("b",labels["q1"])==0
    assert stats.cooccur("a","b",labels["q1"])==0


def test_replacement_classes():
    assert replacement_class({"x"},"d","x")==2
    assert replacement_class({"d"},"d","x")==0
    assert replacement_class({"x","d"},"d","x")==1
    assert replacement_class(set(),"d","x")==1


def test_slate_features_are_finite_and_relational():
    ranking=["a","b","c","d","e","f"]
    systems={"one":ranking,"two":["a","f","b","c","d","e"]}
    titles={"a":"Nghi dinh 43 2014 ND CP","e":"Luật đất đai","f":"Nghi dinh 148 2020 sua doi Nghi dinh 43 2014"}
    labels={"q":{"a","f"}};stats=FoldLabelStats(labels,["q"])
    value=slate_features("quy định đất đai và thủ tục",ranking,5,systems,titles,stats,labels["q"])
    assert value.ndim==1 and len(value)>40
    assert (value==value).all()


def test_dense_relation_features_reward_novel_direction():
    query=np.array([1.,1.,0.],dtype=np.float32)
    documents=np.array([
        [1.,0.,0.],[1.,0.,0.],[1.,0.,0.],[1.,0.,0.],
        [1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
    ],dtype=np.float32)
    novel=dense_relation_features(query,documents,5)
    irrelevant=dense_relation_features(query,documents,6)
    assert novel.shape==(21,)
    assert np.isfinite(novel).all()
    assert novel[0]>irrelevant[0]
    assert novel[18]>irrelevant[18]


def test_dense_relation_features_reject_dimension_mismatch():
    with np.testing.assert_raises(ValueError):
        dense_relation_features(np.ones(2),np.ones((6,3)),5)
