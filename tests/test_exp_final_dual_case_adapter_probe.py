from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from exp_final_dual_case_adapter_probe import bounded_screen_rrf, prototype_orders, weighted_rrf
from exp_final_dual_case_meta_probe import bucket


def test_weighted_rrf_endpoints_and_missing_source_semantics():
    left = ["a", "b", "c"]
    right = ["c", "d"]
    assert weighted_rrf({"left": left, "right": right}, {"left": 1.0, "right": 0.0})[:3] == left
    assert weighted_rrf({"left": left, "right": right}, {"left": 0.0, "right": 1.0})[:2] == right
    assert len(weighted_rrf({"left": left, "right": right}, {"left": .9, "right": .1})) == 4


def test_weighted_rrf_rejects_invalid_contract():
    with pytest.raises(ValueError):
        weighted_rrf({"a": ["x"]}, {"b": 1.0})
    with pytest.raises(ValueError):
        weighted_rrf({"a": ["x"]}, {"a": .5})


def test_bounded_screen_matches_full_rrf_top_five():
    anchor = [f"d{i:03d}" for i in range(500)]
    content = list(reversed(anchor))
    prototype = anchor[3:] + anchor[:3]
    rankings = {"incumbent": anchor, "content": content, "prototype": prototype}
    weights = {"incumbent": .70, "content": .15, "prototype": .15}
    assert bounded_screen_rrf(rankings, weights)[:5] == weighted_rrf(rankings, weights)[:5]
    with pytest.raises(ValueError):
        bounded_screen_rrf(rankings, {"incumbent": .6, "content": .2, "prototype": .2})


def test_prototype_orders_reward_matching_label_support():
    support_qids = ["s1", "s2", "s3"]
    labels = {"s1": {"a"}, "s2": {"a"}, "s3": {"b"}}
    support = np.asarray([[1.0, 0.0], [.8, .2], [0.0, 1.0]], dtype=np.float32)
    support /= np.linalg.norm(support, axis=1, keepdims=True)
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = prototype_orders(targets, support, support_qids, labels, ["a", "b", "c"])
    for mode in ("max", "top2", "logmeanexp"):
        assert result[mode][0][0] == "a"
        assert result[mode][1][0] == "b"
        assert set(result[mode][0]) == {"a", "b"}


def test_meta_bucket_is_stable_and_bounded():
    assert bucket("123") == bucket("123")
    assert 0 <= bucket("different") < 4
