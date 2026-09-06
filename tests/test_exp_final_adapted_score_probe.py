from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp_final_adapted_score_probe as probe


def test_continuous_feature_contract_at_cutoff():
    compact = {
        "q": (
            {"frozen_query_cosine": 0.9, "rank5_rank6_gap": 0.02, "top10_std": 0.1},
            {"d": (5, 0.7, 1.5, -0.1, 0.0, 0.02, 0.05)},
        )
    }
    value = probe.continuous_features("q", ["d"], compact)
    assert value.shape == (1, len(probe.adapted_feature_names()))
    assert np.isfinite(value).all()
    assert value[0, 0] == 5
    assert value[0, 8] == 1
    assert value[0, 9] == 1
    assert np.isclose(value[0, 10], 0.1)


def test_feature_names_are_unique():
    names = probe.adapted_feature_names()
    assert len(names) == len(set(names))
