from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from exp_final_lal_adapter_probe import rrf


def test_rrf_endpoints_reproduce_inputs():
    left = ["a", "b", "c"]
    right = ["c", "b", "a"]
    assert rrf(left, right, 0.0) == left
    assert rrf(left, right, 1.0) == right


def test_rrf_union_is_unique():
    result = rrf(["a", "b"], ["b", "c"], 0.5)
    assert set(result) == {"a", "b", "c"}
    assert len(result) == len(set(result))
