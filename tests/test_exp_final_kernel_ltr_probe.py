from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp_final_kernel_ltr_probe", ROOT / "scripts/exp_final_kernel_ltr_probe.py"
)
KERNEL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(KERNEL)

META_SPEC = importlib.util.spec_from_file_location(
    "exp_final_meta_ltr_probe", ROOT / "scripts/exp_final_meta_ltr_probe.py"
)
META = importlib.util.module_from_spec(META_SPEC)
assert META_SPEC.loader is not None
META_SPEC.loader.exec_module(META)


def test_top_indices_masks_heldout_support_and_is_deterministic():
    row = csr_matrix([[0.9, 0.9, 0.8, 0.7]], dtype=np.float32)
    columns, values = KERNEL.top_indices(row, np.asarray([False, True, True, True]), depth=2)
    assert columns.tolist() == [1, 2]
    assert np.allclose(values, [0.9, 0.8])


def test_channel_evidence_uses_only_three_strongest_neighbours():
    labels = {"q0": {"d"}, "q1": {"d"}, "q2": {"d"}, "q3": {"d"}}
    result = KERNEL.channel_evidence(
        np.asarray([0, 1, 2, 3]), np.asarray([1.0, 0.8, 0.6, 0.5]),
        ["q0", "q1", "q2", "q3"], labels,
    )
    expected = 1.0 + 0.4 * 0.8**2 + 0.15 * 0.6**2
    assert np.isclose(result["d"][3], expected)


def test_kernel_feature_schema_is_finite_and_missing_docs_are_zero():
    values, scores = KERNEL.kernel_features(
        {"seen": (0.8, 0.2, 0.0, 0.88)},
        {"seen": (0.7, 0.0, 0.0, 0.7)},
        ["seen", "missing"],
        {"seen": 2},
    )
    assert values.shape == (2, len(KERNEL.KERNEL_NAMES))
    assert np.isfinite(values).all()
    assert np.count_nonzero(values[1]) == 0
    assert scores["seen"] == 1.0


def test_meta_features_encode_consensus_without_ids():
    systems = {
        "a": {"q": ["d1", "d2", "d3"]},
        "b": {"q": ["d2", "d1", "d4"]},
    }
    docs, values = META.rows_for_query("q", systems, ["a", "b"])
    assert values.shape == (4, len(META.feature_names(["a", "b"])))
    assert docs == sorted(docs)
    row = docs.index("d1")
    # d1 occurs in both sources and is top-3 in both.
    assert values[row, -13] == 2
    assert values[row, -11] == 2
    assert np.isfinite(values).all()
