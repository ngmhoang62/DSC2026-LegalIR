from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from exp_final_memory_vote_fix_probe import repaired_memory_features


def test_repaired_votes_are_distinct_and_frequency_is_fold_local():
    labels = {"q1": {"a", "b"}, "q2": {"b"}}
    support = ["q1", "q2"]
    by_doc = defaultdict(list, {"a": [0], "b": [0, 1]})
    frequency = Counter({"a": 1, "b": 2})
    block = repaired_memory_features(
        np.asarray([0.9, 0.8], dtype=np.float32),
        ["a", "b"], support, labels, by_doc, frequency,
    )
    assert block.shape == (2, 14)
    assert np.isfinite(block).all()
    # Column 6 is raw/cardinality-normalized mass; column 7 additionally
    # compensates for document frequency, so frequent label b must differ.
    assert block[1, 6] > block[1, 7]
    assert block[0, 6] == block[0, 7]


def test_self_query_is_removed_from_similarity_and_frequency_support():
    labels = {"q1": {"a"}, "q2": {"b"}}
    support = ["q1", "q2"]
    by_doc = defaultdict(list, {"a": [0], "b": [1]})
    frequency = Counter({"a": 1, "b": 1})
    block = repaired_memory_features(
        np.asarray([1.0, 0.5], dtype=np.float32),
        ["a", "b"], support, labels, by_doc, frequency,
        self_qid="q1", self_index=0,
    )
    assert block[0, 0] == 0.0
    assert block[0, 1] == 0.0
    assert block[1, 0] == 1.0
