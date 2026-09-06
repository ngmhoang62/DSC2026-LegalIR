"""Unit tests for Gemini unified LTR feature pipeline and integrity checks."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap


def test_label_contract():
    labels, audit = get_canonical_labels()
    assert audit["evaluable_query_count"] == 6991
    assert audit["non_evaluable_query_count"] == 9
    assert audit["duplicate_occurrences"] == 2
    assert audit["empty_occurrences"] == 11
    assert len(labels) == 7000


def test_cv_folds_integrity():
    folds = get_cv_folds()
    assert len(folds) == 5
    all_qids = set()
    for f_name, qids in folds.items():
        assert len(qids) == 1400
        assert not (set(qids) & all_qids), f"Duplicate qids across folds: {f_name}"
        all_qids.update(qids)
    assert len(all_qids) == 7000


def test_metrics_correctness():
    labels = {
        "q1": {"d1"},
        "q2": {"d2", "d3"},
        "q3": {"d4"},
    }
    rankings = {
        "q1": ["d1", "d10", "d11", "d12", "d13"], # hit at rank 1 -> r=1.0, p=0.2, mrr=1.0
        "q2": ["d5", "d2", "d6", "d7", "d8"],    # 1 hit at rank 2 out of 2 -> r=0.5, p=0.2, mrr=0.5
        "q3": ["d9", "d10", "d11", "d12", "d13"], # 0 hits -> r=0.0, p=0.0, mrr=0.0
    }
    m = compute_metrics(rankings, labels, k=5)
    assert abs(m["recall_at_5"] - (1.0 + 0.5 + 0.0) / 3.0) < 1e-6
    assert abs(m["precision_at_5"] - (0.2 + 0.2 + 0.0) / 3.0) < 1e-6
    assert abs(m["multi_gold_recall_at_5"] - 0.5) < 1e-6
    assert abs(m["mrr_at_5"] - (1.0 + 0.5 + 0.0) / 3.0) < 1e-6


def test_training_feature_blocks_alignment():
    # Verify fold 0 pre-computed feature blocks align perfectly
    mem = np.load(ROOT / "results/exp_final_retrieval/memory_ltr_probe/fold_0/train_augmented.f32.npy", mmap_mode="r")
    prof = np.load(ROOT / "results/exp_final_retrieval/profile_ltr_probe/fold_0/train_profile.f32.npy", mmap_mode="r")
    kern = np.load(ROOT / "results/exp_final_retrieval/kernel_ltr_probe/fold_0/train_kernel.f32.npy", mmap_mode="r")
    
    assert mem.shape[0] == prof.shape[0] == kern.shape[0] == 1312217
    assert mem.shape[1] == 86
    assert prof.shape[1] == 12
    assert kern.shape[1] == 21
    assert mem.shape[1] + prof.shape[1] + kern.shape[1] == 119
