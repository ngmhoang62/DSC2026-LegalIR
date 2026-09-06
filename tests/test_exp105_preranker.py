import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"D:\Study\DSC2026\LegalIR\src")))
from exp105_synergy_preranker import (
    FEATURE_COLUMNS,
    evaluate_rankings,
)


def test_36d_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 36, f"Feature columns must be exactly 36, got {len(FEATURE_COLUMNS)}"
    print("test_36d_feature_columns_count: PASSED")


def test_evaluate_rankings():
    pred = {"q1": ["d1", "d2", "d3"], "q2": ["d4", "d5", "d6"]}
    answers = {"q1": {"d1"}, "q2": {"d6"}}
    metrics = evaluate_rankings(pred, answers, ks=(1, 3, 5))
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@5"] == 0.5 * (1.0 + 1.0/3.0)
    print("test_evaluate_rankings: PASSED")


if __name__ == "__main__":
    test_36d_feature_columns_count()
    test_evaluate_rankings()
    print("\nALL UNIT TESTS FOR EXP-105 PASSED SUCCESSFULLY!")
