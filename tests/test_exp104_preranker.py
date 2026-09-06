import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(r"D:\Study\DSC2026\LegalIR\src")))
from exp104_preranker_benchmark import (
    FEATURE_COLUMNS,
    extract_28d_features,
    evaluate_rankings,
)


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 28, f"Feature columns must be exactly 28, got {len(FEATURE_COLUMNS)}"
    print("test_feature_columns_count: PASSED")


def test_extract_28d_features():
    doc_meta = {"token_length": 300, "doc_type": "law", "scope_nodes": 2, "parse_fallback": 0}
    feats = extract_28d_features(
        qid="1001",
        doc_id="77220",
        d_rank=1.0,
        d_score=0.0303,
        b_rank=5.0,
        b_score=0.0270,
        top1_d_score=0.0303,
        q_tokens=12,
        doc_meta=doc_meta,
        statutory_q=1.0,
    )
    assert len(feats) == 28, f"Extracted vector length must be 28, got {len(feats)}"
    assert feats[0] == 1.0  # dense_rank
    assert feats[3] == 5.0  # bm25_rank
    assert feats[24] == 1.0  # is_law
    assert feats[27] == 1.0  # statutory_q
    print("test_extract_28d_features: PASSED")


def test_evaluate_rankings():
    pred = {"q1": ["d1", "d2", "d3"], "q2": ["d4", "d5", "d6"]}
    answers = {"q1": {"d1"}, "q2": {"d6"}}
    metrics = evaluate_rankings(pred, answers, ks=(1, 3, 5))
    assert metrics["recall@1"] == 0.5  # q1 hit at rank 1, q2 hit at rank 3
    assert metrics["recall@3"] == 1.0  # both hit within top 3
    assert metrics["mrr@5"] == 0.5 * (1.0 + 1.0/3.0)
    print("test_evaluate_rankings: PASSED")


if __name__ == "__main__":
    test_feature_columns_count()
    test_extract_28d_features()
    test_evaluate_rankings()
    print("\nALL UNIT TESTS FOR EXP-104 PASSED SUCCESSFULLY!")
