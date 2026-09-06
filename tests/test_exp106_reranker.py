import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"D:\Study\DSC2026\LegalIR\src")))
from exp106_cross_encoder_reranker import (
    LegalPairTrainDataset,
    evaluate_rankings,
    extract_intact_article_capsule,
    zscore,
)


def test_intact_article_capsule():
    mock_meta = {
        "101": {
            "official_title": {
                "status": "VERIFIED",
                "document_type": "NGHỊ ĐỊNH",
                "number": "100/2019/NĐ-CP",
                "display_text": "QUY ĐỊNH XỬ PHẠT VI PHẠM HÀNH CHÍNH TRONG LĨNH VỰC GIAO THÔNG",
            }
        }
    }
    # Test capsule creation logic
    cap = extract_intact_article_capsule("101", "mức phạt nồng độ cồn", mock_meta)
    assert "[NGHỊ ĐỊNH 100/2019/NĐ-CP]: QUY ĐỊNH XỬ PHẠT" in cap
    print("test_intact_article_capsule: PASSED")


def test_multi_gold_dataset_expansion():
    # Test multi-gold expansion: query with 2 gold docs
    capsules = {
        "q1": [
            {"doc_id": "g1", "capsule_text": "text_g1", "question": "q1_text"},
            {"doc_id": "n1", "capsule_text": "text_n1", "question": "q1_text"},
            {"doc_id": "g2", "capsule_text": "text_g2", "question": "q1_text"},
            {"doc_id": "n2", "capsule_text": "text_n2", "question": "q1_text"},
            {"doc_id": "n3", "capsule_text": "text_n3", "question": "q1_text"},
        ]
    }
    answers = {"q1": {"g1", "g2"}}
    dataset = LegalPairTrainDataset(["q1"], capsules, answers)
    
    # Must have 2 positive pairs (g1, g2) and 2 true negative pairs (n1, n2) -> 4 pairs total
    assert len(dataset) == 4, f"Expected 4 pairs, got {len(dataset)}"
    pos_pairs = [p for p in dataset.pairs if p[2] == 1.0]
    neg_pairs = [p for p in dataset.pairs if p[2] == 0.0]
    assert len(pos_pairs) == 2
    assert len(neg_pairs) == 2
    # Ensure neither g1 nor g2 is in neg_pairs
    assert "text_g1" not in [p[1] for p in neg_pairs]
    assert "text_g2" not in [p[1] for p in neg_pairs]
    print("test_multi_gold_dataset_expansion: PASSED")


def test_zscore_and_metrics():
    raw_scores = [10.0, 5.0, 0.0, -5.0]
    z = zscore(raw_scores)
    assert abs(float(np.mean(z))) < 1e-4
    assert abs(float(np.std(z)) - 1.0) < 1e-4
    
    pred = {"q1": ["g1", "n1", "n2", "n3", "n4", "g2"]}
    answers = {"q1": {"g1", "g2"}}
    metrics = evaluate_rankings(pred, answers, ks=(1, 5, 6))
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@5"] == 0.5
    assert metrics["recall@6"] == 1.0
    assert metrics["mrr@5"] == 1.0
    print("test_zscore_and_metrics: PASSED")


if __name__ == "__main__":
    test_intact_article_capsule()
    test_multi_gold_dataset_expansion()
    test_zscore_and_metrics()
    print("\nALL EXP-106 UNIT TESTS PASSED SUCCESSFULLY!")
