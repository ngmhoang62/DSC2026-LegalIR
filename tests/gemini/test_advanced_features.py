"""Unit tests for Gemini AdvancedFeatureExtractor (14D)."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.advanced_features import AdvancedFeatureExtractor, strip_accents


def test_strip_accents():
    text = "Nghị định sửa đổi, bổ sung Luật Đất đai năm 2024"
    norm = strip_accents(text).lower()
    assert "nghi dinh sua doi" in norm
    assert "luat dat dai" in norm


def test_advanced_feature_extractor_init():
    extractor = AdvancedFeatureExtractor()
    assert len(extractor.doc_labels) > 0
    assert len(extractor.doc_is_amendment) > 0
    assert len(extractor.doc_is_cong_van) > 0


def test_extract_single():
    extractor = AdvancedFeatureExtractor()
    # Test query
    q = "Mức xử phạt theo Nghị định 100 2019 là bao nhiêu?"
    # Pick a document
    doc_id = next(iter(extractor.doc_labels.keys()))
    source_ranks = {"e5": 1, "lal": 2, "bm25": 5, "trigram": 10, "jina": 3}
    top_consensus = [doc_id, "some_other_doc"]
    
    feats = extractor.extract_single(q, doc_id, source_ranks, top_consensus)
    assert len(feats) == 14
    assert all(isinstance(x, (float, int)) for x in feats)
    assert np.isfinite(feats).all()
