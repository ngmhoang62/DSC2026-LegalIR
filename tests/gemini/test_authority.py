"""Unit tests for Statutory Authority and Legal Hierarchy Features."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.authority import AuthorityExtractor, HIERARCHY_LEVELS, STOPWORDS


def test_hierarchy_mapping():
    assert HIERARCHY_LEVELS["luat"] == 1
    assert HIERARCHY_LEVELS["bo luat"] == 1
    assert HIERARCHY_LEVELS["nghi dinh"] == 3
    assert HIERARCHY_LEVELS["thong tu"] == 5
    assert HIERARCHY_LEVELS["quyet dinh"] == 4
    assert HIERARCHY_LEVELS["qcvn"] == 7
    assert HIERARCHY_LEVELS["tcvn"] == 7


def test_authority_extractor_live():
    auth = AuthorityExtractor()
    assert len(auth.doc_labels) > 8000
    assert len(auth.doc_hierarchy) > 8000
    assert len(auth.doc_words) > 8000

    # Test single extraction
    # Q: asks for decree, doc is law -> mismatch
    q_decree = "Theo quy định tại Nghị định hướng dẫn về đất đai"
    # Find a law doc and a decree doc
    law_doc = next(d for d, lvl in auth.doc_hierarchy.items() if lvl == 1)
    decree_doc = next(d for d, lvl in auth.doc_hierarchy.items() if lvl == 3)

    f_law = auth.extract_single(q_decree, law_doc)
    f_decree = auth.extract_single(q_decree, decree_doc)

    assert len(f_law) == 12
    assert len(f_decree) == 12

    # Law should get mismatch penalty when query asks for decree
    assert f_law[10] == 1.0  # authority_mismatch_penalty
    assert f_decree[10] == 0.0

    # Decree should get exact match
    assert f_decree[9] == 1.0  # authority_exact_match
    assert f_law[9] == 0.0


def test_block_extraction():
    auth = AuthorityExtractor()
    questions = {
        "q1": "Thủ tục thu hồi đất theo Nghị định 43/2014",
        "q2": "Mức xử phạt vi phạm giao thông theo Luật",
    }
    docs_list = [
        list(auth.doc_hierarchy.keys())[:10],
        list(auth.doc_hierarchy.keys())[10:25],
    ]
    block = auth.extract_block(questions, ["q1", "q2"], docs_list)
    assert block.shape == (25, 12)
    assert block.dtype == np.float32
    assert np.isfinite(block).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
