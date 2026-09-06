"""Unit tests for Statutory Kinship Co-Retrieval."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gemini"))

from exp_statutory_kinship import apply_kinship_promotion


def test_kinship_promotion_logic():
    # Synthetic scenario:
    # Top 2: doc_100 (original decree: 43 2014 nd cp)
    # Rank 5: doc_500 (unrelated: 151 2017 nd cp)
    # Rank 7: doc_700 (amending decree: 148 2020 nd cp sua doi 43 2014)
    doc_labels = {
        "doc_100": "nghi dinh 43 2014 nd cp huong dan luat dat dai",
        "doc_200": "luat dat dai 2013",
        "doc_300": "thong tu 24 2014 tt btnmt",
        "doc_400": "luat dat dai 523642",
        "doc_500": "nghi dinh 151 2017 nd cp quan ly tai san cong",
        "doc_600": "nghi dinh 181 2004 nd cp",
        "doc_700": "nghi dinh 148 2020 nd cp sua doi mot so nghi dinh huong dan 43 2014",
    }
    
    base_rankings = {
        "q1": ["doc_100", "doc_200", "doc_300", "doc_400", "doc_500", "doc_600", "doc_700"]
    }
    
    promoted, count = apply_kinship_promotion(base_rankings, doc_labels, ["q1"], top_k=2, cand_max=9)
    assert count == 1
    # doc_700 should be promoted to rank 5 (index 4)
    assert promoted["q1"][4] == "doc_700"
    assert promoted["q1"][5] == "doc_500"
    assert promoted["q1"][:4] == ["doc_100", "doc_200", "doc_300", "doc_400"]


def test_kinship_guard_prevents_displacing_amendment():
    # If doc_500 is ALREADY an amendment, it should NOT be displaced
    doc_labels = {
        "doc_100": "nghi dinh 43 2014 nd cp huong dan luat dat dai",
        "doc_200": "luat dat dai 2013",
        "doc_300": "thong tu 24 2014 tt btnmt",
        "doc_400": "luat dat dai 523642",
        "doc_500": "nghi dinh 01 2017 nd cp sua doi nghi dinh huong dan luat dat dai",  # already an amendment!
        "doc_600": "nghi dinh 181 2004 nd cp",
        "doc_700": "nghi dinh 148 2020 nd cp sua doi 43 2014",
    }
    
    base_rankings = {
        "q1": ["doc_100", "doc_200", "doc_300", "doc_400", "doc_500", "doc_600", "doc_700"]
    }
    
    promoted, count = apply_kinship_promotion(base_rankings, doc_labels, ["q1"], top_k=2, cand_max=9)
    # Should NOT promote because rank 5 is already guarded
    assert count == 0
    assert promoted["q1"][4] == "doc_500"


def test_kinship_promotes_via_law_name_match():
    # Candidate amends base law without mentioning specific decree number
    doc_labels = {
        "doc_100": "luat dat dai 2013 215836",
        "doc_200": "nghi dinh 43 2014 nd cp huong dan thi hanh luat dat dai",
        "doc_300": "thong tu 24 2014 tt btnmt",
        "doc_400": "thong tu 30 2014 tt btnmt",
        "doc_500": "quyet dinh 19 2020 qd ttg",
        "doc_600": "nghi dinh 148 2020 nd cp sua doi mot so nghi dinh huong dan luat dat dai",
    }
    base_rankings = {
        "q1": ["doc_100", "doc_200", "doc_300", "doc_400", "doc_500", "doc_600"]
    }
    promoted, count = apply_kinship_promotion(base_rankings, doc_labels, ["q1"], top_k=2, cand_max=9)
    assert count == 1
    assert promoted["q1"][4] == "doc_600"
    assert promoted["q1"][5] == "doc_500"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
