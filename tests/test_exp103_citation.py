import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(r"D:\Study\DSC2026\LegalIR\src")))
from exp103_preamble_citation import (
    normalize_code,
    normalize_law_title,
    expand_single_ranking,
    SELF_CODE_REGEX,
    CITED_CODE_REGEX,
)


def test_normalize_code():
    assert normalize_code("  02/2009/tt-btp ") == "02/2009/TT-BTP"
    assert normalize_code("204/2004/NĐ-CP") == "204/2004/NĐ-CP"
    print("test_normalize_code: PASSED")


def test_regex_extraction():
    header = """
    BỘ TƯ PHÁP
    Số: 02/2009/TT-BTP
    Hà Nội, ngày 17 tháng 9 năm 2009
    
    Căn cứ Nghị định số 204/2004/NĐ-CP ngày 14 tháng 12 năm 2004;
    Căn cứ Nghị định số 67/2005/NĐ-CP ngày 19 tháng 5 năm 2005;
    """
    
    m = SELF_CODE_REGEX.search(header)
    assert m is not None, "Must extract self code"
    assert normalize_code(m.group(1)) == "02/2009/TT-BTP"
    
    cited = CITED_CODE_REGEX.findall(header)
    assert "204/2004/NĐ-CP" in [normalize_code(c) for c in cited]
    assert "67/2005/NĐ-CP" in [normalize_code(c) for c in cited]
    print("test_regex_extraction: PASSED")


def test_expand_single_ranking():
    # Synthetic citation graph: doc_A cites doc_P1 and doc_P2
    citation_graph = {
        "doc_A": ["doc_P1", "doc_P2"],
        "doc_B": ["doc_P3"],
    }
    
    # Original ranking where doc_P1 is deep at rank 60
    raw_ranking = ["doc_A", "doc_B", "doc_C"] + [f"doc_{i}" for i in range(10, 60)] + ["doc_P1"]
    
    expanded = expand_single_ranking(
        raw_ranking=raw_ranking,
        citation_graph=citation_graph,
        seed_k=1,
        max_citations_per_seed=2,
        max_total_citations=3,
        max_output=50,
    )
    
    # doc_A is rank 1, doc_P1 and doc_P2 should be injected immediately at rank 2 and 3!
    assert expanded[0] == "doc_A"
    assert expanded[1] == "doc_P1"
    assert expanded[2] == "doc_P2"
    assert expanded[3] == "doc_B"
    assert len(expanded) <= 50
    assert len(set(expanded)) == len(expanded), "No duplicates allowed in expanded ranking"
    print("test_expand_single_ranking: PASSED")


if __name__ == "__main__":
    test_normalize_code()
    test_regex_extraction()
    test_expand_single_ranking()
    print("\nALL UNIT TESTS FOR EXP-103 PASSED SUCCESSFULLY!")
