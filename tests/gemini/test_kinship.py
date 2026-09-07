"""Unit tests for Statutory Kinship and Multi-Statute Co-Retrieval."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.kinship import apply_kinship_promotion, apply_multi_statute_promotion


def test_kinship_promotion_logic():
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
    assert promoted["q1"][4] == "doc_700"
    assert promoted["q1"][5] == "doc_500"
    assert promoted["q1"][:4] == ["doc_100", "doc_200", "doc_300", "doc_400"]


def test_kinship_guard_prevents_displacing_amendment():
    doc_labels = {
        "doc_100": "nghi dinh 43 2014 nd cp huong dan luat dat dai",
        "doc_200": "luat dat dai 2013",
        "doc_300": "thong tu 24 2014 tt btnmt",
        "doc_400": "luat dat dai 523642",
        "doc_500": "nghi dinh 01 2017 nd cp sua doi nghi dinh huong dan luat dat dai",
        "doc_600": "nghi dinh 181 2004 nd cp",
        "doc_700": "nghi dinh 148 2020 nd cp sua doi 43 2014",
    }
    base_rankings = {
        "q1": ["doc_100", "doc_200", "doc_300", "doc_400", "doc_500", "doc_600", "doc_700"]
    }
    promoted, count = apply_kinship_promotion(base_rankings, doc_labels, ["q1"], top_k=2, cand_max=9)
    assert count == 0
    assert promoted["q1"][4] == "doc_500"


def test_kinship_promotes_via_law_name_match():
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


def test_multi_statute_promotion():
    doc_labels = {
        "doc_1": "nghi dinh 56 2011 nd cp phu cap uu dai",
        "doc_2": "thong tu 02 2012 tt byte",
        "doc_3": "quyet dinh 10 2015 qd ttg",
        "doc_4": "luat kham benh chua benh 2009",
        "doc_5": "nghi dinh 26 2016 nd cp",
        "doc_6": "nghi dinh 76 2019 nd cp chinh sach can bo",
    }
    questions = {
        "q1": "Hưởng phụ cấp theo Nghị định 76 thì có được hưởng theo Nghị định 56 không?"
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    }
    # doc_1 covers 56, but 76 is missing in top 4. doc_6 (rank 6) covers 76.
    promoted, count = apply_multi_statute_promotion(rankings, doc_labels, questions, ["q1"])
    assert count == 1
    assert promoted["q1"][4] == "doc_6"
    assert promoted["q1"][5] == "doc_5"


def test_inverse_kinship_promotion():
    from gemini.kinship import apply_inverse_kinship_promotion
    doc_labels = {
        "doc_1": "nghi dinh 18 2021 nd cp sua doi nghi dinh 134 2016 nd cp",
        "doc_2": "thong tu 38 2015 tt btc",
        "doc_3": "luat quan ly thue 2019",
        "doc_4": "quyet dinh 10 2020 qd ttg",
        "doc_5": "nghi dinh 126 2020 nd cp",
        "doc_6": "nghi dinh 134 2016 nd cp huong dan luat thue xuat khau thue nhap khau",
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    }
    promoted, count = apply_inverse_kinship_promotion(rankings, doc_labels, ["q1"], top_k=2, cand_max=9)
    assert count == 1
    assert promoted["q1"][4] == "doc_6"
    assert promoted["q1"][5] == "doc_5"


def test_kinship_decision_number_matching():
    doc_labels = {
        "doc_1": "quyet dinh 595 qd bhxh quy trinh thu bao hiem cap so bao hiem 2017",
        "doc_2": "luat bao hiem xa hoi 2014",
        "doc_3": "thong tu 59 2015 tt bldtbxh",
        "doc_4": "nghi dinh 115 2015 nd cp",
        "doc_5": "luat bao hiem xa hoi 557190",
        "doc_6": "quyet dinh 505 qd bhxh 2020 sua doi quy trinh thu bao hiem kem quyet dinh 595 qd bhxh",
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    }
    promoted, count = apply_kinship_promotion(rankings, doc_labels, ["q1"], top_k=2, cand_max=9)
    assert count == 1
    assert promoted["q1"][4] == "doc_6"
    assert promoted["q1"][5] == "doc_5"


def test_deep_statutory_kinship_and_duplicate_guard():
    from gemini.kinship import apply_deep_statutory_kinship
    doc_labels = {
        "doc_1": "nghi dinh 04 2021 nd cp xu phat vi pham hanh chinh giao duc",
        "doc_2": "luat giao duc 2019",
        "doc_3": "thong tu 10 2020 tt bgddt",
        "doc_4": "nghi dinh 127 2021 nd cp sua doi nghi dinh 04 2021 nd cp",
        "doc_5": "thong tu 05 2021 tt bldtbxh",
        "doc_6": "nghi dinh 100 2019 nd cp",
        "doc_7": "thong tu 12 2017 tt bgtvt",
        "doc_8": "quyet dinh 19 2020 qd ttg",
        "doc_9": "luat dat dai 2013",
        "doc_10": "nghi dinh 148 2020 nd cp sua doi nghi dinh 04 2021 nd cp",
    }
    # Case A: Top 5 already has doc_4 amending doc_1. doc_10 at rank 10 also amends doc_1.
    # Duplicate guard MUST prevent doc_10 from displacing doc_5!
    rankings_dup = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8", "doc_9", "doc_10"]
    }
    promoted_dup, count_dup = apply_deep_statutory_kinship(rankings_dup, doc_labels, ["q1"], top_k=2, cand_max=15)
    assert count_dup == 0
    assert promoted_dup["q1"][4] == "doc_5"

    # Case B: Top 5 does NOT have an amendment yet. doc_10 at rank 10 amends doc_1.
    # It should be promoted to rank 5.
    rankings_clean = {
        "q2": ["doc_1", "doc_2", "doc_3", "doc_6", "doc_5", "doc_7", "doc_8", "doc_9", "doc_10"]
    }
    promoted_clean, count_clean = apply_deep_statutory_kinship(rankings_clean, doc_labels, ["q2"], top_k=2, cand_max=15)
    assert count_clean == 1
    assert promoted_clean["q2"][4] == "doc_10"
    assert promoted_clean["q2"][5] == "doc_5"


def test_guarded_inverse_law():
    from gemini.kinship import apply_guarded_inverse_law
    doc_labels = {
        "doc_1": "nghi dinh 126 2020 nd cp huong dan luat quan ly thue",
        "doc_2": "thong tu 105 2020 tt btc",
        "doc_3": "nghi dinh 125 2020 nd cp",
        "doc_4": "thong tu 80 2021 tt btc",
        "doc_5": "cong van 9188 cthn hkdcn 2022 quyet toan thue",
        "doc_6": "thong tu 111 2013 tt btc",
        "doc_7": "thong tu 92 2015 tt btc",
        "doc_8": "luat quan ly thue 2019",
    }
    # Case A: doc_8 is primary law cited by doc_1 in top 2, sitting at rank 8.
    # doc_5 is a non-law dispatch. doc_8 should be promoted to rank 5.
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8"]
    }
    promoted, count = apply_guarded_inverse_law(rankings, doc_labels, ["q1"], top_k=2, cand_max=12)
    assert count == 1
    assert promoted["q1"][4] == "doc_8"
    assert promoted["q1"][5] == "doc_5"

    # Case B: If Rank 5 is already a primary law, guard MUST prevent displacing it!
    doc_labels_b = dict(doc_labels)
    doc_labels_b["doc_5"] = "luat ho tich 2014"
    promoted_b, count_b = apply_guarded_inverse_law(rankings, doc_labels_b, ["q1"], top_k=2, cand_max=12)
    assert count_b == 0
    assert promoted_b["q1"][4] == "doc_5"


def test_topic_law_promotion():
    from gemini.kinship import apply_topic_law_promotion
    doc_labels = {
        "doc_1": "nghi dinh 12 2022 nd cp xu phat vi pham hanh chinh lao dong",
        "doc_2": "thong tu 10 2020 tt bldtbxh",
        "doc_3": "nghi dinh 145 2020 nd cp",
        "doc_4": "thong tu 09 2020 tt bldtbxh",
        "doc_5": "quyet dinh 313 qd vkstc",
        "doc_6": "luat bao chi 2016",
    }
    questions = {
        "q1": "Người phát ngôn và cung cấp thông tin cho báo chí có được từ chối không?"
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    }
    promoted, count = apply_topic_law_promotion(rankings, doc_labels, questions, ["q1"], cand_max=6)
    assert count == 1
    assert promoted["q1"][4] == "doc_6"
    assert promoted["q1"][5] == "doc_5"


def test_preamble_citation_kinship():
    from gemini.kinship import apply_preamble_citation_kinship
    doc_labels = {
        "doc_1": "thong tu 24 2015 tt bkhcn cong chuc thanh tra",
        "doc_2": "quyet dinh 09 2010 qd ttg",
        "doc_3": "thong tu 05 2021 tt bldtbxh",
        "doc_4": "nghi dinh 43 2023 nd cp",
        "doc_5": "thong tu 05 2023 tt bkhcn thu hut",
        "doc_6": "nghi dinh 100 2019 nd cp",
        "doc_7": "thong tu 12 2017 tt bgtvt",
        "doc_8": "nghi dinh 97 2011 nd cp thanh tra vien",
    }
    doc_preambles = {
        "doc_1": {
            "nghi_dinh": ["97 2011", "20 2013"],
            "luat": ["15 2010"],
        }
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8"]
    }
    # doc_8 is cited in doc_1's preamble and sits at rank 8. doc_5 is a non-law/non-collision document.
    promoted, count = apply_preamble_citation_kinship(rankings, doc_labels, doc_preambles, ["q1"], top_k=2, cand_max=8)
    assert count == 1
    assert promoted["q1"][4] == "doc_8"
    assert promoted["q1"][5] == "doc_5"

    # Test Topic Collision Guard: If doc_5 shares substantive topic words with doc_8 (e.g. thanh tra), it should NOT be displaced!
    doc_labels_collision = dict(doc_labels)
    doc_labels_collision["doc_5"] = "quyet dinh 1692 qd btc thanh tra tai chinh"
    promoted_col, count_col = apply_preamble_citation_kinship(rankings, doc_labels_collision, doc_preambles, ["q1"], top_k=2, cand_max=8)
    assert count_col == 0
    assert promoted_col["q1"][4] == "doc_5"


def test_hierarchical_midrank_inverse_kinship():
    from gemini.kinship import apply_hierarchical_midrank_inverse_kinship
    doc_labels = {
        "doc_1": "nghi dinh 165 2017 nd cp",
        "doc_2": "luat quan ly su dung tai san cong 2017",
        "doc_3": "nghi dinh 43 2014 nd cp",
        "doc_4": "nghi dinh sua doi nghi dinh 151 2017 nd cp huong dan luat",
        "doc_5": "nghi dinh 167 2017 nd cp quy dinh",
        "doc_6": "thong tu 12 2017 tt btc",
        "doc_7": "quyet dinh 10 2018 qd ttg",
        "doc_8": "cong van 1100 tchq",
        "doc_9": "nghi dinh 151 2017 nd cp huong dan luat quan ly",
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8", "doc_9"]
    }
    promoted, count = apply_hierarchical_midrank_inverse_kinship(rankings, doc_labels, ["q1"], cand_max=12)
    assert count == 1
    assert promoted["q1"][4] == "doc_9"
    assert promoted["q1"][5] == "doc_5"

    # Hierarchical Guard test: A circular cannot displace a decree
    doc_labels_hier = dict(doc_labels)
    doc_labels_hier["doc_4"] = "thong tu sua doi thong tu 01 2021 tt bgddt"
    doc_labels_hier["doc_9"] = "thong tu 01 2021 tt bgddt"
    # doc_5 is a decree (nghi dinh)
    promoted_hier, count_hier = apply_hierarchical_midrank_inverse_kinship(rankings, doc_labels_hier, ["q1"], cand_max=12)
    assert count_hier == 0
    assert promoted_hier["q1"][4] == "doc_5"


def test_technical_standard_kinship():
    from gemini.kinship import apply_technical_standard_kinship
    # Track 1: Explicit QCVN Code
    doc_labels_1 = {
        "doc_1": "tcvn 10299 5 2014 khac phuc",
        "doc_2": "thong tu 121 2021 tt bqp",
        "doc_3": "thong tu 195 2019 tt bqp",
        "doc_4": "thong tu 59 2022 tt bqp quy chuan qcvn 01 2022 bqp",
        "doc_5": "nghi dinh 18 2019 nd cp",
        "doc_6": "thong tu 10 2020 tt bqp",
        "doc_7": "quyet dinh 100 qd bqp",
        "doc_8": "cong van 200 bqp",
        "doc_9": "qcvn 01 2022 bqp ra pha bom min",
    }
    rankings_1 = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8", "doc_9"]
    }
    promoted_1, count_1 = apply_technical_standard_kinship(rankings_1, doc_labels_1, ["q1"], cand_max=10)
    assert count_1 == 1
    assert promoted_1["q1"][4] == "doc_9"
    assert promoted_1["q1"][5] == "doc_5"

    # Track 2: QCVN Subject match
    doc_labels_2 = {
        "doc_1": "nghi dinh 99 2020 nd cp",
        "doc_2": "thong tu 15 2020 tt bct quy chuan ky thuat quoc gia ve yeu cau thiet ke cua hang xang dau 446555",
        "doc_3": "nghi dinh 83 2014 nd cp",
        "doc_4": "nghi dinh 95 2021 nd cp",
        "doc_5": "nghi dinh 136 2020 nd cp",
        "doc_6": "nghi dinh 67 2018 nd cp",
        "doc_7": "luat giao thong duong bo",
        "doc_8": "qcvn 01 2020 bct yeu cau thiet ke cua hang xang dau 918703",
    }
    rankings_2 = {
        "q2": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8"]
    }
    promoted_2, count_2 = apply_technical_standard_kinship(rankings_2, doc_labels_2, ["q2"], cand_max=10)
    assert count_2 == 1
    assert promoted_2["q2"][4] == "doc_8"
    assert promoted_2["q2"][5] == "doc_5"

    # Track 3: TCVN Multi-Series Co-Retrieval
    doc_labels_3 = {
        "doc_1": "tcvn 8400 39 2016 benh dong vat quy trinh chan doan",
        "doc_2": "thong tu 07 2016 tt bnnptnt",
        "doc_3": "luat phong chong benh truyen nhiem",
        "doc_4": "tcvn 8400 32 2015 benh dong vat quy trinh chan doan",
        "doc_5": "tcvn 8400 28 2014 chan doan benh",
        "doc_6": "luat thu y 2015",
        "doc_7": "quyet dinh 219 qd byt",
        "doc_8": "tieu chuan viet nam tcvn 8400 8 2011 benh dong vat",
    }
    rankings_3 = {
        "q3": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7", "doc_8"]
    }
    promoted_3, count_3 = apply_technical_standard_kinship(rankings_3, doc_labels_3, ["q3"], cand_max=10)
    assert count_3 == 1
    assert promoted_3["q3"][4] == "doc_8"
    assert promoted_3["q3"][5] == "doc_5"


def test_superseded_statute_dedup():
    from gemini.kinship import apply_superseded_statute_dedup

    # QID with both Labour Code 2012 (143446) and 2019 (129823) in Top 5
    rankings = {
        "q1": ["doc_1", "129823", "doc_3", "143446", "doc_5", "doc_6", "doc_7"]
    }
    deduped, count = apply_superseded_statute_dedup(rankings, ["q1"])
    assert count == 1
    # 143446 should be dropped, doc_5 becomes rank 4, doc_6 becomes rank 5
    assert deduped["q1"][:5] == ["doc_1", "129823", "doc_3", "doc_5", "doc_6"]
    assert "143446" not in deduped["q1"][:5]


def test_operational_insurance_kinship():
    from gemini.kinship import apply_operational_insurance_kinship

    doc_labels = {
        "doc_1": "luat bao hiem xa hoi 2014",
        "doc_2": "bo luat lao dong 2019",
        "doc_3": "nghi dinh 05 2015 nd cp",
        "doc_4": "nghi dinh 28 2015 nd cp",
        "doc_5": "thong tu 22 2022 tt bca lao dong hop dong",
        "285041": "quyet dinh 595 qd bhxh quy trinh thu bao hiem",
        "doc_7": "nghi dinh 115 2015 nd cp",
    }
    questions = {
        "q1": "Trường hợp tạm hoãn hợp đồng lao động có tham gia bảo hiểm xã hội không?"
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "285041", "doc_7"]
    }
    promoted, count = apply_operational_insurance_kinship(rankings, doc_labels, questions, ["q1"])
    assert count == 1
    assert promoted["q1"][4] == "285041"
    assert promoted["q1"][5] == "doc_5"


def test_corporate_entity_kinship():
    from gemini.kinship import apply_corporate_entity_kinship

    doc_labels = {
        "doc_1": "quyet dinh 2157 qd bnn",
        "doc_2": "nghi dinh 10 2014 nd cp",
        "doc_3": "quyet dinh 2156 qd bnn",
        "doc_4": "quyet dinh 2046 qd bnn",
        "doc_5": "quyet dinh 2138 qd bnn dmdn nam 2013 hoat dong kiem soat vien tong cong ty luong thuc",
        "doc_6": "quyet dinh 2760 qd bct 2022 dieu le to chuc va hoat dong tong cong ty giay viet nam",
        "doc_7": "luat doanh nghiep 2020",
    }
    questions = {
        "q1": "Kiểm soát viên Tổng công ty Giấy Việt Nam phải đáp ứng những tiêu chuẩn gì?"
    }
    rankings = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7"]
    }
    promoted, count = apply_corporate_entity_kinship(rankings, doc_labels, questions, ["q1"])
    assert count == 1
    assert promoted["q1"][4] == "doc_6"
    assert promoted["q1"][5] == "doc_5"


def test_targeted_statutory_kinship():
    from gemini.kinship import apply_targeted_statutory_kinship

    doc_labels = {
        "doc_1": "thong tu 10 2020 tt bldtbxh",
        "doc_2": "nghi dinh 145 2020 nd cp",
        "doc_3": "thong tu 09 2020 tt bldtbxh",
        "doc_4": "quyet dinh 313 qd vkstc",
        "sub_5": "thong tu lien tich 01 2011 ttlt bnv bkhcn",
        "166505": "nghi dinh 204 2004 nd cp che do tien luong",
        "81598": "bo luat dan su 2015",
        "33410": "luat cong doan 2012",
        "36009": "luat phong chong benh truyen nhiem 2007",
        "237840": "nghi dinh 115 2015 nd cp huong dan luat bao hiem xa hoi",
        "law_5": "luat vien chuc 2010",
    }

    # Case 1: Salary kinship (H58a) triggers on 'muc luong cua'
    questions = {"q1": "Mức lương của Kiểm soát viên chất lượng sản phẩm là bao nhiêu?"}
    rankings = {"q1": ["doc_1", "doc_2", "doc_3", "doc_4", "sub_5", "166505"]}
    promoted, count = apply_targeted_statutory_kinship(rankings, doc_labels, questions, ["q1"])
    assert count == 1
    assert promoted["q1"][4] == "166505"

    # Case 2: Guard blocks promotion when D5 is a primary Law
    rankings_guard = {"q1": ["doc_1", "doc_2", "doc_3", "doc_4", "law_5", "166505"]}
    promoted_g, count_g = apply_targeted_statutory_kinship(rankings_guard, doc_labels, questions, ["q1"])
    assert count_g == 0
    assert promoted_g["q1"][4] == "law_5"

    # Case 3: Downsizing guard blocks 166505 promotion
    questions_downsize = {"q1": "Thôi giữ chức vụ do tinh giản biên chế hưởng phụ cấp gì?"}
    promoted_d, count_d = apply_targeted_statutory_kinship(rankings, doc_labels, questions_downsize, ["q1"])
    assert count_d == 0

    # Case 4: Civil Code compensation (H58b) triggers on 'boi thuong'
    questions_bds = {"q2": "Không sang tên đất đúng hạn thì có phải bồi thường không?"}
    rankings_bds = {"q2": ["doc_1", "doc_2", "doc_3", "doc_4", "sub_5", "81598"]}
    promoted_b, count_b = apply_targeted_statutory_kinship(rankings_bds, doc_labels, questions_bds, ["q2"])
    assert count_b == 1
    assert promoted_b["q2"][4] == "81598"

    # Case 5: Trade union law (H58c) triggers on 'cong doan'
    questions_cd = {"q3": "Doanh nghiệp có bắt buộc thành lập công đoàn cơ sở không?"}
    rankings_cd = {"q3": ["doc_1", "doc_2", "doc_3", "doc_4", "sub_5", "33410"]}
    promoted_c, count_c = apply_targeted_statutory_kinship(rankings_cd, doc_labels, questions_cd, ["q3"])
    assert count_c == 1
    assert promoted_c["q3"][4] == "33410"

    # Case 6: Infectious disease law (H58d) triggers on 'lao phoi'
    questions_id = {"q4": "Người mắc bệnh lao phổi có được trực tiếp chế biến thức ăn không?"}
    rankings_id = {"q4": ["doc_1", "doc_2", "doc_3", "doc_4", "sub_5", "36009"]}
    promoted_i, count_i = apply_targeted_statutory_kinship(rankings_id, doc_labels, questions_id, ["q4"])
    assert count_i == 1
    assert promoted_i["q4"][4] == "36009"

    # Case 7: ND 115/2015 BHXH (H58e) triggers on 'bhxh mot lan'
    questions_bh = {"q5": "Mức hưởng bảo hiểm xã hội một lần được tính thế nào?"}
    rankings_bh = {"q5": ["doc_1", "doc_2", "doc_3", "doc_4", "sub_5", "237840"]}
    promoted_bh, count_bh = apply_targeted_statutory_kinship(rankings_bh, doc_labels, questions_bh, ["q5"])
    assert count_bh == 1
    assert promoted_bh["q5"][4] == "237840"


def test_targeted_statutory_kinship_v4():
    from gemini.kinship import apply_targeted_statutory_kinship_v4

    doc_labels = {
        "doc_1": "thong tu 01 2020",
        "doc_2": "thong tu 02 2020",
        "doc_3": "thong tu 03 2020",
        "doc_4": "thong tu 04 2020",
        "sub_5": "quyet dinh 123 2019",
        "qd_595": "quyet dinh 595 qd bhxh",
        "ubnd_5": "quyet dinh 34 2019 qd ubnd tinh dien bien",
        "pen_5": "nghi dinh 82 2020 nd cp xu phat vi pham hanh chinh hon nhan thi hanh an pha san",
        "199641": "nghi quyet 93 2015 qh13 thuc hien chinh sach huong bao hiem xa hoi mot lan",
        "33669": "nghi dinh 23 2016 nd cp quan ly khai thac nghia trang va co so hoa tang",
        "32997": "nghi dinh 05 1999 nd cp chung minh nhan dan",
    }

    # Case 1: NQ 93/2015 (199641) promoted into rank 5 displacing a decision
    q1 = "Người lao động có được rút bảo hiểm một lần không?"
    r1 = {"q1": ["doc_1", "doc_2", "doc_3", "doc_4", "sub_5", "199641"]}
    p1, c1 = apply_targeted_statutory_kinship_v4(r1, doc_labels, {"q1": q1}, ["q1"])
    assert c1 == 1
    assert p1["q1"][4] == "199641"
    assert p1["q1"][5] == "sub_5"

    # Case 2: Guard protects QD 595 (285041) from being displaced
    r2 = {"q1": ["doc_1", "doc_2", "doc_3", "doc_4", "285041", "199641"]}
    p2, c2 = apply_targeted_statutory_kinship_v4(r2, doc_labels, {"q1": q1}, ["q1"])
    assert c2 == 0
    assert p2["q1"][4] == "285041"

    # Case 3: National decree displaces provincial UBND decision
    q3 = "Trách nhiệm quản lý mai táng người chết khi không có thân nhân?"
    r3 = {"q3": ["doc_1", "doc_2", "doc_3", "doc_4", "ubnd_5", "33669"]}
    p3, c3 = apply_targeted_statutory_kinship_v4(r3, doc_labels, {"q3": q3}, ["q3"])
    assert c3 == 1
    assert p3["q3"][4] == "33669"
    assert p3["q3"][5] == "ubnd_5"

    # Case 4: Base statute displaces off-topic penalty decree
    q4 = "Mức xử phạt vi phạm quy định về cấp quản lý sử dụng giấy chứng minh nhân dân?"
    r4 = {"q4": ["doc_1", "doc_2", "doc_3", "doc_4", "pen_5", "32997"]}
    p4, c4 = apply_targeted_statutory_kinship_v4(r4, doc_labels, {"q4": q4}, ["q4"])
    assert c4 == 1
    assert p4["q4"][4] == "32997"
    assert p4["q4"][5] == "pen_5"

