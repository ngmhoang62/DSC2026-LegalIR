from exp_final.relations import KinshipPolicy, amendment_relation, apply_kinship, instrument_keys


def test_instrument_keys_and_exact_amendment_relation():
    anchor = "Nghi-dinh-43-2014-ND-CP-huong-dan-Luat-dat-dai"
    candidate = "Nghi-dinh-148-2020-ND-CP-sua-doi-Nghi-dinh-43-2014-ND-CP"
    assert ("nghi dinh", "43", "2014") in instrument_keys(anchor)
    assert amendment_relation(candidate, anchor) == (True, "instrument")


def test_non_amendment_never_matches():
    assert amendment_relation("Nghi dinh 43 2014 ND CP", "Nghi dinh 43 2014 ND CP") == (False, "none")


def test_guarded_promotion_only_changes_fifth_slot():
    ranking = ["a", "b", "c", "d", "e", "f", "g"]
    titles = {"a":"Nghi dinh 43 2014 ND CP", "g":"Nghi dinh 148 2020 ND CP sua doi Nghi dinh 43 2014 ND CP"}
    changed, event = apply_kinship(ranking, titles, KinshipPolicy(top_k=1, candidate_max=7))
    assert changed[:5] == ["a", "b", "c", "d", "g"]
    assert changed[5:7] == ["e", "f"]
    assert event["displaced"] == "e" and event["from_rank"] == 7


def test_rank5_amendment_guard_blocks_replacement():
    ranking = ["a", "b", "c", "d", "e", "g"]
    titles = {"a":"Nghi dinh 43 2014 ND CP", "e":"Thong tu 1 2020 sua doi abc", "g":"Nghi dinh 148 2020 sua doi Nghi dinh 43 2014"}
    changed, event = apply_kinship(ranking, titles, KinshipPolicy())
    assert changed == ranking and event is None
