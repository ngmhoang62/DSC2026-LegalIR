"""Label-free statutory relationship features for EXP-final slate research."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def ascii_words(text: str) -> list[str]:
    value = unicodedata.normalize("NFD", str(text).lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    return re.findall(r"[a-z0-9]+", value)


def normalized_title(text: str) -> str:
    return " ".join(ascii_words(text))


def is_amendment(title: str) -> bool:
    value = normalized_title(title)
    return "sua doi" in value or "bo sung" in value


def authority(title: str) -> str:
    value = normalized_title(title)
    for name in ("hien phap", "bo luat", "luat", "phap lenh", "nghi quyet", "nghi dinh",
                 "quyet dinh", "thong tu", "cong van", "qcvn", "tcvn"):
        if value.startswith(name):
            return name
    return "other"


def instrument_keys(title: str) -> set[tuple[str, str, str]]:
    """Extract conservative type/number/year identifiers from slug-like titles."""
    value = normalized_title(title)
    kinds = "hien phap|bo luat|luat|phap lenh|nghi quyet|nghi dinh|quyet dinh|thong tu|cong van"
    keys = set()
    for match in re.finditer(rf"\b({kinds})\s+(\d{{1,4}})(?:\s+[a-z]{{1,8}})*?\s+((?:19|20)\d{{2}})\b", value):
        keys.add((match.group(1), str(int(match.group(2))), match.group(3)))
    return keys


def law_subjects(title: str) -> set[str]:
    value = normalized_title(title)
    subjects = set()
    stops = {"nam", "so", "sua", "doi", "bo", "sung", "mot", "so", "dieu", "cua", "va", "ve", "thi", "hanh"}
    for match in re.finditer(r"\b(?:bo luat|luat)\s+([a-z ]{6,80}?)(?=\s+(?:nam|so|sua doi|bo sung|\d{4})\b|$)", value):
        words = [word for word in match.group(1).split() if word not in stops]
        if len(words) >= 2:
            subjects.add(" ".join(words))
    return subjects


def amendment_relation(candidate_title: str, anchor_title: str, *, allow_subject=True) -> tuple[bool, str]:
    if not is_amendment(candidate_title):
        return False, "none"
    candidate_keys, anchor_keys = instrument_keys(candidate_title), instrument_keys(anchor_title)
    if candidate_keys & anchor_keys:
        return True, "instrument"
    # The amending title normally contains both its own key and the amended
    # instrument's key. Match any anchor key as normalized tokens too.
    candidate = normalized_title(candidate_title)
    for kind, number, year in anchor_keys:
        if f"{kind} {number}" in candidate and year in candidate:
            return True, "instrument"
    if allow_subject:
        candidate_subjects, anchor_subjects = law_subjects(candidate_title), law_subjects(anchor_title)
        if candidate_subjects & anchor_subjects:
            return True, "law_subject"
        for left in candidate_subjects:
            for right in anchor_subjects:
                if left in right or right in left:
                    return True, "law_subject"
    return False, "none"


@dataclass(frozen=True)
class KinshipPolicy:
    top_k: int = 2
    candidate_max: int = 9
    allow_subject: bool = True
    guard_rank5_amendment: bool = True


def apply_kinship(ranking, titles, policy: KinshipPolicy):
    result = list(ranking)
    if len(result) < 6:
        return result, None
    displaced = result[4]
    if policy.guard_rank5_amendment and is_amendment(titles.get(displaced, "")):
        return result, None
    anchors = result[:policy.top_k]
    for index in range(5, min(len(result), policy.candidate_max)):
        candidate = result[index]
        for anchor in anchors:
            matched, reason = amendment_relation(
                titles.get(candidate, ""), titles.get(anchor, ""), allow_subject=policy.allow_subject,
            )
            if matched:
                result.pop(index); result.insert(4, candidate)
                return result, {"candidate":candidate,"displaced":displaced,"from_rank":index+1,"anchor":anchor,"reason":reason}
    return result, None
