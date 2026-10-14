"""Unit tests for the canonicalizer, from plan.md Step 2."""
from __future__ import annotations

import pytest

from korean_proverb.canonicalize import canonicalize


CASES = [
    ("가난 구제는 나라[나라님/임금]도 못한다[어렵다]", "가난 구제는 나라도 못한다"),
    ("굼벵이 천장(遷葬)하듯", "굼벵이 천장하듯"),
    ("살림에는 눈이 보배(라)", "살림에는 눈이 보배라"),
    ("섭산적이 되도록 맞(았)다", "섭산적이 되도록 맞았다"),
    ("가을비는 장인[시아버지/시아비]의 나룻[수염/턱] 밑에서도 긋는다",
     "가을비는 장인의 나룻 밑에서도 긋는다"),
    ("권에 띄어[못 이겨] 방립(方笠) 산다[쓴다]", "권에 띄어 방립 산다"),
    ("곁가마가 (더/먼저) 끓는다", "곁가마가 더 끓는다"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_canonicalize(raw: str, expected: str) -> None:
    canonical, _flags = canonicalize(raw)
    assert canonical == expected


@pytest.mark.parametrize("raw,expected", CASES)
def test_no_leftover_metachars(raw: str, expected: str) -> None:
    canonical, _flags = canonicalize(raw)
    for ch in "[](/)":
        assert ch not in canonical, f"found {ch!r} in {canonical!r}"


def test_hanja_flag() -> None:
    _, flags = canonicalize("순한자(純漢字")
    # Trailing unmatched paren - just make sure Hanja flag can fire when needed.
    # The actual Hanja content is removed by the (遷葬) rule; a leftover Hanja
    # outside parens would trigger has_hanja.
    canonical, flags = canonicalize("가난 漢 못한다")
    assert "has_hanja" in flags
    assert canonical == "가난 漢 못한다"
