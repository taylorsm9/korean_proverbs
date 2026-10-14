"""Smoke tests for the definition splitter and sense picker."""
from __future__ import annotations

from korean_proverb.extract import build_rows, pick_sense, split_definition


def test_split_no_synonym() -> None:
    d = "남의 가난한 살림을 도와주기란 끝이 없는 일이어서..."
    before, syns = split_definition(d)
    assert before == d
    assert syns == []


def test_split_one_synonym() -> None:
    d = "쓸데없다는 말. <동의 속담> ‘가난은 나라(님)도 못 당한다’"
    before, syns = split_definition(d)
    assert before == "쓸데없다는 말."
    assert syns == ["가난은 나라(님)도 못 당한다"]


def test_split_two_synonyms_no_separator() -> None:
    d = "뜻. <동의 속담>‘가 나’‘다 라’"
    before, syns = split_definition(d)
    assert before == "뜻."
    assert syns == ["가 나", "다 라"]


def test_split_gwanyonggu_tag() -> None:
    d = "뜻. <동의 관용구> ‘ㄱ ㄴ’"
    _, syns = split_definition(d)
    assert syns == ["ㄱ ㄴ"]


def _s(sid: str, no: str, t: str) -> dict:
    return {"id": sid, "sense_no": no, "type": t, "definition_ko": "", "regions": [], "headword_raw": "x", "group_code": ""}


def test_pick_standard_over_nk() -> None:
    senses = [_s("a", "001", "북한어"), _s("b", "002", "일반어")]
    kept, others = pick_sense(senses)
    assert kept["id"] == "b"
    assert [o["id"] for o in others] == ["a"]


def test_pick_lowest_standard() -> None:
    senses = [
        _s("x", "003", "일반어"),
        _s("y", "001", "북한어"),
        _s("z", "002", "일반어"),
    ]
    kept, _ = pick_sense(senses)
    assert kept["id"] == "z"  # lowest sense_no among 일반어


def test_pick_only_nk() -> None:
    senses = [_s("a", "002", "북한어"), _s("b", "001", "북한어")]
    kept, _ = pick_sense(senses)
    assert kept["id"] == "b"


def _f(sid: str, no: str, t: str, head: str, defn: str) -> dict:
    return {"id": sid, "sense_no": no, "type": t, "definition_ko": defn,
            "regions": [], "headword_raw": head, "group_code": ""}


def test_build_rows_one_per_sense() -> None:
    senses = [
        _f("1", "001", "일반어", "가 나", "뜻 하나."),
        _f("2", "002", "일반어", "가 나", "뜻 둘."),
        _f("3", "003", "북한어", "가 나", "뜻 셋."),
        _f("4", "001", "일반어", "다 라", "홀로."),
    ]
    rows = build_rows(senses)
    assert len(rows) == 4
    by_id = {r["id"]: r for r in rows}
    # Canonical pick: lowest sense_no among 일반어 for "가 나" is id=1
    assert by_id["1"]["is_canonical_sense"] is True
    assert by_id["2"]["is_canonical_sense"] is False
    assert by_id["3"]["is_canonical_sense"] is False
    # Siblings for id=1 are its headword mates (2, 3)
    assert set(by_id["1"]["sibling_sense_ids"]) == {"2", "3"}
    # Solo headword has no siblings
    assert by_id["4"]["sibling_sense_ids"] == []
    assert by_id["4"]["is_canonical_sense"] is True
