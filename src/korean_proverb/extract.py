"""Step 2: Stream Urimalsaem XML dumps → one proverb per headword.

Reads every *.xml under data/raw/korean-dict-nikl/opendict/, filters to
word_unit == '속담', applies the one-meaning-per-proverb rule, canonicalizes
the headword, splits synonyms out of the definition, and writes
data/interim/proverbs.jsonl.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree.ElementTree import Element, iterparse

from . import config
from .canonicalize import canonicalize

# Definition cleanup
SYNONYM_SPLIT_RE = re.compile(r"<동의\s*(?:속담|관용구)>")
SYNONYM_QUOTED_RE = re.compile(r"‘([^’]*)’")  # ‘…’


def _text(elem: Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def iter_proverb_senses(xml_dir: Path) -> Iterator[dict]:
    """Yield one dict per sense with word_unit == '속담', streaming the XML."""
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files in {xml_dir}")
    for path in xml_files:
        for _event, elem in iterparse(path, events=("end",)):
            if elem.tag != "item":
                continue
            word_info = elem.find("wordInfo")
            if word_info is None:
                elem.clear()
                continue
            if _text(word_info.find("word_unit")) != "속담":
                elem.clear()
                continue
            headword_raw = _text(word_info.find("word"))
            # One <item> has exactly one <senseInfo>; the dictionary carries
            # multiple senses by having multiple <item>s with the same headword.
            sense_info = elem.find("senseInfo")
            if sense_info is None:
                elem.clear()
                continue
            target_code = _text(elem.find("target_code"))
            group_code = _text(elem.find("group_code"))
            sense_no = _text(sense_info.find("sense_no"))
            stype = _text(sense_info.find("type"))
            definition = _text(sense_info.find("definition"))
            regions = [
                _text(r.find("region"))
                for r in sense_info.findall("region_info")
                if _text(r.find("region"))
            ]
            yield {
                "id": target_code,
                "group_code": group_code,
                "headword_raw": headword_raw,
                "sense_no": sense_no,
                "type": stype,
                "regions": regions,
                "definition_ko": definition,
            }
            elem.clear()


def split_definition(definition_ko: str) -> tuple[str, list[str]]:
    """Return (clean_definition, synonyms). Synonyms come from ‘…’ quotes
    after the <동의 속담> / <동의 관용구> tag."""
    m = SYNONYM_SPLIT_RE.search(definition_ko)
    if not m:
        return definition_ko.strip(), []
    before = definition_ko[: m.start()].rstrip()
    after = definition_ko[m.end():]
    synonyms = [s.strip() for s in SYNONYM_QUOTED_RE.findall(after) if s.strip()]
    return before, synonyms


def _sense_no_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 10**9


def pick_sense(senses: list[dict]) -> tuple[dict, list[dict]]:
    """Return (kept, others). Rule: lowest sense_no among 일반어; else lowest overall."""
    standard = [s for s in senses if s["type"] == "일반어"]
    pool = standard if standard else senses
    pool_sorted = sorted(pool, key=lambda s: _sense_no_int(s["sense_no"]))
    kept = pool_sorted[0]
    others = [s for s in senses if s["id"] != kept["id"]]
    others.sort(key=lambda s: _sense_no_int(s["sense_no"]))
    return kept, others


def build_rows(senses: Iterable[dict]) -> list[dict]:
    """One row per sense (10,763 rows). Each sense of a given proverb
    becomes its own row, with `is_canonical_sense` marking the dictionary's
    primary sense per the Step 2 selection rule, and `sibling_sense_ids`
    pointing to the other senses of the same proverb so researchers can
    filter down to primary senses or regroup."""
    by_head: dict[str, list[dict]] = defaultdict(list)
    senses_list = list(senses)
    for s in senses_list:
        by_head[s["headword_raw"]].append(s)
    canonical_id_by_head: dict[str, str] = {}
    for head, group in by_head.items():
        kept, _ = pick_sense(group)
        canonical_id_by_head[head] = kept["id"]

    rows: list[dict] = []
    for head, group in by_head.items():
        ko, flags = canonicalize(head)
        canonical_id = canonical_id_by_head[head]
        sibling_ids = [s["id"] for s in group]
        for s in group:
            clean_def, synonyms_ko = split_definition(s["definition_ko"])
            figurative = "비유적으로" in clean_def
            literal_gloss = "뜻으로" in clean_def
            rows.append({
                "id": s["id"],
                "headword_raw": head,
                "ko": ko,
                "sense_no": s["sense_no"],
                "type": s["type"],
                "regions": s["regions"],
                "definition_ko": clean_def,
                "synonyms_ko": synonyms_ko,
                "sibling_sense_ids": [i for i in sibling_ids if i != s["id"]],
                "is_canonical_sense": s["id"] == canonical_id,
                "flags": flags,
                "figurative": figurative,
                "literal_gloss": literal_gloss,
            })
    return rows


def resolve_synonym_ids(rows: list[dict]) -> tuple[int, int]:
    """Add synonym_ids by matching synonym strings to headword_raw exactly.

    With one row per sense, the same headword string appears on multiple
    rows; synonym references resolve to the *canonical* sense's id for
    stability.

    Returns (resolved_links, total_links)."""
    by_raw_canonical: dict[str, str] = {}
    for r in rows:
        if r.get("is_canonical_sense"):
            by_raw_canonical[r["headword_raw"]] = r["id"]
    resolved = 0
    total = 0
    for r in rows:
        ids: list[str] = []
        for syn in r["synonyms_ko"]:
            total += 1
            rid = by_raw_canonical.get(syn)
            if rid:
                ids.append(rid)
                resolved += 1
        r["synonym_ids"] = ids
    return resolved, total


def _ko_has_hanja_or_pua(rows: list[dict]) -> tuple[int, int]:
    hanja = sum(1 for r in rows if "has_hanja" in r["flags"])
    pua = sum(1 for r in rows if "has_pua" in r["flags"])
    return hanja, pua


def summarize(senses: list[dict], rows: list[dict]) -> dict:
    """One row per sense; headword-level counts come from grouping on headword_raw."""
    by_head: dict[str, list[dict]] = defaultdict(list)
    for s in senses:
        by_head[s["headword_raw"]].append(s)
    headwords = set(r["headword_raw"] for r in rows)
    multi_sense_heads = sum(1 for g in by_head.values() if len(g) > 1)
    extra_senses = sum(len(g) - 1 for g in by_head.values())
    both_types_heads = 0
    sense001_nk = 0
    for head, group in by_head.items():
        types = {s["type"] for s in group}
        if "일반어" in types and "북한어" in types:
            both_types_heads += 1
            group_sorted = sorted(group, key=lambda s: _sense_no_int(s["sense_no"]))
            if group_sorted[0]["type"] == "북한어":
                sense001_nk += 1
    standard_multi = sum(
        1 for group in by_head.values()
        if sum(1 for s in group if s["type"] == "일반어") > 1
    )
    heads_with_bracket = sum(1 for h in headwords if "[" in h)
    heads_with_paren = sum(1 for h in headwords if "(" in h)
    heads_with_slash = sum(1 for h in headwords if "/" in h)
    hanja, pua = _ko_has_hanja_or_pua(rows)
    unique_ko = len({r["ko"] for r in rows})
    total_proverb_chars_unique = sum(len(ko) for ko in {r["ko"] for r in rows})
    total_def_chars = sum(len(r["definition_ko"]) for r in rows)
    row_types = Counter(r["type"] for r in rows)
    canonical_rows = sum(1 for r in rows if r.get("is_canonical_sense"))
    return {
        "senses_total": len(senses),
        "rows_total": len(rows),
        "unique_headwords": len(headwords),
        "sense_types_all": dict(row_types),
        "canonical_rows": canonical_rows,
        "multi_sense_heads": multi_sense_heads,
        "extra_senses": extra_senses,
        "both_types_heads": both_types_heads,
        "sense001_nk_in_mixed": sense001_nk,
        "heads_with_more_than_one_standard": standard_multi,
        "heads_with_bracket": heads_with_bracket,
        "heads_with_paren": heads_with_paren,
        "heads_with_slash": heads_with_slash,
        "ko_has_hanja": hanja,
        "ko_has_pua": pua,
        "unique_ko": unique_ko,
        "total_proverb_chars_unique": total_proverb_chars_unique,
        "total_def_chars": total_def_chars,
    }


EXPECTED = {
    "senses_total": 10763,
    "rows_total": 10763,
    "unique_headwords": 9612,
    "canonical_rows": 9612,
    "sense_types_all": {"일반어": 8422, "북한어": 2341},
    "multi_sense_heads": 1094,
    "extra_senses": 1151,
    "both_types_heads": 347,
    "sense001_nk_in_mixed": 176,
    "heads_with_more_than_one_standard": 760,
    "heads_with_bracket": 569,
    "heads_with_paren": 415,
    "heads_with_slash": 41,
}


def check_counts(summary: dict) -> list[str]:
    """Return a list of discrepancies (empty if everything matches within tol)."""
    issues: list[str] = []
    for key, expected in EXPECTED.items():
        got = summary.get(key)
        if isinstance(expected, dict):
            for k, v in expected.items():
                g = (got or {}).get(k, 0)
                if abs(g - v) / max(v, 1) > 0.02:
                    issues.append(f"{key}.{k}: expected {v}, got {g}")
        else:
            if abs(got - expected) / max(expected, 1) > 0.02:
                issues.append(f"{key}: expected {expected}, got {got}")
    return issues


def main() -> int:
    config.ensure_dirs()
    xml_dir = config.RAW_OPENDICT
    print(f"Reading XML from {xml_dir}", file=sys.stderr)
    # Capture raw definitions so we can count '<동의 속담>' pre-split.
    senses: list[dict] = []
    raw_defs: list[str] = []
    for s in iter_proverb_senses(xml_dir):
        raw_defs.append(s["definition_ko"])
        senses.append(s)
    print(f"Loaded {len(senses):,} proverb senses", file=sys.stderr)

    rows = build_rows(senses)
    resolved, total = resolve_synonym_ids(rows)
    print(f"Resolved {resolved:,}/{total:,} synonym links", file=sys.stderr)

    summary = summarize(senses, rows)

    # Write JSONL (stable key order).
    config.PROVERBS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with config.PROVERBS_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows to {config.PROVERBS_JSONL}", file=sys.stderr)

    # Dump summary to stderr next to the expected table.
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    issues = check_counts(summary)
    if issues:
        print("DISCREPANCIES (>2% from expected):", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        print(
            "The plan says: stop and ask the user if extraction counts are off by more than ~2%.",
            file=sys.stderr,
        )
        return 2
    print("Counts match expected values within 2%.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
