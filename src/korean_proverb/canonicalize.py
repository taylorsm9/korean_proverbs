"""Canonicalize a raw Urimalsaem headword into the proverb's canonical form.

Rules (from plan.md Step 2):
1. Delete every `[...]` block, including its contents.
2. `(...)` whose contents contain Hanja (U+4E00–9FFF): delete parens AND contents.
3. Any other `(...)` (Hangul inside): keep the contents, drop parens.
   - If contents contain `/`, keep only the first alternative.
4. Collapse whitespace and strip.
5. Flag `has_hanja` / `has_pua` if those characters remain.
"""
from __future__ import annotations

import re

HANJA_RE = re.compile(r"[一-鿿]")
PUA_RE = re.compile(r"[-]")
BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def _resolve_paren(contents: str) -> str:
    """Decide what to put in place of a `(contents)` group."""
    if HANJA_RE.search(contents):
        # Hanja gloss, drop entirely
        return ""
    # Hangul: keep contents, pick first alternative if there's a slash
    if "/" in contents:
        contents = contents.split("/", 1)[0]
    return contents


def canonicalize(headword_raw: str) -> tuple[str, list[str]]:
    """Return (canonical, flags). Flags is a list of 'has_hanja' / 'has_pua'."""
    s = headword_raw

    # 1. Delete every [...] block, including contents. Handle nested-free brackets
    # by repeated removal (the data has no nested brackets in practice).
    prev = None
    while prev != s:
        prev = s
        s = BRACKET_RE.sub("", s)

    # 2 + 3. Parentheses: process innermost first in case of nesting.
    prev = None
    while prev != s:
        prev = s
        s = PAREN_RE.sub(lambda m: _resolve_paren(m.group(1)), s)

    # 4. Collapse whitespace, strip.
    s = WS_RE.sub(" ", s).strip()

    flags: list[str] = []
    if HANJA_RE.search(s):
        flags.append("has_hanja")
    if PUA_RE.search(s):
        flags.append("has_pua")
    return s, flags
