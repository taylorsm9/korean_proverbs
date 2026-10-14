"""Step 5a: Mechanical QA checks → qa_flags per row.

Pure functions so they can be reused by the timing probe report and by the
main QA pass.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from rapidfuzz import fuzz

from .schema import ProverbTranslation

HANGUL_LO = "가"
HANGUL_HI = "힣"


def _has_hangul(s: str) -> bool:
    return any(HANGUL_LO <= c <= HANGUL_HI for c in s)


def flags_for(
    *,
    content: str,
    definition_ko: str,
    synonyms_ko: list[str],
    thinking: bool,
    reasoning: str,
    finish_reason: str,
    budget_hit: bool,
) -> tuple[list[str], ProverbTranslation | None]:
    """Return (qa_flags, parsed-model-or-None)."""
    flags: list[str] = []
    parsed: ProverbTranslation | None = None
    try:
        parsed = ProverbTranslation.model_validate_json(content)
    except (ValidationError, ValueError, json.JSONDecodeError):
        flags.append("invalid_json")
        return flags, None

    proverb_en = parsed.proverb_en.strip()
    definition_en = parsed.definition_en.strip()

    if not proverb_en or not definition_en:
        flags.append("empty_field")

    if _has_hangul(proverb_en) or _has_hangul(definition_en):
        flags.append("hangul_in_en")

    if len(definition_ko) > 0:
        ratio = len(definition_en) / len(definition_ko)
        if ratio < 0.8 or ratio > 5.0:
            flags.append("length_ratio")

    if len(proverb_en.split()) < 3:
        flags.append("too_short")

    if "(" in proverb_en or ")" in proverb_en:
        flags.append("parenthetical")

    low = definition_en.lower()
    if "synonym" in low:
        flags.append("synonym_leak")
    else:
        for syn in synonyms_ko:
            if syn and syn in definition_en:
                flags.append("synonym_leak")
                break

    if finish_reason == "length" or budget_hit:
        flags.append("truncated")

    if thinking:
        if not reasoning or len(reasoning) < 20:
            flags.append("empty_reasoning")
        elif proverb_en:
            score = fuzz.partial_ratio(proverb_en.lower(), reasoning.lower())
            if score < 80:
                flags.append("trace_mismatch")

    return flags, parsed
