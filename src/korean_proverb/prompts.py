"""Load system prompts from runs/prompts/."""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Load prompt text by version name (e.g. 'p1-zero', 'j1-row')."""
    path = config.PROMPTS / f"{name}.txt"
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def user_message_translate(ko: str, definition_ko: str, nk: bool) -> str:
    msg = f"Proverb: {ko}\nDefinition: {definition_ko}"
    if nk:
        msg += "\nNote: this proverb is recorded as North Korean usage."
    return msg


def user_message_nodef(ko: str) -> str:
    return f"Proverb: {ko}"


def user_message_judge_row(ko: str, definition_ko: str, proverb_en: str, definition_en: str) -> str:
    return (
        f"Proverb: {ko}\n"
        f"Definition: {definition_ko}\n"
        f"proverb_en: {proverb_en}\n"
        f"definition_en: {definition_en}"
    )


def user_message_judge_pair(
    ko: str, definition_ko: str,
    a_prov: str, a_def: str, b_prov: str, b_def: str,
) -> str:
    return (
        f"Proverb: {ko}\n"
        f"Definition: {definition_ko}\n"
        f"A.proverb_en: {a_prov}\n"
        f"A.definition_en: {a_def}\n"
        f"B.proverb_en: {b_prov}\n"
        f"B.definition_en: {b_def}"
    )


def user_message_corrupt(
    ko: str, definition_ko: str, proverb_en: str, definition_en: str, error_type: str,
) -> str:
    return (
        f"Proverb (Korean): {ko}\n"
        f"Definition (Korean): {definition_ko}\n"
        f"proverb_en: {proverb_en}\n"
        f"definition_en: {definition_en}\n"
        f"error_type: {error_type}"
    )
