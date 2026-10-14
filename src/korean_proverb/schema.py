"""Pydantic output models and JSON-schema response_format helpers."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProverbTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proverb_en: str = Field(description=(
        "Faithful English translation of the Korean proverb itself. Keeps every "
        "concrete image and the proverb's form. No explanation, no moral, no "
        "substituted English proverb."))
    definition_en: str = Field(description=(
        "Faithful English translation of the Korean dictionary definition, and "
        "nothing else."))


class ProverbOnly(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proverb_en: str = Field(description="Faithful English translation of the Korean proverb.")


class JudgeRowFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    substituted: bool
    explained: bool
    image_dropped: bool
    added: bool
    mistranslated: bool
    def_unfaithful: bool


class JudgePairChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winner: str  # "A" | "B" | "tie"


def response_format(model_cls: type[BaseModel], name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": model_cls.model_json_schema(),
            "strict": True,
        },
    }
