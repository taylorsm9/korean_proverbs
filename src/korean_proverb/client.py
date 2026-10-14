"""Thin wrappers around the OpenAI SDK pointed at the Spark vLLM endpoint
and (optionally) the 4090 judge endpoint. Also handles reading the reasoning
field under its possible names."""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from . import config


def sync_client(base_url: str | None = None) -> OpenAI:
    base = base_url or config.qwen_base_url()
    # vLLM has no auth; openai SDK requires a non-empty key.
    return OpenAI(base_url=base, api_key="EMPTY", timeout=600.0)


def async_client(base_url: str | None = None) -> AsyncOpenAI:
    base = base_url or config.qwen_base_url()
    return AsyncOpenAI(base_url=base, api_key="EMPTY", timeout=600.0)


def extract_reasoning(message: Any) -> str:
    """Pull the reasoning trace off a chat message, with fallbacks for older
    vLLM field names. Order: `reasoning`, `reasoning_content`, model_extra."""
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None)
        if val:
            return val
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        v = extra.get(key)
        if v:
            return v
    return ""


def reasoning_tokens(usage: Any) -> int | None:
    """Return reasoning_tokens from usage.completion_tokens_details, if the
    server exposes it (vLLM >= 0.28)."""
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return None
    val = getattr(details, "reasoning_tokens", None)
    if val is None and hasattr(details, "model_extra"):
        val = (details.model_extra or {}).get("reasoning_tokens")
    return val
