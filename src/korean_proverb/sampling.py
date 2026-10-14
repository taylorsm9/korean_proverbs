"""Sampling parameters per plan.md Step 4."""
from __future__ import annotations

from typing import Any


def chat_kwargs(thinking: bool, thinking_token_budget: int = 8192) -> dict[str, Any]:
    """Return the kwargs to pass to client.chat.completions.create().

    Does not include `messages`, `model`, `response_format`, `seed`, `timeout`.
    """
    if thinking:
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "max_tokens": 12288,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_token_budget": thinking_token_budget,
            },
        }
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "max_tokens": 1024,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def retry_kwargs(thinking: bool) -> dict[str, Any]:
    """Larger budget for a `finish_reason == 'length'` retry (one retry only)."""
    if thinking:
        kw = chat_kwargs(thinking=True, thinking_token_budget=16384)
        kw["max_tokens"] = 20480
        return kw
    kw = chat_kwargs(thinking=False)
    kw["max_tokens"] = 2048
    return kw


def default_timeout(thinking: bool) -> int:
    # Under concurrent load a single thinking-on request can take well past
    # 600s to produce its full 12,288-token budget (plan.md). The plan calls
    # out 600s as a baseline; we use 1800s as a safety ceiling. For thinking
    # off, 120s is still comfortably over the real latency.
    return 1800 if thinking else 120
