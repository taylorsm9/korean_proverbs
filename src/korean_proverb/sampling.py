"""Sampling parameters per plan.md Step 4."""
from __future__ import annotations

from typing import Any


THINKING_TOKEN_BUDGET = 16384
THINKING_MAX_TOKENS = 20480


def chat_kwargs(thinking: bool, thinking_token_budget: int = THINKING_TOKEN_BUDGET) -> dict[str, Any]:
    """Return the kwargs to pass to client.chat.completions.create().

    Does not include `messages`, `model`, `response_format`, `seed`, `timeout`.

    Thinking-on uses Qwen3.5-122B-A10B's general-tasks preset, verified
    against both Spark vLLM 0.28 instances behind the nginx load balancer.
    """
    if thinking:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "max_tokens": THINKING_MAX_TOKENS,
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
        # Thinking-on already runs at the server's recommended cap. A length
        # retry gets the same budget: if we hit max_tokens once at 20480 with
        # a 16384-token thinking cap, the retry gets fresh randomness via a
        # shifted seed, not more room.
        return chat_kwargs(thinking=True)
    kw = chat_kwargs(thinking=False)
    kw["max_tokens"] = 2048
    return kw


def default_timeout(thinking: bool) -> int:
    # Thinking-on: server-side single-request latency is ~165-230s at low
    # concurrency; under load each request is slower. The nginx LB cap is
    # 3600s; we stay safely under at 3000s. The server operator can raise
    # the LB cap if the probe shows the slowest requests pushing close.
    return 3000 if thinking else 120
