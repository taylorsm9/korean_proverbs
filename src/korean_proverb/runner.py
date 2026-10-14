"""Async request runner used by the timing probe, the pilot and the main run.

A single call to `translate_row` executes one attempt and returns the attempt
record. Retries and acceptance are driven by `translate_row_with_retry`.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, APITimeoutError, APIError

from . import config
from .attempts import build_attempt_record, new_attempt_id
from .prompts import load as load_prompt, user_message_translate, user_message_nodef
from .sampling import chat_kwargs, retry_kwargs, default_timeout
from .schema import ProverbOnly, ProverbTranslation, response_format


@dataclass
class AttemptResult:
    record: dict[str, Any]
    hit_budget: bool
    finish_reason: str
    content: str
    reasoning: str


def _request_params(thinking: bool, seed: int, retry: bool, with_def: bool, budget_accepted: bool = True) -> dict[str, Any]:
    kw = retry_kwargs(thinking) if retry else chat_kwargs(thinking)
    extra = dict(kw["extra_body"])
    if not budget_accepted:
        extra.pop("thinking_token_budget", None)
    return {
        "temperature": kw["temperature"],
        "top_p": kw["top_p"],
        "presence_penalty": kw["presence_penalty"],
        "max_tokens": kw["max_tokens"],
        "seed": int(seed),
        "extra_body": extra,
    }


async def one_attempt(
    *,
    client: AsyncOpenAI,
    row: dict,
    seed: int,
    prompt_version: str,
    thinking: bool,
    reason: str,
    attempt_no: int,
    retry: bool = False,
    vllm_version: str | None = None,
    budget_accepted: bool = True,
    with_definition: bool = True,
) -> AttemptResult:
    """Fire one chat.completions.create call. The caller logs the record."""
    if with_definition:
        system_prompt = load_prompt(prompt_version)
        user_msg = user_message_translate(row["ko"], row["definition_ko"], row["type"] == "북한어")
        rf = response_format(ProverbTranslation, "ProverbTranslation")
    else:
        system_prompt = load_prompt("p1-nodef")
        user_msg = user_message_nodef(row["ko"])
        rf = response_format(ProverbOnly, "ProverbOnly")

    params = _request_params(thinking, seed, retry, with_definition, budget_accepted=budget_accepted)
    started = time.time()
    resp = await client.chat.completions.create(
        model=config.TRANSLATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format=rf,
        timeout=default_timeout(thinking),
        **params,
    )
    record = build_attempt_record(
        row_id=row["id"],
        attempt_no=attempt_no,
        prompt_version=prompt_version if with_definition else "p1-nodef",
        thinking=thinking,
        reason=reason,
        request={
            "user_message": user_msg,
            "params": {k: v for k, v in params.items() if k != "extra_body"},
            "extra_body": params["extra_body"],
        },
        response=resp,
        started_at=started,
        model=config.TRANSLATION_MODEL,
        vllm_version=vllm_version,
        attempt_id=new_attempt_id(),
    )
    hit_budget = (
        thinking
        and record.get("reasoning_tokens") is not None
        and record["reasoning_tokens"] >= params["extra_body"].get("thinking_token_budget", 10**9)
    )
    return AttemptResult(
        record=record,
        hit_budget=hit_budget,
        finish_reason=record["finish_reason"],
        content=record["content"],
        reasoning=record["reasoning"],
    )


async def gather_bounded(
    tasks: list[Any],
    *,
    concurrency: int,
    return_exceptions: bool = True,
) -> list[Any]:
    """Run coroutines with a bounded worker pool.

    By default, exceptions are returned in-place rather than cancelling the
    whole batch — critical when the server is slow and some requests may
    time out. Callers must check each result for Exception.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(t) for t in tasks], return_exceptions=return_exceptions)
