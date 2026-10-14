"""Attempt record construction and writing.

Every call to the model — pilot, main run, retry, QA re-run, timing probe,
RQ1 arm — becomes one record in data/traces/attempts.jsonl. The accepted
answer is additionally written to data/output/translations.jsonl.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .client import extract_reasoning, reasoning_tokens
from .jsonl import append_jsonl

HANGUL_RE = re.compile(r"[가-힣]")
LETTER_RE = re.compile(r"\w", re.UNICODE)


def _hangul_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hangul = sum(1 for c in letters if "가" <= c <= "힣")
    return hangul / len(letters)


def new_attempt_id() -> str:
    return uuid.uuid4().hex[:16]


def build_attempt_record(
    *,
    row_id: str,
    attempt_no: int,
    prompt_version: str,
    thinking: bool,
    reason: str,
    request: dict[str, Any],
    response: Any,
    started_at: float,
    model: str,
    vllm_version: str | None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Build the full attempt record for attempts.jsonl."""
    latency_s = time.time() - started_at
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    content = (message.content or "") if message else ""
    reasoning = extract_reasoning(message) if message else ""
    finish_reason = choice.finish_reason if choice else "none"
    usage = getattr(response, "usage", None)
    rtok = reasoning_tokens(usage) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    try:
        raw_response = response.model_dump(exclude_none=True)
    except Exception:
        raw_response = json.loads(json.dumps(response, default=str))

    return {
        "attempt_id": attempt_id or new_attempt_id(),
        "id": row_id,
        "attempt_no": attempt_no,
        "prompt_version": prompt_version,
        "thinking": thinking,
        "reason": reason,
        "request": request,
        "reasoning": reasoning,
        "content": content,
        "raw_response": raw_response,
        "usage": raw_response.get("usage"),
        "reasoning_tokens": rtok,
        "reasoning_words": len(reasoning.split()) if reasoning else 0,
        "reasoning_hangul_ratio": _hangul_ratio(reasoning) if reasoning else 0.0,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "latency_s": latency_s,
        "timestamp": time.time(),
        "model": model,
        "vllm_version": vllm_version,
    }


def log_attempt(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, [record])
