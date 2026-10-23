"""Thinking-on translation pass over the full dataset.

Runs the same 10,763-row set as `translate.py`, but with `enable_thinking=True`
and Qwen3.5-122B-A10B's thinking-mode general-tasks sampling preset. Writes to
a separate output file so the thinking-off dataset is left untouched.

Resume-safe: skips ids already present in `TRANSLATIONS_THINKING_JSONL`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from tqdm import tqdm

from . import config
from .client import async_client
from .jsonl import accepted_ids, append_jsonl, read_jsonl
from .runner import FatalModelError, gather_bounded, reset_fatal_event
from .sampling import THINKING_TOKEN_BUDGET
from .schema import ProverbTranslation
from .translate import read_prompt_choice, translate_one


def _accepted_to_output_row(row: dict, record: dict) -> dict:
    """Produce one translations_thinking.jsonl row from an accepted attempt.

    Mirrors `translate._accepted_to_output_row` but adds the reasoning trace
    and the budget-overflow flag.
    """
    parsed = ProverbTranslation.model_validate_json(record["content"].strip())
    rtok = record.get("reasoning_tokens")
    budget = record["request"]["extra_body"].get("thinking_token_budget", THINKING_TOKEN_BUDGET)
    thinking_budget_hit = bool(rtok is not None and rtok >= budget)
    return {
        "id": row["id"],
        "proverb_en": parsed.proverb_en,
        "definition_en": parsed.definition_en,
        "reasoning": record.get("reasoning", ""),
        "reasoning_tokens": rtok,
        "reasoning_hangul_ratio": record.get("reasoning_hangul_ratio"),
        "thinking_budget_hit": thinking_budget_hit,
        "finish_reason": record["finish_reason"],
        "completion_tokens": record.get("completion_tokens"),
        "attempt_id": record["attempt_id"],
        "prompt_version": record["prompt_version"],
        "thinking": record["thinking"],
        "model": record["model"],
        "seed": record["request"]["params"]["seed"],
        "latency_s": record["latency_s"],
        "timestamp": record["timestamp"],
    }


async def run_thinking(concurrency: int) -> int:
    config.ensure_dirs()
    prompt_version = read_prompt_choice()
    print(f"[thinking] prompt={prompt_version} thinking=True concurrency={concurrency}",
          file=sys.stderr)

    preflight_path = config.RUNS / "preflight.json"
    if not preflight_path.exists():
        print(f"[thinking] {preflight_path} missing; run `make preflight` first.",
              file=sys.stderr)
        return 2
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    vllm_version = preflight.get("vllm_version")
    budget_accepted = preflight.get("thinking_token_budget_accepted", True)

    rows = list(read_jsonl(config.PROVERBS_JSONL))
    done = accepted_ids(config.TRANSLATIONS_THINKING_JSONL)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[thinking] {len(rows):,} total, {len(done):,} already done, {len(todo):,} to translate",
          file=sys.stderr)
    if not todo:
        return 0

    client = async_client()
    pbar = tqdm(total=len(todo), desc="thinking")
    log_fp = (config.RUNS / "translate_thinking.log").open("a", encoding="utf-8")
    start = time.time()
    done_count = 0

    async def _one(row):
        nonlocal done_count
        try:
            accepted, attempts = await translate_one(
                client=client,
                row=row,
                prompt_version=prompt_version,
                thinking=True,
                reason="thinking_first",
                seed=int(row["id"]),
                vllm_version=vllm_version,
                budget_accepted=budget_accepted,
            )
        except FatalModelError:
            raise
        except Exception as e:
            # Log the failure; batch continues. attempts.jsonl will not have
            # a record for this row, but the retry mechanism inside
            # translate_one only covers finish_reason/invalid-JSON — a hard
            # exception bypasses it.
            log_fp.write(f"{time.strftime('%H:%M:%S')} id={row['id']} error={type(e).__name__}: {e}\n")
            log_fp.flush()
            return
        append_jsonl(config.ATTEMPTS_JSONL, attempts)
        if accepted is not None:
            append_jsonl(config.TRANSLATIONS_THINKING_JSONL, [_accepted_to_output_row(row, accepted)])
        pbar.update(1)
        done_count += 1
        elapsed = time.time() - start
        rph = done_count / elapsed * 3600
        remain = (len(todo) - done_count) / max(rph / 3600, 1e-6)
        if done_count % 25 == 0 or done_count == len(todo):
            log_fp.write(f"{time.strftime('%H:%M:%S')} {done_count}/{len(todo)} "
                         f"rows_per_hour={rph:.0f} eta_s={remain:.0f}\n")
            log_fp.flush()

    reset_fatal_event()
    tasks = [_one(r) for r in todo]
    results = await gather_bounded(tasks, concurrency=concurrency)
    pbar.close()
    log_fp.close()
    fatal = [r for r in results if isinstance(r, FatalModelError)]
    if fatal:
        print(f"[thinking] FATAL: {len(fatal):,}/{len(results):,} tasks saw FatalModelError. "
              f"Server model likely changed; check /v1/models and re-run preflight.",
              file=sys.stderr)
        return 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()
    return asyncio.run(run_thinking(args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
