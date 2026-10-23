"""Step 4: Translate every proverb.

Reads the thinking mode from `runs/mode.json` and the prompt version from
`runs/prompt_choice.json`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from pydantic import ValidationError
from tqdm import tqdm

from . import config
from .client import async_client
from .jsonl import accepted_ids, append_jsonl, read_jsonl
from .runner import FatalModelError, gather_bounded, one_attempt, reset_fatal_event
from .schema import NODEF_SCHEMAS, ProverbTranslation


# ---------- mode / choice helpers ----------

def read_mode() -> str:
    if not config.MODE_FILE.exists():
        print(f"[translate] {config.MODE_FILE} missing; write `{{\"mode\":\"B\"}}` to it "
              "to use thinking-off.", file=sys.stderr)
        sys.exit(2)
    return json.loads(config.MODE_FILE.read_text(encoding="utf-8"))["mode"]


def thinking_for_mode(mode: str, *, dataset: bool = True) -> bool:
    """Return the thinking flag for the main-dataset path of each mode."""
    if dataset:
        return mode == "A"
    return mode == "A"  # RQ1 dataset path uses the same thinking mode


def read_prompt_choice() -> str:
    """Return the chosen prompt version from runs/prompt_choice.json."""
    path = config.RUNS / "prompt_choice.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["prompt_version"]
    return "p1-zero"


# ---------- translation ----------

async def translate_one(
    *,
    client,
    row: dict,
    prompt_version: str,
    thinking: bool,
    reason: str,
    seed: int,
    vllm_version: str | None,
    budget_accepted: bool,
    with_definition: bool = True,
    nodef_prompt: str = "p1-nodef",
) -> tuple[dict | None, list[dict]]:
    """Translate one row with up to one retry on `length` or invalid JSON.

    Returns (accepted_record_or_None, attempts_logged).
    """
    attempts: list[dict] = []
    schema_cls = ProverbTranslation if with_definition else NODEF_SCHEMAS[nodef_prompt]

    r1 = await one_attempt(
        client=client,
        row=row,
        seed=seed,
        prompt_version=prompt_version,
        thinking=thinking,
        reason=reason,
        attempt_no=1,
        vllm_version=vllm_version,
        budget_accepted=budget_accepted,
        with_definition=with_definition,
        nodef_prompt=nodef_prompt,
    )
    attempts.append(r1.record)

    accepted: dict | None = None
    try:
        schema_cls.model_validate_json(r1.content.strip())
        validates = True
    except (ValidationError, ValueError, json.JSONDecodeError):
        validates = False

    needs_retry = (r1.finish_reason == "length") or (not validates)

    if not needs_retry:
        accepted = r1.record
    else:
        retry_reason = "retry_length" if r1.finish_reason == "length" else "retry_invalid"
        r2 = await one_attempt(
            client=client,
            row=row,
            seed=seed + 97,
            prompt_version=prompt_version,
            thinking=thinking,
            reason=retry_reason,
            attempt_no=2,
            vllm_version=vllm_version,
            budget_accepted=budget_accepted,
            with_definition=with_definition,
            nodef_prompt=nodef_prompt,
        )
        attempts.append(r2.record)
        try:
            schema_cls.model_validate_json(r2.content.strip())
            validates = True
        except Exception:
            validates = False
        if validates and r2.finish_reason != "length":
            accepted = r2.record
        else:
            accepted = None  # mark as failed; main loop records this

    return accepted, attempts


def _accepted_to_output_row(row: dict, record: dict) -> dict:
    """Produce one translations.jsonl row from an accepted attempt."""
    parsed = ProverbTranslation.model_validate_json(record["content"].strip())
    return {
        "id": row["id"],
        "proverb_en": parsed.proverb_en,
        "definition_en": parsed.definition_en,
        "attempt_id": record["attempt_id"],
        "prompt_version": record["prompt_version"],
        "thinking": record["thinking"],
        "model": record["model"],
        "seed": record["request"]["params"]["seed"],
        "finish_reason": record["finish_reason"],
        "completion_tokens": record.get("completion_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens"),
        "latency_s": record["latency_s"],
        "timestamp": record["timestamp"],
    }


# ---------- main run ----------

async def run_main(mode: str, concurrency: int) -> int:
    config.ensure_dirs()
    prompt_version = read_prompt_choice()
    thinking = thinking_for_mode(mode, dataset=True)
    print(f"[main] mode={mode} prompt={prompt_version} thinking={thinking} concurrency={concurrency}",
          file=sys.stderr)

    preflight = json.loads((config.RUNS / "preflight.json").read_text(encoding="utf-8"))
    vllm_version = preflight.get("vllm_version")
    budget_accepted = preflight.get("thinking_token_budget_accepted", True)

    rows = list(read_jsonl(config.PROVERBS_JSONL))
    done = accepted_ids(config.TRANSLATIONS_JSONL)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[main] {len(rows):,} total, {len(done):,} already done, {len(todo):,} to translate",
          file=sys.stderr)
    if not todo:
        return 0

    client = async_client()
    pbar = tqdm(total=len(todo), desc="translate")
    log_fp = (config.RUNS / "translate.log").open("a", encoding="utf-8")
    start = time.time()
    done_count = 0

    async def _one(row):
        nonlocal done_count
        accepted, attempts = await translate_one(
            client=client,
            row=row,
            prompt_version=prompt_version,
            thinking=thinking,
            reason="first",
            seed=int(row["id"]),
            vllm_version=vllm_version,
            budget_accepted=budget_accepted,
        )
        append_jsonl(config.ATTEMPTS_JSONL, attempts)
        if accepted is not None:
            append_jsonl(config.TRANSLATIONS_JSONL, [_accepted_to_output_row(row, accepted)])
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
        print(f"[main] FATAL: {len(fatal):,}/{len(results):,} tasks saw FatalModelError. "
              f"Server model likely changed; check /v1/models and re-run preflight.",
              file=sys.stderr)
        return 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    mode = read_mode()
    return asyncio.run(run_main(mode, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
