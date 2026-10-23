"""Thinking-on research runs F and G from `plan_reasoning_embeddings.md`.

All stages use the thinking run's settings (Qwen3.5-122B-A10B thinking preset,
`enable_thinking=True`) on the canonical senses of the 1,094 multi-sense
headwords from `rq1.build_pairs`.

Stages:
    reseed         F: 300-row sample, prompt p1-zero, seed = id + 1,000,000
    nodef          G: all 1,094 rows, prompt p1-nodef, seed = id
    nodef-meaning  G: prompt p1-nodef-meaning (adds `meaning_en`), seed = id.
                   `--subset sample` runs the 300-row sample, `--subset all`
                   the rest.

The 300-row sample is drawn once (100 per definition-length tercile, like the
timing probe) and saved to `data/research/thinking_sample_ids.json`, so every
stage uses the same rows. Outputs go to `data/research/thinking_<stage>.jsonl`.
Resume-safe: skips ids already present in the stage's output.
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
from .probe_thinking import _stratified_sample
from .rq1 import build_pairs
from .runner import FatalModelError, gather_bounded, reset_fatal_event
from .sampling import THINKING_TOKEN_BUDGET
from .schema import NODEF_SCHEMAS, ProverbTranslation
from .translate import read_prompt_choice, translate_one

SAMPLE_PER_TERCILE = 100
SAMPLE_SEED = 20261022
SAMPLE_IDS_JSON = config.RESEARCH / "thinking_sample_ids.json"

# stage -> (with_definition, nodef_prompt, seed offset)
STAGES = {
    "reseed": (True, None, 1_000_000),
    "nodef": (False, "p1-nodef", 0),
    "nodef-meaning": (False, "p1-nodef-meaning", 0),
}


def _canonical_rows() -> list[dict]:
    by_id = {r["id"]: r for r in read_jsonl(config.PROVERBS_JSONL)}
    return [by_id[p["id"]] for p in build_pairs()]


def _sample_ids(rows: list[dict]) -> list[str]:
    """Load the fixed 300-row sample, drawing and saving it on first use."""
    if SAMPLE_IDS_JSON.exists():
        return json.loads(SAMPLE_IDS_JSON.read_text(encoding="utf-8"))["ids"]
    ids = [r["id"] for r in _stratified_sample(rows, SAMPLE_PER_TERCILE, SAMPLE_SEED)]
    SAMPLE_IDS_JSON.write_text(json.dumps({
        "seed": SAMPLE_SEED, "per_tercile": SAMPLE_PER_TERCILE, "ids": ids,
    }, indent=1), encoding="utf-8")
    return ids


def _accepted_to_output_row(row: dict, record: dict, schema_cls) -> dict:
    """Like `translate_thinking._accepted_to_output_row`, for any output schema."""
    parsed = schema_cls.model_validate_json(record["content"].strip())
    rtok = record.get("reasoning_tokens")
    budget = record["request"]["extra_body"].get("thinking_token_budget", THINKING_TOKEN_BUDGET)
    return {
        "id": row["id"],
        **parsed.model_dump(),
        "reasoning": record.get("reasoning", ""),
        "reasoning_tokens": rtok,
        "reasoning_hangul_ratio": record.get("reasoning_hangul_ratio"),
        "thinking_budget_hit": bool(rtok is not None and rtok >= budget),
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


async def run_stage(stage: str, subset: str, concurrency: int) -> int:
    config.ensure_dirs()
    with_def, nodef_prompt, seed_offset = STAGES[stage]
    prompt_version = read_prompt_choice() if with_def else nodef_prompt
    schema_cls = ProverbTranslation if with_def else NODEF_SCHEMAS[nodef_prompt]
    out_path = config.RESEARCH / f"thinking_{stage.replace('-', '_')}.jsonl"

    preflight = json.loads((config.RUNS / "preflight.json").read_text(encoding="utf-8"))
    vllm_version = preflight.get("vllm_version")
    budget_accepted = preflight.get("thinking_token_budget_accepted", True)

    rows = _canonical_rows()
    if subset == "sample":
        keep = set(_sample_ids(rows))
        rows = [r for r in rows if r["id"] in keep]
    done = accepted_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{stage}] prompt={prompt_version} subset={subset} concurrency={concurrency}: "
          f"{len(rows):,} rows, {len(done):,} already done, {len(todo):,} to run",
          file=sys.stderr)
    if not todo:
        return 0

    client = async_client()
    pbar = tqdm(total=len(todo), desc=stage)
    log_fp = (config.RUNS / "thinking_research.log").open("a", encoding="utf-8")
    log_fp.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {stage} subset={subset} todo={len(todo)}\n")
    log_fp.flush()
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
                reason=f"research_{stage}",
                seed=int(row["id"]) + seed_offset,
                vllm_version=vllm_version,
                budget_accepted=budget_accepted,
                with_definition=with_def,
                nodef_prompt=nodef_prompt or "p1-nodef",
            )
        except FatalModelError:
            raise
        except Exception as e:
            log_fp.write(f"{time.strftime('%H:%M:%S')} {stage} id={row['id']} error={type(e).__name__}: {e}\n")
            log_fp.flush()
            return
        append_jsonl(config.ATTEMPTS_JSONL, attempts)
        if accepted is not None:
            append_jsonl(out_path, [_accepted_to_output_row(row, accepted, schema_cls)])
        pbar.update(1)
        done_count += 1
        elapsed = time.time() - start
        rph = done_count / elapsed * 3600
        remain = (len(todo) - done_count) / max(rph / 3600, 1e-6)
        if done_count % 25 == 0 or done_count == len(todo):
            log_fp.write(f"{time.strftime('%H:%M:%S')} {stage} {done_count}/{len(todo)} "
                         f"rows_per_hour={rph:.0f} eta_s={remain:.0f}\n")
            log_fp.flush()

    reset_fatal_event()
    results = await gather_bounded([_one(r) for r in todo], concurrency=concurrency)
    pbar.close()
    log_fp.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end {stage} subset={subset} "
                 f"accepted_total={len(accepted_ids(out_path))}\n")
    log_fp.close()
    fatal = [r for r in results if isinstance(r, FatalModelError)]
    if fatal:
        print(f"[{stage}] FATAL: {len(fatal):,}/{len(results):,} tasks saw FatalModelError. "
              f"Server model likely changed; check /v1/models.", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("--subset", choices=["sample", "all"], default="all")
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()
    return asyncio.run(run_stage(args.stage, args.subset, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
