"""Timing probe for the thinking-on translation run.

Samples 24 rows across short/mid/long thirds of definition_ko length, runs
them against the live LB endpoint at the production concurrency, and writes
`runs/thinking_probe.md` with the numbers the server operator asked for:

- median and max seconds per request
- median and max reasoning_tokens
- overall proverbs/hour
- thinking_budget_hit rate
- 3 sample traces
- projected wall-clock for the full 10,763-row run
- a flag if the slowest request comes within 10% of the 3000s client timeout

Attempts land in `data/traces/attempts.jsonl` with reason `thinking_probe`.
Outputs are not written to the production thinking-on jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

from tqdm import tqdm

from . import config
from .client import async_client
from .jsonl import append_jsonl, read_jsonl
from .runner import FatalModelError, gather_bounded, reset_fatal_event
from .sampling import THINKING_TOKEN_BUDGET, default_timeout
from .translate import read_prompt_choice, translate_one


PROBE_N_PER_TERCILE = 8
TIMEOUT_WARN_FRACTION = 0.9  # warn if slowest request > 90% of client timeout


def _stratified_sample(rows: list[dict], n_per_bucket: int, seed: int) -> list[dict]:
    """Sample n_per_bucket rows each from short/mid/long thirds of
    definition_ko length."""
    rng = random.Random(seed)
    sortable = sorted(rows, key=lambda r: len(r["definition_ko"]))
    k = len(sortable) // 3
    short = sortable[:k]
    mid = sortable[k:2 * k]
    long_ = sortable[2 * k:]
    sample = []
    for bucket in (short, mid, long_):
        sample.extend(rng.sample(bucket, n_per_bucket))
    rng.shuffle(sample)
    return sample


def _ms_to_s(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "n/a"


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def run_probe(concurrency: int, n_per_tercile: int, seed: int, out_md: Path, out_jsonl: Path) -> int:
    config.ensure_dirs()
    prompt_version = read_prompt_choice()

    preflight_path = config.RUNS / "preflight.json"
    if not preflight_path.exists():
        print(f"[probe] {preflight_path} missing; run `make preflight` first.", file=sys.stderr)
        return 2
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    vllm_version = preflight.get("vllm_version")
    budget_accepted = preflight.get("thinking_token_budget_accepted", True)

    rows = list(read_jsonl(config.PROVERBS_JSONL))
    sample = _stratified_sample(rows, n_per_tercile, seed)
    print(f"[probe] {len(sample)} rows, concurrency={concurrency}, prompt={prompt_version}, "
          f"vllm={vllm_version}, budget_accepted={budget_accepted}", file=sys.stderr)

    client = async_client()
    pbar = tqdm(total=len(sample), desc="probe")
    results: list[dict] = []
    batch_start = time.time()

    async def _one(row):
        try:
            accepted, attempts = await translate_one(
                client=client,
                row=row,
                prompt_version=prompt_version,
                thinking=True,
                reason="thinking_probe",
                seed=int(row["id"]),
                vllm_version=vllm_version,
                budget_accepted=budget_accepted,
            )
        except FatalModelError:
            raise
        except Exception as e:
            results.append({
                "id": row["id"], "error": f"{type(e).__name__}: {e}",
            })
            pbar.update(1)
            return
        append_jsonl(config.ATTEMPTS_JSONL, attempts)
        rec = accepted if accepted is not None else attempts[-1]
        results.append({
            "id": row["id"],
            "ko": row["ko"],
            "def_ko_len": len(row["definition_ko"]),
            "latency_s": rec["latency_s"],
            "reasoning_tokens": rec.get("reasoning_tokens"),
            "completion_tokens": rec.get("completion_tokens"),
            "finish_reason": rec["finish_reason"],
            "reasoning": rec.get("reasoning", ""),
            "content": rec.get("content", ""),
            "accepted": accepted is not None,
            "attempts": len(attempts),
        })
        pbar.update(1)

    reset_fatal_event()
    tasks = [_one(r) for r in sample]
    gathered = await gather_bounded(tasks, concurrency=concurrency)
    pbar.close()
    wall = time.time() - batch_start
    fatal = [r for r in gathered if isinstance(r, FatalModelError)]
    if fatal:
        print(f"[probe] FATAL: {len(fatal)} FatalModelError; aborting.", file=sys.stderr)
        return 2

    # Persist per-row probe results.
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for r in results:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    successes = [r for r in results if r.get("accepted")]
    errors = [r for r in results if r.get("error")]
    latencies = [r["latency_s"] for r in successes]
    rtok = [r["reasoning_tokens"] for r in successes if r.get("reasoning_tokens") is not None]
    budget_hits = sum(1 for r in successes
                      if r.get("reasoning_tokens") is not None
                      and r["reasoning_tokens"] >= THINKING_TOKEN_BUDGET)
    length_truncations = sum(1 for r in successes if r.get("finish_reason") == "length")

    median_latency = statistics.median(latencies) if latencies else float("nan")
    max_latency = max(latencies) if latencies else float("nan")
    p90_latency = _percentile(latencies, 0.9) if latencies else float("nan")
    median_rtok = statistics.median(rtok) if rtok else float("nan")
    max_rtok = max(rtok) if rtok else 0
    rows_per_hour = (len(successes) / wall) * 3600 if wall > 0 else float("nan")
    full_hours = 10763 / rows_per_hour if rows_per_hour > 0 else float("nan")

    client_timeout = default_timeout(True)
    near_timeout = max_latency / client_timeout if latencies else 0.0

    # ---- Write the report ----
    lines: list[str] = []
    lines.append(f"# Thinking-on timing probe — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- Sample: {len(sample)} rows ({n_per_tercile} per definition-length tercile), seed={seed}")
    lines.append(f"- Concurrency: {concurrency}")
    lines.append(f"- Prompt: `{prompt_version}` · vLLM: {vllm_version} · "
                 f"thinking_token_budget accepted: {budget_accepted}")
    lines.append(f"- Successes: {len(successes)}/{len(sample)}; errors: {len(errors)}")
    lines.append(f"- Wall clock for this batch: {wall:.1f}s")
    lines.append("")
    lines.append("## Latency per request (s)")
    lines.append(f"- median: {median_latency:.1f}")
    lines.append(f"- p90: {p90_latency:.1f}")
    lines.append(f"- max: {max_latency:.1f}")
    lines.append(f"- client timeout: {client_timeout}s; "
                 f"max is {near_timeout:.0%} of timeout "
                 f"{'(WARN: within 10% of cap, consider raising LB cap)' if near_timeout >= TIMEOUT_WARN_FRACTION else ''}")
    lines.append("")
    lines.append("## Reasoning tokens")
    lines.append(f"- median: {median_rtok}")
    lines.append(f"- max: {max_rtok}")
    lines.append(f"- budget hits (reasoning_tokens >= {THINKING_TOKEN_BUDGET}): {budget_hits}/{len(successes)}")
    lines.append(f"- finish_reason=='length' truncations: {length_truncations}/{len(successes)}")
    lines.append("")
    lines.append("## Throughput")
    lines.append(f"- rows/hour at concurrency {concurrency}: {rows_per_hour:,.0f}")
    lines.append(f"- projected full run (10,763 rows): {full_hours:.1f} hours ({full_hours / 24:.2f} days)")
    lines.append("")
    lines.append("## Per-row")
    lines.append("| id | def_ko_len | latency_s | reasoning_tokens | finish_reason | attempts |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['id']} | - | ERROR | - | {r['error']} | - |")
            continue
        lines.append(f"| {r['id']} | {r['def_ko_len']} | {r['latency_s']:.1f} | "
                     f"{r.get('reasoning_tokens')} | {r['finish_reason']} | {r['attempts']} |")
    lines.append("")

    # 3 sample traces: shortest, median, longest by reasoning_tokens
    traced = sorted([r for r in successes if r.get("reasoning")], key=lambda r: r.get("reasoning_tokens") or 0)
    if traced:
        picks = []
        if len(traced) >= 1: picks.append(("shortest reasoning", traced[0]))
        if len(traced) >= 3: picks.append(("median reasoning", traced[len(traced) // 2]))
        if len(traced) >= 2: picks.append(("longest reasoning", traced[-1]))
        for label, pick in picks:
            lines.append(f"## Sample trace — {label} (id={pick['id']})")
            lines.append(f"- `ko`: {pick['ko']}")
            lines.append(f"- reasoning_tokens: {pick.get('reasoning_tokens')}; latency: {pick['latency_s']:.1f}s")
            lines.append("### Reasoning")
            lines.append("```")
            lines.append(pick["reasoning"][:4000])
            if len(pick["reasoning"]) > 4000:
                lines.append(f"\n... [{len(pick['reasoning']) - 4000} more chars truncated]")
            lines.append("```")
            lines.append("### Content")
            lines.append("```")
            lines.append(pick["content"][:1000])
            lines.append("```")
            lines.append("")

    if errors:
        lines.append("## Errors")
        for r in errors:
            lines.append(f"- id={r['id']}: {r['error']}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[probe] wrote {out_md}", file=sys.stderr)
    print(f"[probe] rows/hour={rows_per_hour:.0f} projected_full_run={full_hours:.1f}h", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--per-tercile", type=int, default=PROBE_N_PER_TERCILE)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--out-md", type=Path, default=config.RUNS / "thinking_probe.md")
    p.add_argument("--out-jsonl", type=Path, default=config.RUNS / "thinking_probe.jsonl")
    args = p.parse_args()
    return asyncio.run(run_probe(
        concurrency=args.concurrency,
        n_per_tercile=args.per_tercile,
        seed=args.seed,
        out_md=args.out_md,
        out_jsonl=args.out_jsonl,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
