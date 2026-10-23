"""Thinking-on vs thinking-off analysis ("Does thinking change the translation?" in the README).

Arms, all on the canonical senses of the 1,094 multi-sense proverbs (`rq1.build_pairs`):

- A1t : main thinking-on translation (`proverb_en_thinking`)
- F   : thinking-on re-seed of a 300-row sample (`thinking_research reseed`)
- N1t : thinking-on, no definition in the prompt (`thinking_research nodef`)
- A1  : main thinking-off translation (`proverb_en`)
- N1  : thinking-off, no definition (the RQ1 N1 arm)

Measures per proverb (per embedding model), all 1 - cos between English proverb translations:

    reseed_shift   = A1t vs F     # chance alone; NaN outside the 300-row sample
    nodef_shift    = A1t vs N1t   # definition removed, thinking on
    mode_shift     = A1t vs A1    # thinking on vs off (also differs in sampling)
    nodef_shift_off = A1 vs N1    # definition removed, thinking off (RQ1 drift_N)

The F and N1t outputs are embedded first; cached rows are skipped, so this only needs the
embedding server (EMBED_BASE_URL) the first time. Writes data/research/thinking_measures.*
and prints the summary.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .embed import CANDIDATES, embed_base_url, embed_texts, load_aligned
from .jsonl import read_jsonl
from .rq1 import build_pairs
from .rq1_analyze import _bootstrap_median_ci, _cos_rowwise

# research output file -> embedding cache field
ARMS = {
    "thinking_reseed.jsonl": "f_reseed_proverb_en",
    "thinking_nodef.jsonl": "g_nodef_proverb_en",
}


def embed_arms(model_keys: list[str]) -> None:
    for fname, cache_name in ARMS.items():
        rows = list(read_jsonl(config.RESEARCH / fname))
        for mk in model_keys:
            embed_texts(mk, [r["proverb_en"] for r in rows], [r["id"] for r in rows],
                        cache_name, remote_url=embed_base_url())


def compute_measures(model_key: str, ids: list[str]) -> pd.DataFrame:
    """One row per multi-sense proverb × this model."""
    name = CANDIDATES[model_key].name
    a1t, _ = load_aligned(name, "proverb_en_thinking", ids)
    a1, _ = load_aligned(name, "proverb_en", ids)
    f, in_sample = load_aligned(name, "f_reseed_proverb_en", ids)
    n1t, _ = load_aligned(name, "g_nodef_proverb_en", ids)
    n1, _ = load_aligned(name, "n1_proverb_en", ids)

    reseed_shift = 1.0 - _cos_rowwise(a1t, f)
    reseed_shift[~in_sample] = np.nan
    return pd.DataFrame({
        "id": ids,
        "in_sample": in_sample,
        "reseed_shift": reseed_shift,
        "nodef_shift": 1.0 - _cos_rowwise(a1t, n1t),
        "mode_shift": 1.0 - _cos_rowwise(a1t, a1),
        "nodef_shift_off": 1.0 - _cos_rowwise(a1, n1),
    })


def summarize(df: pd.DataFrame) -> list[str]:
    s = df[df["in_sample"]]
    lines = [f"  300-row sample (n={len(s)}): median shift [95% CI]"]
    for col in ("reseed_shift", "nodef_shift", "mode_shift"):
        lo, hi = _bootstrap_median_ci(s[col].values)
        lines.append(f"    {col:16s} {s[col].median():.3f} [{lo:.3f}, {hi:.3f}]")
    for col in ("mode_shift", "nodef_shift"):
        d = s[col] - s["reseed_shift"]
        p = stats.wilcoxon(s[col], s["reseed_shift"]).pvalue
        lines.append(f"    {col} - reseed_shift: median paired diff {d.median():+.3f}, Wilcoxon p={p:.1e}")
    lines.append(f"  All rows (n={len(df)})")
    for col in ("nodef_shift", "nodef_shift_off"):
        lo, hi = _bootstrap_median_ci(df[col].values)
        lines.append(f"    {col:16s} {df[col].median():.3f} [{lo:.3f}, {hi:.3f}]")
    d = df["nodef_shift"] - df["nodef_shift_off"]
    p = stats.wilcoxon(df["nodef_shift"], df["nodef_shift_off"]).pvalue
    lines.append(f"    nodef_shift - nodef_shift_off: median paired diff {d.median():+.3f}, Wilcoxon p={p:.1e}")
    return lines


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["kanana", "qwen3"])
    args = p.parse_args()
    config.ensure_dirs()

    embed_arms(args.models)
    ids = [pair["id"] for pair in build_pairs()]
    frames = []
    for mk in args.models:
        df = compute_measures(mk, ids)
        print(f"[thinking] {CANDIDATES[mk].name}", file=sys.stderr)
        print("\n".join(summarize(df)), file=sys.stderr)
        df["model"] = mk
        frames.append(df)

    merged = pd.concat(frames)
    merged.to_csv(config.RESEARCH / "thinking_measures.csv", index=False)
    merged.to_parquet(config.RESEARCH / "thinking_measures.parquet", index=False, compression="zstd")
    print(f"[thinking] wrote {config.RESEARCH / 'thinking_measures.*'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
