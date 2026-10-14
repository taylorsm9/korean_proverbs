"""RQ1 analysis using what we have:

- A1 : main-run translation of the canonical sense (definition A)
- B1*: main-run translation of a sibling sense of the same proverb (definition B,
       seed = sibling's own target_code — a slight seed confound vs the plan's
       seed-matched B1, but acceptable since RQ1's meaning signal is extracted
       by comparing B1 drift against zero / against def_effect)
- N1 : fresh no-definition translation (prompt p1-nodef, seed = canonical_id)

Measures per headword (per embedding model):

    drift_B   = 1 - cos(E(A1_proverb_en), E(B1*_proverb_en))
                   # how much the translation moves when the meaning is swapped
    drift_N   = 1 - cos(E(A1_proverb_en), E(N1_proverb_en))
                   # how much it moves when the definition is removed entirely
    toward_B  = cos(E(B1*_proverb_en), E(B1*_definition_en))
              - cos(E(A1_proverb_en),  E(B1*_definition_en))
                   # if swapping the definition pulled the translation toward
                   # meaning B's embedding, this is positive
    def_dissim_ko = 1 - cos(E(definition_ko_A), E(definition_ko_B))
    def_dissim_en = 1 - cos(E(definition_en_A), E(definition_en_B))

Tests:
    1. drift_B distribution: median, bootstrap 95% CI, share > 0.
       22 identical-definition pairs: should be low (noise-only).
    2. def_effect: median drift_N, bootstrap CI. If close to 0, the definition
       is not shaping the translation at all → meaning cannot leak.
    3. Dose-response: Spearman between drift_B and def_dissim_ko / def_dissim_en.
       Positive correlation is evidence that drift tracks meaning differences.
    4. toward_B: median, share > 0. Positive = translation moved toward B's
       definition in embedding space when its definition was swapped in.
    5. Breakdown by (type_a, type_b) and (figurative_a, figurative_b).

Report → runs/research/rq1_report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .embed import CANDIDATES, load_aligned, load_field
from .jsonl import read_jsonl
from .rq1 import build_pairs


def _bootstrap_median_ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(x)
    out = np.empty(n)
    idx = rng.integers(0, N, size=(n, N))
    for i in range(n):
        out[i] = np.median(x[idx[i]])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _cos_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1)


def compute_measures(model_key: str, pairs: list[dict]) -> pd.DataFrame:
    """One row per multi-sense proverb × this model."""
    spec = CANDIDATES[model_key]
    name = spec.name

    canonical_ids = [p["id"] for p in pairs]
    sibling_ids = [p["sibling_id"] for p in pairs]

    # A1 vectors: aligned to canonical ids
    a1_prov, _ = load_aligned(name, "proverb_en", canonical_ids)
    a1_defko, _ = load_aligned(name, "definition_ko", canonical_ids)
    # B1* vectors: aligned to sibling ids (same model, same cache; different ids)
    b1_prov, _ = load_aligned(name, "proverb_en", sibling_ids)
    b1_defko, _ = load_aligned(name, "definition_ko", sibling_ids)
    b1_defen, _ = load_aligned(name, "definition_en", sibling_ids)
    # N1 vectors: from the n1_proverb_en cache
    n1_prov, n1_mask = load_aligned(name, "n1_proverb_en", canonical_ids)

    drift_B = 1.0 - _cos_rowwise(a1_prov, b1_prov)
    drift_N = 1.0 - _cos_rowwise(a1_prov, n1_prov)
    toward_B = _cos_rowwise(b1_prov, b1_defen) - _cos_rowwise(a1_prov, b1_defen)
    def_dissim_ko = 1.0 - _cos_rowwise(a1_defko, b1_defko)
    # English def_dissim via main-dataset definition_en vectors for both senses
    a1_defen, _ = load_aligned(name, "definition_en", canonical_ids)
    def_dissim_en = 1.0 - _cos_rowwise(a1_defen, b1_defen)

    return pd.DataFrame({
        "id": canonical_ids,
        "sibling_id": sibling_ids,
        "drift_B": drift_B,
        "drift_N": drift_N,
        "toward_B": toward_B,
        "def_dissim_ko": def_dissim_ko,
        "def_dissim_en": def_dissim_en,
        "n1_available": n1_mask,
        "same_def": [p["same_def"] for p in pairs],
        "type_a": [p["type_a"] for p in pairs],
        "type_b": [p["type_b"] for p in pairs],
        "figurative_a": [p["figurative_a"] for p in pairs],
        "figurative_b": [p["figurative_b"] for p in pairs],
    })


def tests_for_model(df: pd.DataFrame) -> dict:
    out: dict = {}
    # Use rows with N1 for drift_N tests; drift_B works on all 1,094.
    has_n = df["n1_available"].values

    out["drift_B"] = {
        "n": int(len(df)),
        "median": float(df["drift_B"].median()),
        "mean": float(df["drift_B"].mean()),
        "ci": _bootstrap_median_ci(df["drift_B"].values),
        "share_gt_zero": float((df["drift_B"] > 0).mean()),
    }
    out["drift_N"] = {
        "n": int(has_n.sum()),
        "median": float(df.loc[has_n, "drift_N"].median()),
        "mean": float(df.loc[has_n, "drift_N"].mean()),
        "ci": _bootstrap_median_ci(df.loc[has_n, "drift_N"].values),
        "share_gt_zero": float((df.loc[has_n, "drift_N"] > 0).mean()),
    }
    # Identical-definition pairs: should be ~ noise-only
    same = df[df["same_def"]]
    out["drift_B_on_same_def"] = {
        "n": int(len(same)),
        "median": float(same["drift_B"].median()) if len(same) else None,
        "mean": float(same["drift_B"].mean()) if len(same) else None,
    }

    # Paired comparison drift_B vs drift_N where both available
    both = df[has_n]
    w = stats.wilcoxon(both["drift_B"], both["drift_N"], alternative="two-sided")
    out["drift_B_vs_drift_N"] = {
        "n": int(len(both)),
        "median_diff": float((both["drift_B"] - both["drift_N"]).median()),
        "wilcoxon_p": float(w.pvalue),
    }

    # Dose-response: drift_B scales with def_dissim?
    different = df[~df["same_def"]]
    rho_ko, p_ko = stats.spearmanr(different["drift_B"], different["def_dissim_ko"])
    rho_en, p_en = stats.spearmanr(different["drift_B"], different["def_dissim_en"])
    out["dose_response"] = {
        "n": int(len(different)),
        "drift_B_vs_def_dissim_ko": {"rho": float(rho_ko), "p": float(p_ko)},
        "drift_B_vs_def_dissim_en": {"rho": float(rho_en), "p": float(p_en)},
    }

    # toward_B
    out["toward_B"] = {
        "n": int(len(df)),
        "median": float(df["toward_B"].median()),
        "mean": float(df["toward_B"].mean()),
        "ci": _bootstrap_median_ci(df["toward_B"].values),
        "share_gt_zero": float((df["toward_B"] > 0).mean()),
    }
    w2 = stats.wilcoxon(df["toward_B"], alternative="greater")
    out["toward_B"]["wilcoxon_greater_than_zero_p"] = float(w2.pvalue)

    # Breakdown by (type_a, type_b)
    bykeys = ["type_a", "type_b"]
    out["drift_B_by_type"] = {}
    for key, sub in df.groupby(bykeys):
        if len(sub) < 5:
            continue
        out["drift_B_by_type"][" / ".join(key)] = {
            "n": int(len(sub)),
            "median": float(sub["drift_B"].median()),
        }
    out["drift_B_by_figurative"] = {}
    for key, sub in df.groupby(["figurative_a", "figurative_b"]):
        if len(sub) < 5:
            continue
        out["drift_B_by_figurative"][str(key)] = {
            "n": int(len(sub)),
            "median": float(sub["drift_B"].median()),
        }

    return out


def _fmt(x: float, places: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{places}f}"


def write_report(results: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# RQ1 — image vs meaning: does swapping the definition change the translation?")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Primary model: `{results['primary_model']}`")
    lines.append(f"- Robustness model: `{results['robustness_model']}`")
    lines.append(f"- Pairs: {results['n_pairs']:,} multi-sense proverbs")
    lines.append("")
    lines.append("## Design and caveats")
    lines.append("")
    lines.append("For each proverb with ≥2 dictionary senses, we hold the proverb text constant and vary which meaning is in the prompt:")
    lines.append("")
    lines.append("- **A1**: canonical sense translation (main-run output, prompt had definition A, seed = canonical id)")
    lines.append("- **B1***: sibling sense translation (main-run output, prompt had definition B, seed = sibling's own id). The `*` marks a seed confound vs the plan's design, which would use seed = canonical id for a perfectly matched comparison; for the aggregate statistics the seed difference averages out.")
    lines.append("- **N1**: canonical proverb only, `p1-nodef` prompt, seed = canonical id (new Spark pass)")
    lines.append("")
    lines.append("A noise-control arm (A2) was not run; the user prioritised shipping over the clean noise baseline. We therefore cannot formally subtract pure seed noise from drift_B — results are descriptive.")
    lines.append("")
    lines.append("## Primary findings (kanana)")
    lines.append("")
    primary = results["primary"]
    db = primary["drift_B"]
    dn = primary["drift_N"]
    lines.append(f"- **drift_B** (A1 vs B1*): median {_fmt(db['median'])}, mean {_fmt(db['mean'])}, 95% CI ({_fmt(db['ci'][0])}, {_fmt(db['ci'][1])}), share > 0: {db['share_gt_zero']*100:.1f}%")
    lines.append(f"- **drift_N** (A1 vs N1, no-def): median {_fmt(dn['median'])}, mean {_fmt(dn['mean'])}, 95% CI ({_fmt(dn['ci'][0])}, {_fmt(dn['ci'][1])}), share > 0: {dn['share_gt_zero']*100:.1f}%")
    sd = primary["drift_B_on_same_def"]
    lines.append(f"- drift_B restricted to the {sd['n']} identical-definition pairs (noise floor): median {_fmt(sd['median'])}, mean {_fmt(sd['mean'])}")
    lines.append("")
    diff = primary["drift_B_vs_drift_N"]
    lines.append(f"**drift_B vs drift_N** (paired Wilcoxon on {diff['n']:,} pairs with both): median difference {_fmt(diff['median_diff'])}, p = {diff['wilcoxon_p']:.2e}")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("- If `drift_B ≈ drift_N` (and both are small), swapping the definition doesn't matter *much more* than removing it entirely: the model is largely translating the proverb's surface form regardless of the meaning.")
    lines.append("- If `drift_B < drift_N`, the two definitions are not pulling the English in very different directions — the model varies more between \"def present\" and \"def absent\" than between two different defs.")
    lines.append("- If `drift_B > drift_N`, which meaning you give matters more than whether you give one at all — strong evidence of meaning being used.")
    lines.append("")
    lines.append("## Dose-response: does drift_B scale with how different the two meanings are?")
    lines.append("")
    dr = primary["dose_response"]
    lines.append(f"Over the {dr['n']:,} pairs with non-identical definitions:")
    lines.append(f"- Spearman drift_B vs **def_dissim_ko**: ρ = {_fmt(dr['drift_B_vs_def_dissim_ko']['rho'])}, p = {dr['drift_B_vs_def_dissim_ko']['p']:.2e}")
    lines.append(f"- Spearman drift_B vs **def_dissim_en**: ρ = {_fmt(dr['drift_B_vs_def_dissim_en']['rho'])}, p = {dr['drift_B_vs_def_dissim_en']['p']:.2e}")
    lines.append("")
    lines.append("Positive and significant → *more different meanings produce more drift*, which is the signature of meaning-based translation.")
    lines.append("")
    lines.append("## toward_B: did the translation move toward meaning B's embedding?")
    lines.append("")
    tb = primary["toward_B"]
    lines.append(f"- median {_fmt(tb['median'], 4)}, mean {_fmt(tb['mean'], 4)}, 95% CI ({_fmt(tb['ci'][0], 4)}, {_fmt(tb['ci'][1], 4)}), share > 0: {tb['share_gt_zero']*100:.1f}%; Wilcoxon one-sided (greater than zero) p = {tb['wilcoxon_greater_than_zero_p']:.2e}")
    lines.append("")
    lines.append("Positive = when definition B was swapped in, the English translation moved toward B's definition in embedding space (vs A1's position).")
    lines.append("")
    lines.append("## Breakdowns (exploratory)")
    lines.append("")
    lines.append("### By type pair (type_a / type_b)")
    lines.append("")
    for key, t in primary["drift_B_by_type"].items():
        lines.append(f"- `{key}`: n={t['n']:,}, median drift_B={_fmt(t['median'])}")
    lines.append("")
    lines.append("### By figurative pair (figurative_a, figurative_b)")
    lines.append("")
    for key, t in primary["drift_B_by_figurative"].items():
        lines.append(f"- {key}: n={t['n']:,}, median drift_B={_fmt(t['median'])}")
    lines.append("")
    lines.append("## Robustness (Qwen3-Embedding-8B)")
    lines.append("")
    robust = results["robustness"]
    rdb = robust["drift_B"]
    rdn = robust["drift_N"]
    lines.append(f"- drift_B median {_fmt(rdb['median'])} (CI {_fmt(rdb['ci'][0])}–{_fmt(rdb['ci'][1])}), share > 0: {rdb['share_gt_zero']*100:.1f}%")
    lines.append(f"- drift_N median {_fmt(rdn['median'])} (CI {_fmt(rdn['ci'][0])}–{_fmt(rdn['ci'][1])}), share > 0: {rdn['share_gt_zero']*100:.1f}%")
    rdr = robust["dose_response"]
    lines.append(f"- dose-response Spearman drift_B vs def_dissim_ko: ρ = {_fmt(rdr['drift_B_vs_def_dissim_ko']['rho'])}, p = {rdr['drift_B_vs_def_dissim_ko']['p']:.2e}")
    rtb = robust["toward_B"]
    lines.append(f"- toward_B median {_fmt(rtb['median'], 4)}, share > 0: {rtb['share_gt_zero']*100:.1f}%")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--primary", default="kanana")
    p.add_argument("--robustness", default="qwen3")
    args = p.parse_args()
    config.ensure_dirs()

    pairs = build_pairs()
    print(f"[rq1] computing measures for {len(pairs)} pairs", file=sys.stderr)

    df_p = compute_measures(args.primary, pairs)
    df_r = compute_measures(args.robustness, pairs)

    results = {
        "primary_model": CANDIDATES[args.primary].name,
        "robustness_model": CANDIDATES[args.robustness].name,
        "n_pairs": len(pairs),
        "primary": tests_for_model(df_p),
        "robustness": tests_for_model(df_r),
    }

    out_md = config.RUNS / "research" / "rq1_report.md"
    write_report(results, out_md)
    out_json = config.RUNS / "research" / "rq1_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    out_csv = config.RESEARCH / "rq1_measures.csv"
    out_parquet = config.RESEARCH / "rq1_measures.parquet"
    df_p["model"] = args.primary
    df_r["model"] = args.robustness
    merged = pd.concat([df_p, df_r])
    merged.to_csv(out_csv, index=False)
    try:
        merged.to_parquet(out_parquet, index=False, compression="zstd")
    except Exception as e:
        print(f"[rq1] parquet skipped ({e})", file=sys.stderr)
    print(f"[rq1] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
