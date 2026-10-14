"""RQ2 (plan.md Step 7): How far is the image from the meaning, and what's
lost in translation?

Measures per row, computed for the primary embedding model (kanana) and the
robustness model (qwen3):

    d_ko   = 1 - cos(E(ko), E(definition_ko))
    d_en   = 1 - cos(E(proverb_en), E(definition_en))
    rank_ko = rank of own definition_ko among all 10,763 (1 = closest)
    rank_en = same in English
    asym   = log(rank_en) - log(rank_ko)    # primary
    z_ko, z_en, asym_z  = language-centered distances
    xfer_proverb = cos(E(ko), E(proverb_en))
    xfer_def     = cos(E(definition_ko), E(definition_en))

Tests (per plan Step 7 RQ2):
    1. Does distance detect figurative proverbs? AUC of d_ko, log(rank_ko).
       Same for d_en, log(rank_en). Mann-Whitney U + medians.
    2. What translation does to the gap: median asym, bootstrap 95% CI,
       share > 0; compare asym for figurative vs non-figurative (Mann-Whitney).
       Robustness: repeat with asym_z.
    3. Is the ranking preserved across languages? Spearman d_ko vs d_en,
       rank_ko vs rank_en.
    4. literal_gloss subset: repeat 1 and 2.
    5. Breakdowns by type (일반어 / 북한어). Judge-flag breakdown is skipped
       (judge pass not run).
    6. Robustness: repeat 1 and 2 with qwen3; appendix.
    7. Plots → runs/research/figures/.

Writes:
    data/research/proverb_distances.csv  (the per-row artifact for the art project)
    runs/research/rq2_report.md
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
from .embed import CANDIDATES, load_aligned
from .jsonl import read_jsonl


# ---------- primitive stats ----------

def _bootstrap_ci(x: np.ndarray, stat_fn, *, n: int = 10_000, seed: int = 0,
                  low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(x)
    boots = np.empty(n, dtype=np.float64)
    idx_all = rng.integers(0, N, size=(n, N))
    for i in range(n):
        boots[i] = stat_fn(x[idx_all[i]])
    return float(np.percentile(boots, low)), float(np.percentile(boots, high))


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score))


def _auc_with_ci(score: np.ndarray, label: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    N = len(score)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        try:
            boots[i] = _auc(score[idx], label[idx])
        except ValueError:
            boots[i] = np.nan
    boots = boots[np.isfinite(boots)]
    return _auc(score, label), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------- core measures ----------

def compute_distances(model_name: str, source: list[dict]) -> pd.DataFrame:
    """Return a DataFrame with d_ko, d_en, rank_ko, rank_en, asym and z-score
    variants for every row in `source`, using the given embedding model."""
    ids = [r["id"] for r in source]
    pko, _ = load_aligned(model_name, "ko", ids)
    pdk, _ = load_aligned(model_name, "definition_ko", ids)
    pen, _ = load_aligned(model_name, "proverb_en", ids)
    pde, _ = load_aligned(model_name, "definition_en", ids)

    # Pair similarities: each row's own (ko, def_ko) and (proverb_en, def_en)
    sim_ko_self = (pko * pdk).sum(axis=1)
    sim_en_self = (pen * pde).sum(axis=1)
    d_ko = 1.0 - sim_ko_self
    d_en = 1.0 - sim_en_self

    # Full retrieval ranks (10k × 10k ≈ 460 MB at 1024-dim).
    t0 = time.time()
    sim_ko_all = pko @ pdk.T
    order_ko = np.argsort(-sim_ko_all, axis=1)
    gold = np.arange(len(ids))
    rank_ko = (np.argmax(order_ko == gold[:, None], axis=1) + 1).astype(np.int32)
    print(f"[rq2 {model_name}] rank_ko done ({time.time()-t0:.1f}s)", file=sys.stderr)

    t0 = time.time()
    sim_en_all = pen @ pde.T
    order_en = np.argsort(-sim_en_all, axis=1)
    rank_en = (np.argmax(order_en == gold[:, None], axis=1) + 1).astype(np.int32)
    print(f"[rq2 {model_name}] rank_en done ({time.time()-t0:.1f}s)", file=sys.stderr)

    log_rank_ko = np.log(rank_ko.astype(np.float64))
    log_rank_en = np.log(rank_en.astype(np.float64))
    asym = log_rank_en - log_rank_ko

    # Within-language z-scores (robust alternative to raw d_).
    z_ko = (d_ko - d_ko.mean()) / d_ko.std(ddof=1)
    z_en = (d_en - d_en.mean()) / d_en.std(ddof=1)
    asym_z = z_en - z_ko

    xfer_proverb = (pko * pen).sum(axis=1)
    xfer_def = (pdk * pde).sum(axis=1)

    df = pd.DataFrame({
        "id": ids,
        "d_ko": d_ko,
        "d_en": d_en,
        "rank_ko": rank_ko,
        "rank_en": rank_en,
        "log_rank_ko": log_rank_ko,
        "log_rank_en": log_rank_en,
        "asym": asym,
        "z_ko": z_ko,
        "z_en": z_en,
        "asym_z": asym_z,
        "xfer_proverb": xfer_proverb,
        "xfer_def": xfer_def,
    })
    meta = pd.DataFrame({
        "id": [r["id"] for r in source],
        "figurative": [r.get("figurative", False) for r in source],
        "literal_gloss": [r.get("literal_gloss", False) for r in source],
        "type": [r["type"] for r in source],
        "is_canonical_sense": [r.get("is_canonical_sense", True) for r in source],
    })
    return df.merge(meta, on="id")


# ---------- test batteries ----------

def test_figurative_auc(df: pd.DataFrame, label_col: str = "figurative") -> dict:
    out: dict = {}
    label = df[label_col].astype(int).values
    for score_name, direction in [
        ("d_ko", +1), ("d_en", +1),
        ("log_rank_ko", +1), ("log_rank_en", +1),
    ]:
        s = direction * df[score_name].values
        auc, lo, hi = _auc_with_ci(s, label, n_boot=1000)
        u, p = stats.mannwhitneyu(df.loc[df[label_col], score_name],
                                  df.loc[~df[label_col], score_name], alternative="two-sided")
        pos = df.loc[df[label_col], score_name]
        neg = df.loc[~df[label_col], score_name]
        out[score_name] = {
            "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
            "median_pos": float(pos.median()),
            "median_neg": float(neg.median()),
            "u": float(u), "p": float(p),
            "n_pos": int(df[label_col].sum()), "n_neg": int((~df[label_col]).sum()),
        }
    return out


def test_translation_gap(df: pd.DataFrame) -> dict:
    out: dict = {}
    asym = df["asym"].values
    asym_z = df["asym_z"].values
    out["asym"] = {
        "median": float(np.median(asym)),
        "mean": float(asym.mean()),
        "share_gt_0": float((asym > 0).mean()),
        "ci": _bootstrap_ci(asym, np.median, n=10_000),
    }
    out["asym_z"] = {
        "median": float(np.median(asym_z)),
        "mean": float(asym_z.mean()),
        "share_gt_0": float((asym_z > 0).mean()),
        "ci": _bootstrap_ci(asym_z, np.median, n=10_000),
    }
    for score_name in ("asym", "asym_z"):
        pos = df.loc[df["figurative"], score_name]
        neg = df.loc[~df["figurative"], score_name]
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        out[f"{score_name}_by_figurative"] = {
            "median_fig": float(pos.median()),
            "median_nonfig": float(neg.median()),
            "u": float(u), "p": float(p),
            "n_fig": int(df["figurative"].sum()),
            "n_nonfig": int((~df["figurative"]).sum()),
        }
    return out


def test_cross_lang_rank_corr(df: pd.DataFrame) -> dict:
    rho_d, p_d = stats.spearmanr(df["d_ko"], df["d_en"])
    rho_r, p_r = stats.spearmanr(df["rank_ko"], df["rank_en"])
    return {
        "spearman_d": {"rho": float(rho_d), "p": float(p_d)},
        "spearman_rank": {"rho": float(rho_r), "p": float(p_r)},
    }


def test_type_breakdown(df: pd.DataFrame) -> dict:
    out: dict = {}
    for t, sub in df.groupby("type"):
        if len(sub) < 2:
            continue
        out[t] = {
            "n": int(len(sub)),
            "median_asym": float(sub["asym"].median()),
            "mean_asym": float(sub["asym"].mean()),
            "share_asym_gt_0": float((sub["asym"] > 0).mean()),
        }
    return out


# ---------- plots ----------

def make_plots(df: pd.DataFrame, model_name: str, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    slug = model_name.replace("/", "__")

    # 1. Histograms of d_ko and d_en by figurative
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ["d_ko", "d_en"], ["d_ko (Korean)", "d_en (English)"]):
        ax.hist(df.loc[~df["figurative"], col], bins=40, alpha=0.5, label="non-figurative", density=True)
        ax.hist(df.loc[df["figurative"], col], bins=40, alpha=0.5, label="figurative", density=True)
        ax.set_xlabel(col); ax.set_title(title); ax.legend()
    fig.suptitle(f"d_ko / d_en by figurative — {model_name}")
    p = out_dir / f"{slug}__hist_d_by_figurative.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    # 2. Scatter log(rank_ko) vs log(rank_en) coloured by figurative
    fig, ax = plt.subplots(figsize=(6, 6))
    jitter = np.random.default_rng(0).normal(0, 0.05, size=(len(df), 2))
    x = df["log_rank_ko"].values + jitter[:, 0]
    y = df["log_rank_en"].values + jitter[:, 1]
    colors = np.where(df["figurative"].values, "C1", "C0")
    ax.scatter(x, y, s=2, c=colors, alpha=0.4)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("log(rank_ko)"); ax.set_ylabel("log(rank_en)")
    ax.set_title(f"log-rank KO vs EN (orange=figurative) — {model_name}")
    p = out_dir / f"{slug}__scatter_logranks.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    # 3. asym distribution by figurative
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df.loc[~df["figurative"], "asym"], bins=50, alpha=0.5, label="non-figurative", density=True)
    ax.hist(df.loc[df["figurative"], "asym"], bins=50, alpha=0.5, label="figurative", density=True)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("asym = log(rank_en) − log(rank_ko)")
    ax.set_title(f"asym by figurative (positive = EN farther from meaning than KO) — {model_name}")
    ax.legend()
    p = out_dir / f"{slug}__hist_asym_by_figurative.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)
    return paths


# ---------- main ----------

def _fmt(x: float, places: int = 3) -> str:
    return f"{x:+.{places}f}" if (x >= 0 and x != 0) else f"{x:.{places}f}"


def write_report(results: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# RQ2 — image vs meaning, in Korean and English")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Primary model: `{results['primary_model']}`")
    lines.append(f"- Robustness model: `{results['robustness_model']}`")
    lines.append(f"- N rows: {results['n_rows']:,} ({results['n_canonical']:,} canonical)")
    lines.append("")
    lines.append("## Primary results — Test 1: can distance detect figurative proverbs?")
    lines.append("")
    lines.append("AUC bands and 95% bootstrap CI for predicting `figurative == true` from each score.")
    lines.append("")
    for lang in ("ko", "en"):
        for metric in (f"d_{lang}", f"log_rank_{lang}"):
            t = results["primary"]["test1"][metric]
            lines.append(f"- **{metric}**: AUC {t['auc']:.3f} (95% CI {t['auc_ci_lo']:.3f}–{t['auc_ci_hi']:.3f}); "
                         f"medians fig={t['median_pos']:.3f} vs non-fig={t['median_neg']:.3f}; "
                         f"Mann-Whitney p={t['p']:.2e}")
    lines.append("")
    lines.append("## Primary results — Test 2: what translation does to the gap")
    lines.append("")
    for name in ("asym", "asym_z"):
        t = results["primary"]["test2"][name]
        lo, hi = t["ci"]
        lines.append(f"- **{name}**: median {_fmt(t['median'])} (95% CI {_fmt(lo)} — {_fmt(hi)}); "
                     f"mean {_fmt(t['mean'])}; share > 0: {t['share_gt_0']*100:.1f}%")
    lines.append("")
    lines.append("Figurative vs non-figurative breakdown:")
    lines.append("")
    for score in ("asym", "asym_z"):
        t = results["primary"]["test2"][f"{score}_by_figurative"]
        lines.append(f"- **{score}**: fig median {_fmt(t['median_fig'])} vs non-fig {_fmt(t['median_nonfig'])}; "
                     f"n={t['n_fig']} / {t['n_nonfig']}; Mann-Whitney p={t['p']:.2e}")
    lines.append("")
    lines.append("Interpretation note: `asym > 0` means the English translation is **farther from its meaning** than the Korean original is "
                 "(in rank terms over the same corpus). `asym < 0` means the English is closer to its meaning.")
    lines.append("")
    lines.append("## Primary results — Test 3: cross-language rank correlation")
    lines.append("")
    t = results["primary"]["test3"]
    lines.append(f"- Spearman d_ko vs d_en: ρ = {t['spearman_d']['rho']:.3f}, p = {t['spearman_d']['p']:.2e}")
    lines.append(f"- Spearman rank_ko vs rank_en: ρ = {t['spearman_rank']['rho']:.3f}, p = {t['spearman_rank']['p']:.2e}")
    lines.append("")
    lines.append("## Test 4: literal_gloss subset (exploratory)")
    lines.append("")
    t4 = results["primary"]["test4"]
    if t4 is None:
        lines.append("No rows in literal_gloss subset.")
    else:
        lines.append(f"- n = {t4['n']:,}")
        lines.append(f"- Test 1 AUC d_ko = {t4['test1']['d_ko']['auc']:.3f}; d_en = {t4['test1']['d_en']['auc']:.3f}")
        a = t4["test2"]["asym"]
        lines.append(f"- Test 2 median asym = {_fmt(a['median'])} (CI {_fmt(a['ci'][0])} — {_fmt(a['ci'][1])})")
    lines.append("")
    lines.append("## Test 5: breakdowns (exploratory)")
    lines.append("")
    lines.append("By `type` (일반어 / 북한어):")
    lines.append("")
    for t_name, t5 in results["primary"]["test5_by_type"].items():
        lines.append(f"- `{t_name}`: n={t5['n']:,}, median asym={_fmt(t5['median_asym'])}, "
                     f"share > 0: {t5['share_asym_gt_0']*100:.1f}%")
    lines.append("")
    lines.append("(Judge-flag breakdown skipped: judge pass was not run on this dataset.)")
    lines.append("")
    lines.append("## Test 6: robustness check with Qwen3-Embedding-8B")
    lines.append("")
    lines.append("Primary test 1 (figurative AUC) and test 2 (translation gap) with the robustness model:")
    lines.append("")
    for metric in ("d_ko", "d_en"):
        t = results["robustness"]["test1"][metric]
        lines.append(f"- **{metric}**: AUC {t['auc']:.3f} (CI {t['auc_ci_lo']:.3f}–{t['auc_ci_hi']:.3f})")
    r2 = results["robustness"]["test2"]["asym"]
    lines.append(f"- **asym** median {_fmt(r2['median'])} (CI {_fmt(r2['ci'][0])} — {_fmt(r2['ci'][1])}); share > 0: {r2['share_gt_0']*100:.1f}%")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for p in results["plots"]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("`definition_en` and `proverb_en` are machine translations, so English-side distances also reflect translation quality (not just the proverb's inherent image/meaning gap).")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tests(df: pd.DataFrame) -> dict:
    out: dict = {}
    out["test1"] = test_figurative_auc(df, label_col="figurative")
    out["test2"] = test_translation_gap(df)
    out["test3"] = test_cross_lang_rank_corr(df)
    sub = df[df["literal_gloss"]]
    out["test4"] = None if len(sub) < 10 else {
        "n": int(len(sub)),
        "test1": test_figurative_auc(sub, label_col="figurative"),
        "test2": test_translation_gap(sub),
    }
    out["test5_by_type"] = test_type_breakdown(df)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--primary", default="kanana")
    p.add_argument("--robustness", default="qwen3")
    args = p.parse_args()
    config.ensure_dirs()
    (config.RESEARCH).mkdir(parents=True, exist_ok=True)

    source = list(read_jsonl(config.PROVERBS_JSONL))
    primary_name = CANDIDATES[args.primary].name
    robust_name = CANDIDATES[args.robustness].name

    print(f"[rq2] primary = {primary_name}", file=sys.stderr)
    df_p = compute_distances(primary_name, source)
    print(f"[rq2] robustness = {robust_name}", file=sys.stderr)
    df_r = compute_distances(robust_name, source)

    tests_p = run_tests(df_p)
    tests_r = run_tests(df_r)

    plots = make_plots(df_p, primary_name, config.FIGURES)
    plot_paths = [str(p.relative_to(config.ROOT)) for p in plots]

    results = {
        "primary_model": primary_name, "robustness_model": robust_name,
        "n_rows": len(df_p), "n_canonical": int(df_p["is_canonical_sense"].sum()),
        "primary": tests_p, "robustness": tests_r,
        "plots": plot_paths,
    }

    # Save the per-row CSV for the art project.
    art_cols_primary = {c: f"{c}__{args.primary}" for c in
                        ["d_ko", "d_en", "rank_ko", "rank_en", "asym", "z_ko", "z_en", "xfer_proverb", "xfer_def"]}
    art_cols_robust = {c: f"{c}__{args.robustness}" for c in art_cols_primary}
    merged = (df_p.rename(columns=art_cols_primary)
                 .merge(df_r.rename(columns=art_cols_robust)[["id"] + list(art_cols_robust.values())], on="id"))
    base_cols = ["id", "figurative", "literal_gloss", "type", "is_canonical_sense"]
    # Add the Korean/English texts for the art export, if available.
    rows_by_id = {r["id"]: r for r in source}
    merged["ko"] = merged["id"].map(lambda i: rows_by_id[i]["ko"])
    translations = {r["id"]: r for r in read_jsonl(config.TRANSLATIONS_JSONL)}
    merged["proverb_en"] = merged["id"].map(lambda i: translations.get(i, {}).get("proverb_en", ""))
    cols = base_cols + ["ko", "proverb_en"] + list(art_cols_primary.values()) + list(art_cols_robust.values())
    out_csv = config.RESEARCH / "proverb_distances.csv"
    out_parquet = config.RESEARCH / "proverb_distances.parquet"
    merged[cols].to_csv(out_csv, index=False)
    try:
        merged[cols].to_parquet(out_parquet, index=False, compression="zstd")
        print(f"[rq2] wrote {out_csv} / {out_parquet}", file=sys.stderr)
    except Exception as e:
        print(f"[rq2] parquet skipped ({e}); wrote {out_csv}", file=sys.stderr)

    out_md = config.RUNS / "research" / "rq2_report.md"
    write_report(results, out_md)
    print(f"[rq2] wrote {out_md}", file=sys.stderr)

    # Also save a JSON of all numeric results for programmatic access.
    (config.RUNS / "research" / "rq2_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
