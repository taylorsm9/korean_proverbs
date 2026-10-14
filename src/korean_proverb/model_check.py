"""Step 7.0: Quick embedding-model check.

For each candidate model (already embedded by `korean_proverb.embed`), compute:

1. Proverb → meaning retrieval MRR
   - Korean: rank all `definition_ko` by cosine similarity to `ko`; MRR of own definition.
   - English: rank all `definition_en` by cosine similarity to `proverb_en`; MRR of own definition.
   - Reported on all rows AND on the 9,612 canonical-sense rows only.
2. Character-bigram BM25 baseline on the same tasks.
3. Synonym sanity check: ROC-AUC of `synonym_ids` pairs against an equal number of random pairs.

Writes `runs/research/model_check.md`. Picks the primary model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .embed import CANDIDATES, load_aligned
from .jsonl import read_jsonl


# ---------- metrics ----------

def _mrr(sim: np.ndarray, gold: np.ndarray) -> tuple[float, np.ndarray]:
    """Given (N, M) similarity matrix and gold column index per row, return
    (MRR, per-row rank). Rank = 1 means the gold is top."""
    # argsort descending along axis=1
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == gold[:, None], axis=1) + 1
    return float((1.0 / ranks).mean()), ranks


def _subset_mrr(ranks: np.ndarray, mask: np.ndarray) -> float:
    """Compute MRR over a subset of rows (same candidate set)."""
    if mask.sum() == 0:
        return 0.0
    return float((1.0 / ranks[mask]).mean())


# ---------- BM25 baseline (character bigrams) ----------

def _char_bigrams(s: str) -> list[str]:
    s = s.replace(" ", "")
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) >= 2 else [s]


def _bm25_mrr(queries: list[str], corpus: list[str], gold: np.ndarray) -> tuple[float, np.ndarray]:
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [_char_bigrams(t) for t in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    N, M = len(queries), len(corpus)
    ranks = np.zeros(N, dtype=np.int32)
    for i, q in enumerate(queries):
        scores = bm25.get_scores(_char_bigrams(q))
        order = np.argsort(-scores)
        ranks[i] = int(np.where(order == gold[i])[0][0]) + 1
    return float((1.0 / ranks).mean()), ranks


# ---------- synonym AUC ----------

def _syn_pairs(rows: list[dict], id_to_idx: dict[str, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in rows:
        for sid in r.get("synonym_ids", []):
            j = id_to_idx.get(sid)
            if j is not None:
                out.append((id_to_idx[r["id"]], j))
    return out


def _rand_pairs(n: int, N: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    while len(out) < n:
        a = int(rng.integers(0, N))
        b = int(rng.integers(0, N))
        if a != b:
            out.append((a, b))
    return out


def _cos(vecs: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    a = vecs[[p[0] for p in pairs]]
    b = vecs[[p[1] for p in pairs]]
    return (a * b).sum(axis=1)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s))


# ---------- main check ----------

def check_model(model_key: str, max_rows: int | None = None, baseline: bool = True) -> dict:
    spec = CANDIDATES[model_key]
    source = list(read_jsonl(config.PROVERBS_JSONL))
    translations = {r["id"]: r for r in read_jsonl(config.TRANSLATIONS_JSONL)}
    if max_rows is not None:
        source = source[:max_rows]
    ids_order = [r["id"] for r in source]

    # Load vectors aligned to proverbs.jsonl order; masks tell us which rows
    # have a cached embedding for this field (should be all).
    pko, m_ko = load_aligned(spec.name, "ko", ids_order)
    pdk, m_dk = load_aligned(spec.name, "definition_ko", ids_order)
    pen, m_en = load_aligned(spec.name, "proverb_en", ids_order)
    pde, m_de = load_aligned(spec.name, "definition_en", ids_order)

    N = len(ids_order)
    canonical_mask = np.array([r.get("is_canonical_sense", True) for r in source], dtype=bool)
    print(f"[check {spec.name}] N={N}, canonical={canonical_mask.sum():,}, "
          f"EN coverage={m_en.sum():,}/{N:,}", file=sys.stderr)

    out: dict = {"model": spec.name, "license": spec.license, "n": N,
                 "canonical": int(canonical_mask.sum()), "n_en": int(m_en.sum())}

    # Korean MRR: ko → definition_ko. Both fields fully populated.
    gold = np.arange(N, dtype=np.int32)
    t0 = time.time()
    sim_ko = pko @ pdk.T
    mrr_ko, ranks_ko = _mrr(sim_ko, gold)
    out["mrr_ko_all"] = mrr_ko
    out["mrr_ko_canonical"] = _subset_mrr(ranks_ko, canonical_mask)
    print(f"[check {spec.name}] ko MRR={mrr_ko:.4f} ({time.time()-t0:.1f}s)", file=sys.stderr)

    # English MRR: proverb_en → definition_en.
    # Restrict both the query set and the candidate set to rows that have
    # both EN embeddings, so the ranks are over a self-consistent corpus.
    has_en = m_en & m_de
    if has_en.sum() == 0:
        out["mrr_en_all"] = None
        out["mrr_en_canonical"] = None
    else:
        en_pen = pen[has_en]
        en_pde = pde[has_en]
        local_gold = np.arange(int(has_en.sum()), dtype=np.int32)
        t0 = time.time()
        sim_en = en_pen @ en_pde.T
        mrr_en, ranks_en = _mrr(sim_en, local_gold)
        out["mrr_en_all"] = mrr_en
        out["mrr_en_canonical"] = _subset_mrr(ranks_en, canonical_mask[has_en])
        out["n_en_corpus"] = int(has_en.sum())
        print(f"[check {spec.name}] en MRR={mrr_en:.4f} on {int(has_en.sum()):,} rows "
              f"({time.time()-t0:.1f}s)", file=sys.stderr)

    # Synonym AUC in each language.
    syn_pairs = _syn_pairs(source, {rid: i for i, rid in enumerate(ids_order)})
    if syn_pairs:
        rng = np.random.default_rng(42)
        rand = _rand_pairs(len(syn_pairs), N, rng)
        out["synonym_auc_ko"] = _auc(_cos(pko, syn_pairs), _cos(pko, rand))
        out["synonym_pairs_ko"] = len(syn_pairs)
        if has_en.sum():
            # Restrict synonym pairs to rows with EN on both sides.
            en_syn_mask_pairs = [(a, b) for a, b in syn_pairs if has_en[a] and has_en[b]]
            if en_syn_mask_pairs:
                rand_en = _rand_pairs(len(en_syn_mask_pairs), int(has_en.sum()), rng)
                # Reindex to the EN-restricted array
                enidx = {int(orig): k for k, orig in enumerate(np.where(has_en)[0])}
                en_syn_r = [(enidx[a], enidx[b]) for a, b in en_syn_mask_pairs]
                out["synonym_auc_en"] = _auc(_cos(pen[has_en], en_syn_r), _cos(pen[has_en], rand_en))
                out["synonym_pairs_en"] = len(en_syn_r)

    if baseline:
        gold_ko = np.arange(N, dtype=np.int32)
        t0 = time.time()
        bm25_ko, _ = _bm25_mrr([r["ko"] for r in source], [r["definition_ko"] for r in source], gold_ko)
        out["bm25_ko"] = bm25_ko
        print(f"[check {spec.name}] BM25 ko={bm25_ko:.4f} ({time.time()-t0:.1f}s)", file=sys.stderr)
        if has_en.sum():
            en_ids = [ids_order[i] for i in np.where(has_en)[0]]
            en_q = [translations[i]["proverb_en"] for i in en_ids]
            en_c = [translations[i]["definition_en"] for i in en_ids]
            gold_en = np.arange(len(en_q), dtype=np.int32)
            t0 = time.time()
            bm25_en, _ = _bm25_mrr(en_q, en_c, gold_en)
            out["bm25_en"] = bm25_en
            print(f"[check {spec.name}] BM25 en={bm25_en:.4f} ({time.time()-t0:.1f}s)", file=sys.stderr)

    return out


def write_report(results: list[dict]) -> None:
    out_path = config.RUNS / "research" / "model_check.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Step 7.0 model check", "",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- Candidates: {len(results)}",
             f"- Instruction: `Instruct: Given a Korean or English text, retrieve texts with the same meaning\\nQuery: `",
             ""]
    lines.append("| Model | n | Korean MRR (all / canon) | English MRR (all / canon) | BM25 KO / EN | Syn AUC KO / EN |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        ko = f"{r.get('mrr_ko_all', 0):.3f} / {r.get('mrr_ko_canonical', 0):.3f}"
        if r.get("mrr_en_all") is not None:
            en = f"{r['mrr_en_all']:.3f} / {r['mrr_en_canonical']:.3f}"
        else:
            en = "— (no EN yet)"
        bm = f"{r.get('bm25_ko', 0):.3f} / " + (f"{r['bm25_en']:.3f}" if r.get("bm25_en") is not None else "—")
        syn = f"{r.get('synonym_auc_ko', 0):.3f} / " + (f"{r['synonym_auc_en']:.3f}" if r.get("synonym_auc_en") is not None else "—")
        lines.append(f"| `{r['model']}` | {r['n']:,} | {ko} | {en} | {bm} | {syn} |")
    lines.append("")

    # Pick the primary model: highest Korean canonical MRR among those with
    # synonym_auc_ko >= 0.7. Fall back if none pass.
    eligible = [r for r in results if r.get("synonym_auc_ko", 0) >= 0.7]
    if eligible:
        best = max(eligible, key=lambda r: r.get("mrr_ko_canonical", 0))
    else:
        best = max(results, key=lambda r: r.get("mrr_ko_canonical", 0))
        lines.append("**Note:** every candidate failed the 0.7 synonym AUC sanity check. "
                     "Picking the highest-MRR anyway, but flag as unreliable.")
    lines.append(f"**Primary model:** `{best['model']}` (license: {best['license']}).")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (config.RUNS / "primary_model.json").write_text(json.dumps({
        "model": best["model"], "license": best["license"],
    }, indent=2), encoding="utf-8")
    print(f"[check] wrote {out_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(CANDIDATES))
    p.add_argument("--max-rows", type=int, default=None, help="Limit for quick smoke tests.")
    p.add_argument("--no-baseline", action="store_true")
    args = p.parse_args()
    config.ensure_dirs()
    results = []
    for key in args.models:
        if key not in CANDIDATES:
            print(f"[check] unknown model key: {key}", file=sys.stderr)
            continue
        try:
            r = check_model(key, max_rows=args.max_rows, baseline=not args.no_baseline)
        except FileNotFoundError:
            print(f"[check] no cached embeddings for {key}; run `make embed MODEL={key}` first", file=sys.stderr)
            continue
        results.append(r)
    if results:
        write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
