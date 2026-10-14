"""Step 7 RQ1: does the model translate the proverb's image, or its meaning?

Pair selection (1,094 headwords with multiple senses):
    A = kept sense (same as Step 2 and the dataset).
    B = lowest-sense_no *other* 일반어 sense, else lowest other overall.

New requests (per mode A/B/C):
    A2 — definition A, seed = row_id + 1,000,000
    B1 — definition B, seed = row_id
    N1 — no definition, prompt p1-nodef, seed = row_id
Mode C also runs a thinking-on quartet A1t/A2t/B1t/N1t on the same headwords.

Measures per headword and per embedding model, written to
`data/research/rq1_measures.jsonl`. Statistics in `runs/research/rq1_report.md`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .client import async_client
from .jsonl import append_jsonl, read_jsonl, write_jsonl
from .runner import gather_bounded, one_attempt
from .translate import read_mode, read_prompt_choice


def _pick_sense_no_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 10**9


def build_pairs() -> list[dict]:
    """Build 1,094 A/B pairs from the per-sense proverbs.jsonl.

    A = canonical sense of the proverb (`is_canonical_sense == True`).
    B = sibling sense picked by the plan's rule: lowest-sense_no 일반어 among
        siblings, else lowest-sense_no overall.
    """
    rows = list(read_jsonl(config.PROVERBS_JSONL))
    by_id = {r["id"]: r for r in rows}
    out = []
    for a in rows:
        if not a.get("is_canonical_sense"):
            continue
        siblings = [by_id[sid] for sid in a.get("sibling_sense_ids", []) if sid in by_id]
        if not siblings:
            continue
        std = [s for s in siblings if s["type"] == "일반어"]
        pool = std if std else siblings
        pool_sorted = sorted(pool, key=lambda s: _pick_sense_no_int(s["sense_no"]))
        b = pool_sorted[0]
        out.append({
            "id": a["id"],
            "ko": a["ko"],
            "type_a": a["type"],
            "type_b": b["type"],
            "definition_ko_a": a["definition_ko"],
            "definition_ko_b": b["definition_ko"],
            "figurative_a": a.get("figurative", False),
            "figurative_b": b.get("figurative", False),
            "same_def": a["definition_ko"].strip() == b["definition_ko"].strip(),
            "sense_no_a": a["sense_no"],
            "sense_no_b": b["sense_no"],
            "sibling_id": b["id"],
        })
    return out


async def run_arms(mode: str, concurrency: int, arms: set[str] | None = None) -> int:
    """Run the configured RQ1 arms. `arms` filters which of {A2, B1, N1} to
    fire (default: all three)."""
    config.ensure_dirs()
    pairs = build_pairs()
    print(f"[rq1] {len(pairs)} multi-sense headwords", file=sys.stderr)

    prompt_version = read_prompt_choice()
    preflight = json.loads((config.RUNS / "preflight.json").read_text(encoding="utf-8"))
    vllm_version = preflight.get("vllm_version")
    budget_accepted = preflight.get("thinking_token_budget_accepted", True)

    thinking_dataset = (mode == "A")
    hybrid = (mode == "C")

    # Resume: skip (id, arm) pairs already in rq1_translations.jsonl.
    out_path = config.RESEARCH / "rq1_translations.jsonl"
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            done.add((r["id"], r["arm"]))

    client = async_client()

    async def _run(pair: dict, arm: str, thinking: bool, with_def: bool, defn_text: str | None, seed: int):
        if (pair["id"], arm) in done:
            return
        # Build a row-like dict for `one_attempt`. For B1 we temporarily swap the definition.
        row = {
            "id": pair["id"],
            "ko": pair["ko"],
            "definition_ko": defn_text if defn_text is not None else pair["definition_ko_a"],
            "type": pair["type_a"],
            "synonyms_ko": [],
        }
        res = await one_attempt(
            client=client, row=row, seed=seed,
            prompt_version=prompt_version if with_def else "p1-nodef",
            thinking=thinking, reason=f"rq1_{arm}", attempt_no=1,
            vllm_version=vllm_version, budget_accepted=budget_accepted,
            with_definition=with_def,
        )
        append_jsonl(config.ATTEMPTS_JSONL, [res.record])
        from pydantic import ValidationError
        try:
            if with_def:
                from .schema import ProverbTranslation
                parsed = ProverbTranslation.model_validate_json(res.content)
                out_row = {
                    "id": pair["id"], "arm": arm, "thinking": thinking,
                    "proverb_en": parsed.proverb_en, "definition_en": parsed.definition_en,
                    "attempt_id": res.record["attempt_id"], "seed": seed,
                }
            else:
                from .schema import ProverbOnly
                parsed = ProverbOnly.model_validate_json(res.content)
                out_row = {
                    "id": pair["id"], "arm": arm, "thinking": thinking,
                    "proverb_en": parsed.proverb_en, "attempt_id": res.record["attempt_id"], "seed": seed,
                }
            append_jsonl(out_path, [out_row])
        except (ValidationError, ValueError, json.JSONDecodeError):
            pass

    want = arms or {"A2", "B1", "N1"}
    tasks = []
    for p in pairs:
        row_id = int(p["id"])
        s1, s2 = row_id, row_id + 1_000_000
        if "A2" in want:
            tasks.append(_run(p, "A2", False, True, p["definition_ko_a"], s2))
        if "B1" in want:
            tasks.append(_run(p, "B1", False, True, p["definition_ko_b"], s1))
        if "N1" in want:
            tasks.append(_run(p, "N1", False, False, None, s1))
    print(f"[rq1] firing {len(tasks)} new requests "
          f"({len(pairs)} pairs × arms {sorted(want)})", file=sys.stderr)
    await gather_bounded(tasks, concurrency=concurrency)
    print(f"[rq1] done. wrote {out_path}", file=sys.stderr)
    return 0


# ---------- measurement + stats ----------

def _chrf(a: str, b: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(a, [b]).score


def compute_measures(pairs: list[dict], arm_outputs: dict[tuple[str, str], dict], model_name: str) -> list[dict]:
    """One measure row per headword × embedding model.

    `arm_outputs[(id, arm)]` → {"proverb_en", "definition_en"?, "thinking"}.
    `A1` comes from the main dataset (translations.jsonl)."""
    from .embed import load_cached, _cosine
    ids_order, _ = load_cached(model_name, "ko")
    idx = {i: k for k, i in enumerate(ids_order)}

    pko = load_cached(model_name, "ko")[1]
    pen = load_cached(model_name, "proverb_en")[1]
    dkb = load_cached(model_name, "definition_ko")[1]
    den = load_cached(model_name, "definition_en")[1]
    # Also embed the newly-produced arm texts on the fly.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def _embed(texts: list[str]) -> np.ndarray:
        v = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32,
                         show_progress_bar=False)
        return v.astype(np.float32)

    out = []
    # Pull the main-dataset proverb_en vector for A1.
    for pair in pairs:
        rid = pair["id"]
        if rid not in idx:
            continue
        i = idx[rid]
        a1_vec = pen[i]
        a2 = arm_outputs.get((rid, "A2"))
        b1 = arm_outputs.get((rid, "B1"))
        n1 = arm_outputs.get((rid, "N1"))
        if not (a2 and b1 and n1):
            continue
        # Embed the per-arm proverb_en
        vecs = _embed([a2["proverb_en"], b1["proverb_en"], n1["proverb_en"], b1.get("definition_en", "")])
        a2_vec, b1_vec, n1_vec, def_b_vec = vecs
        row = {
            "id": rid,
            "model": model_name,
            "same_def": pair["same_def"],
            "type_a": pair["type_a"], "type_b": pair["type_b"],
            "figurative_a": pair["figurative_a"], "figurative_b": pair["figurative_b"],
            "changed_A2": int(a2["proverb_en"].strip() != arm_outputs[(rid, "A1")]["proverb_en"].strip()),
            "changed_B1": int(b1["proverb_en"].strip() != arm_outputs[(rid, "A1")]["proverb_en"].strip()),
            "surf_A2": 1 - _chrf(arm_outputs[(rid, "A1")]["proverb_en"], a2["proverb_en"]) / 100.0,
            "surf_B1": 1 - _chrf(arm_outputs[(rid, "A1")]["proverb_en"], b1["proverb_en"]) / 100.0,
            "sem_A2": float(1 - _cosine(a1_vec, a2_vec)),
            "sem_B1": float(1 - _cosine(a1_vec, b1_vec)),
            "def_effect": float(1 - _cosine(a1_vec, n1_vec)),
            "def_dissim": float(1 - _cosine(dkb[i], _embed([pair["definition_ko_b"]])[0])),
        }
        row["leak"] = row["sem_B1"] - row["sem_A2"]
        row["toward_B"] = float(_cosine(b1_vec, def_b_vec) - _cosine(a1_vec, def_b_vec))
        row["toward_B_noise"] = float(_cosine(a2_vec, def_b_vec) - _cosine(a1_vec, def_b_vec))
        out.append(row)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    arms_p = sub.add_parser("arms")
    arms_p.add_argument("--concurrency", type=int, default=16)
    arms_p.add_argument("--arms", nargs="+", default=["A2", "B1", "N1"],
                        choices=["A2", "B1", "N1"],
                        help="Which arms to fire. Default: all three.")
    sub.add_parser("measure")
    args, extra = p.parse_known_args()
    config.ensure_dirs()
    if args.cmd == "arms":
        return asyncio.run(run_arms(read_mode(),
                                     getattr(args, "concurrency", 16),
                                     set(getattr(args, "arms", ["A2", "B1", "N1"]))))
    if args.cmd == "measure":
        pairs = build_pairs()
        arms_path = config.RESEARCH / "rq1_translations.jsonl"
        arms_rows = list(read_jsonl(arms_path))
        trans = {r["id"]: r for r in read_jsonl(config.TRANSLATIONS_JSONL)}
        arm_outputs: dict[tuple[str, str], dict] = {}
        for a in arms_rows:
            arm_outputs[(a["id"], a["arm"])] = a
        for rid, tr in trans.items():
            arm_outputs[(rid, "A1")] = tr
        all_rows = []
        for m in config.EMBED_MODELS:
            all_rows += compute_measures(pairs, arm_outputs, m)
        write_jsonl(config.RESEARCH / "rq1_measures.jsonl", all_rows)
        print(f"[rq1 measure] wrote {len(all_rows)} rows", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
