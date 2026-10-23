# Korean proverbs → English

A dataset of every proverb (속담) in the Urimalsaem dictionary (우리말샘, National Institute of Korean Language), machine-translated to English with the Korean dictionary definition as context. Built 2026-10-14.

Available on Hugging Face: <https://huggingface.co/datasets/NLPatVCU/korean_proverb> (dataset + companion embeddings; neither is shipped here on GitHub).

## Dataset at a glance

- **10,763 rows**, one per (proverb, sense) pair in Urimalsaem
- Covers **9,612 unique proverbs** (group by `headword_raw` for one row per proverb)
- Each row has: Korean proverb, one Korean dictionary definition, English translations of both (thinking off and thinking on), and the model's reasoning for the thinking-on translations

**Files:** `data/korean_proverbs_en.parquet` on Hugging Face. `make package` builds it locally in `data/output/`, along with CSV and JSONL copies.

## Columns

| Column | Description |
|---|---|
| `id` | Urimalsaem `target_code` for this sense |
| `headword_raw` | Raw headword text, with `[A/B]` / `(C)` variant notation |
| `ko` | Canonical form of the proverb (variant notation stripped) |
| `sense_no` | Urimalsaem's own sense number within this headword (`"001"`, `"002"`, …) |
| `type` | Urimalsaem's usage-category label for this sense (`일반어` is the standard-usage code; see Urimalsaem for the other codes) |
| `definition_ko` | Dictionary definition, with `<동의 속담>`/`<동의 관용구>` tag and following synonym list stripped |
| `synonyms_ko` | The stripped synonym list |
| `proverb_en_nonthinking` | English translation of the proverb, thinking off |
| `definition_en_nonthinking` | English translation of the definition, thinking off |
| `proverb_en_thinking` | English translation of the proverb, thinking on |
| `definition_en_thinking` | English translation of the definition, thinking on |
| `reasoning` | The model's reasoning for the thinking-on translations |

## Source

- 우리말샘 (Urimalsaem), National Institute of Korean Language.
- Dump dated 2026-06-02, via the GitHub mirror `spellcheck-ko/korean-dict-nikl`, pinned to commit `42c0d01`, folder `opendict/` only.

## License

CC BY-SA 2.0 KR, inherited from the source. Translations are derived data and carry the same license. Attribution: 국립국어원 (NIKL), 우리말샘.

## Model

- Translator: `nvidia/Qwen3.5-122B-A10B-NVFP4`, served by vLLM 0.28.0 on an NVIDIA DGX Spark.
- Every row was translated twice with the same prompt: once with thinking off (`*_nonthinking` columns) and once with thinking on (`*_thinking` columns).
- Prompt: `p1-zero` (see `runs/prompts/p1-zero.txt`), zero-shot, identical for every row.
- Sampling with thinking off (`enable_thinking: false`): `temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1.0, max_tokens=1024`.
- Sampling with thinking on (`enable_thinking: true`): `temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1.0, max_tokens=20480, thinking_token_budget=16384`.
- Seed: `int(id)`, reproducible per row.
- Output format: Pydantic model → JSON schema via `response_format` on `chat/completions`.

## Canonicalization of the Korean proverb

Headwords in Urimalsaem use `[X/Y]` and `(…)` notation for variants. We derive `ko` by:
1. Delete every `[...]` block including its contents (the text before the bracket is already the first alternative).
2. Delete `(…)` whose contents contain Hanja (dictionary glosses).
3. Keep contents of `(…)` with Hangul inside, taking only the first alternative on `/`.
4. Collapse whitespace.

The raw headword is preserved as `headword_raw`.

## A short experiment

Three embedding checks on the dataset's thinking-off translations, and one comparing thinking on and off. Numbers below use `kakaocorp/kanana-nano-2.1b-embedding` (CC-BY-NC-4.0); all three findings replicate under `Qwen/Qwen3-Embedding-8B` (Apache-2.0).

Both encoders embed the proverb and the definition with the same query prompt, so the Korean and English sides live in the same embedding space and are directly comparable. For Qwen3 the custom instruction is `"Instruct: Given a Korean or English text, retrieve texts with the same meaning\nQuery: "`. Embeddings themselves are shipped separately [on Hugging Face](https://huggingface.co/datasets/NLPatVCU/korean_proverb); `make embed MODELS="kanana qwen3"` regenerates them against the shipped translations.

### Is the Korean proverb closer to its sense than the English translation is?

Cosine similarity between each proverb and its own dictionary definition, averaged across all 10,763 rows:

| Side | Mean cos | Median cos |
|---|---|---|
| Korean proverb ↔ Korean definition | 0.559 | 0.543 |
| English translation ↔ English definition | 0.541 | 0.521 |

Qwen3-Embedding-8B agrees in direction: 0.661 vs 0.642.

The Korean side sits a few points closer, consistently, under both encoders. A plausible read: these models have seen enough Korean text that each proverb's wording has already been pulled toward its idiomatic meaning in embedding space, while the English translations are novel strings and sit a touch further from their definitions.

### Does swapping the definition change the English translation?

For the 1,094 proverbs with ≥2 dictionary senses we re-translated each one two more times: once with the sibling sense's definition in the prompt (B1), once with no definition at all (N1). Then we measured how far each new translation drifted from the main-run output (A1) in cosine terms.

| Pair | Median shift from A1 |
|---|---|
| A1 vs B1 (different definition) | 0.100 |
| A1 vs N1 (no definition) | 0.097 |

The two drifts are nearly identical: the translator preserves the proverb's wording regardless of which definition it saw. There is a small, statistically detectable signal that the definition does matter: drift scales with how dissimilar the two definitions are (Spearman ρ=0.14, p=2e-6), and the English shifts toward B's embedding when B is swapped in (p=9e-14). But the effect is tiny.

### Does the proverb-to-meaning gap survive translation?

For each row we measured the cosine distance between the proverb and its own definition, a proxy for how figurative the proverb is. A rough internal label: definitions where NIKL's own prose contains the phrase `비유적으로` ("said figuratively") tend to be the more metaphorical entries. Entries with that marker should sit further from their definitions than entries without it.

- With-marker vs without, in Korean: AUC 0.581
- With-marker vs without, in English: AUC 0.584
- Rank correlation of the two distances across all 10,763 rows: Spearman ρ = 0.845

(This marker is derived by regex over `definition_ko` and is not shipped as a column in the dataset; see `src/korean_proverb/extract.py` for the exact rule.)

If a proverb is far from its definition in Korean, the English translation is far from its definition in English, at nearly the same rank. The absolute distance shifts a little (finding #1 above), but the ordering is preserved almost perfectly.

### Does thinking change the translation?

We ran two more thinking-on passes over the 1,094 multi-sense proverbs from the previous experiment:
- a re-seed of 300 of them (seed `int(id) + 1000000`), to measure how much a translation changes by chance alone
- all 1,094 with no definition in the prompt (`p1-nodef`), the thinking-on counterpart of N1

Median shift of the English proverb (1 − cos), kanana:

| Pair | Rows | Median shift |
|---|---|---|
| Thinking on vs re-seed (chance alone) | 300 | 0.086 |
| Thinking on vs no definition | 300 | 0.105 |
| Thinking on vs thinking off | 300 | 0.133 |
| Thinking on vs no definition | 1,094 | 0.111 |
| Thinking off vs no definition (A1 vs N1 above) | 1,094 | 0.097 |

Thinking-on and thinking-off translations differ well beyond chance (paired Wilcoxon p=1e-16). The two runs also used different sampling settings (see Model), so the difference can't be pinned on thinking alone.

Removing the definition barely matters with thinking on either: it moves a translation only slightly more than re-seeding does (median paired difference 0.005), and about as much as it moves a thinking-off one.

Qwen3-Embedding-8B agrees in direction on all of these.

Per-row measures:
- `data/research/proverb_distances.csv` / `.parquet`: proverb-to-definition distances (kanana + qwen3 columns)
- `data/research/rq1_measures.csv` / `.parquet`: A1/B1/N1 per-pair measures
- `data/research/thinking_measures.csv` / `.parquet`: thinking-on vs thinking-off per-proverb shifts

## Reproducibility

Anyone with a DGX Spark (or equivalent vLLM-capable host serving `nvidia/Qwen3.5-122B-A10B-NVFP4`) and a GPU for embeddings can rebuild the dataset end to end. Dep versions are in `pyproject.toml`; the exact prompt, sampling, and seed are above.

```bash
make install                 # Python 3.11+; installs korean_proverb + test deps
make test                    # unit tests for the canonicalizer and extractor

make download                # clone the pinned Urimalsaem dump (opendict/@42c0d01)
make extract                 # produce data/interim/proverbs.jsonl (10,763 rows)

export QWEN_BASE_URL=http://<your-spark>:8000/v1
make preflight               # confirm the vLLM endpoint

# One-time: write the mode choice. We used B (thinking off).
echo '{"mode":"B"}' > runs/mode.json

make translate               # produce data/output/translations.jsonl (10,763 rows)
make translate-thinking      # produce data/output/translations_thinking.jsonl (10,763 rows)
make package                 # produce jsonl/csv/parquet in data/output/

# Rebuilding the per-row research measures used in "A short experiment":
make install-research
export EMBED_BASE_URL=http://<your-gpu-host>:8800   # see scripts/4090_setup.md
make embed MODELS="kanana qwen3"
python -m korean_proverb.embed --models kanana qwen3 --fields proverb_en_thinking definition_en_thinking
make rq2                     # writes data/research/proverb_distances.*
python -m korean_proverb.rq1 arms --arms N1 --concurrency 16   # N1 arm (Spark pass)
python -m korean_proverb.rq1_analyze                            # writes data/research/rq1_measures.*
python -m korean_proverb.thinking_research reseed --subset sample   # thinking-on re-seed (Spark pass)
python -m korean_proverb.thinking_research nodef --subset all       # thinking-on, no definition (Spark pass)
python -m korean_proverb.thinking_analyze                       # writes data/research/thinking_measures.*
```

## Limitations

- Machine-translated with no human review.
- One row per sense; dedupe on `headword_raw` for one row per proverb (we don't designate a primary sense for you — Urimalsaem's `sense_no` is their own ordering).
- Only the canonical form is translated; variant combinations in `headword_raw` are not expanded.
