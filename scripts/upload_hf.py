"""Upload the dataset and embeddings to a Hugging Face Datasets repo.

Final HF layout:

    <repo>/
      README.md                              # YAML frontmatter + data card
      data/
        korean_proverbs_en.parquet           # the dataset (10,763 rows per sense)
      embeddings/
        README.md                            # license notes for the embedding models
        kanana.parquet                       # one row per sense id, 6 list[float32] columns
        qwen3.parquet                        # same shape as kanana

Each embeddings/<model>.parquet carries:
    id                             : str (joins to data/korean_proverbs_en.parquet)
    ko_embedding                   : list[float32]
    definition_ko_embedding        : list[float32]
    proverb_en_nonthinking_embedding    : list[float32]
    definition_en_nonthinking_embedding : list[float32]
    proverb_en_thinking_embedding       : list[float32]
    definition_en_thinking_embedding    : list[float32]

Usage:
    huggingface-cli login
    .venv/bin/python scripts/upload_hf.py --repo your-handle/korean-proverbs-en

The script is idempotent: it builds parquet artifacts into a staging dir
and then uses `HfApi.upload_folder` which skips files that haven't changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Make imports work when invoked as `python scripts/upload_hf.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from korean_proverb import config
from korean_proverb.embed import CANDIDATES, _cache_dir, load_field


def _list_array(arr: np.ndarray) -> pa.ListArray:
    """(N, D) float32 → list[float32] column, built from the flat buffer rather
    than N Python lists (those need several GB for the 4,096-dim model)."""
    n, d = arr.shape
    offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, pa.array(arr.reshape(-1)))


def build_embedding_parquet(model_key: str, out_path: Path) -> None:
    """Collect the six fields' .npy caches for a model, align them by id, and
    write to one parquet with six list[float32] columns."""
    spec = CANDIDATES[model_key]
    print(f"[hf] building {out_path.name} from {spec.name}", file=sys.stderr)

    # Load each field; ids may be in insertion order rather than canonical
    # proverbs.jsonl order. Reindex everyone to canonical order so the
    # resulting parquet is deterministic and joins cleanly to the dataset.
    canonical_ids = [r["id"] for r in _read_jsonl(config.PROVERBS_JSONL)]
    cols: dict[str, np.ndarray] = {}
    for field, col_name in [
        ("ko", "ko_embedding"),
        ("definition_ko", "definition_ko_embedding"),
        ("proverb_en", "proverb_en_nonthinking_embedding"),
        ("definition_en", "definition_en_nonthinking_embedding"),
        ("proverb_en_thinking", "proverb_en_thinking_embedding"),
        ("definition_en_thinking", "definition_en_thinking_embedding"),
    ]:
        ids, vecs = load_field(spec.name, field)
        index = {rid: k for k, rid in enumerate(ids)}
        aligned = np.stack([vecs[index[rid]] for rid in canonical_ids])
        cols[col_name] = aligned.astype(np.float32)

    # Each embedding column becomes a list[float32].
    table = pa.table({
        "id": pa.array(canonical_ids, type=pa.string()),
        **{name: _list_array(arr) for name, arr in cols.items()},
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[hf]   wrote {out_path} ({size_mb:.1f} MB, {len(canonical_ids):,} rows, "
          f"dim={cols['ko_embedding'].shape[1]})", file=sys.stderr)


def _read_jsonl(path: Path) -> list[dict]:
    import json
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_card(stage: Path, repo_id: str) -> None:
    """Write the HF-side README (YAML frontmatter + card). Mirrors the live card;
    keep the two in sync when the card changes."""
    pf = {}
    pf_path = config.RUNS / "preflight.json"
    if pf_path.exists():
        pf = json.loads(pf_path.read_text(encoding="utf-8"))
    vllm_version = pf.get("vllm_version", "unknown")

    card = f"""---
language:
  - ko
  - en
license: other
license_name: cc-by-sa-2.0-kr
license_link: https://creativecommons.org/licenses/by-sa/2.0/kr/
homepage: https://github.com/taylorsm9/korean_proverbs
pretty_name: Korean proverbs → English (Urimalsaem)
tags:
  - korean
  - proverbs
  - urimalsaem
  - machine-translation
  - 속담
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/korean_proverbs_en.parquet
annotations_creators:
  - machine-generated
source_datasets:
  - original
task_categories:
  - translation
---

# Korean proverbs → English

Every proverb (속담) entry in the Urimalsaem dictionary (우리말샘, National Institute of Korean Language), machine-translated to English. Code and pipeline details: <https://github.com/taylorsm9/korean_proverbs>.

The thinking-on translations come with the model's full reasoning traces, one per row. To our knowledge, as of October 2026, this is the largest public set of reasoning traces for idioms or proverbs, about six times the size of the next largest ([Chengyu-Bench](https://aclanthology.org/2025.emnlp-main.119/), which released 1,834 DeepSeek-R1 traces), and the only one for translation.

## Structure of this dataset (read this first)

Urimalsaem is a sense-level dictionary: a single proverb can have more than one recorded meaning, and each meaning is a separate entry. This dataset follows that structure.

- **One row per dictionary sense.** 10,763 rows total.
- **9,612 unique proverbs.** 1,094 proverbs have two or more senses; the other 8,518 have one.
- When a proverb has multiple senses, the same `ko` string appears on multiple rows, each with its own `definition_ko` and its own English translations. All senses of a given proverb share the same `headword_raw`.
- `sense_no` is Urimalsaem's own ordering within a headword (`"001"`, `"002"`, …). This dataset does not pick a primary sense for you.

**If you want one row per proverb**, you need to pick a sense yourself. The simplest options:

```python
# Keep the lowest-numbered sense per proverb (Urimalsaem's own order):
one_per = df.sort_values(["headword_raw", "sense_no"]).drop_duplicates("headword_raw", keep="first")
```

## Scale

| Measure | Value |
|---|---|
| Total rows | 10,763 |
| Unique proverbs (unique `headword_raw`) | 9,612 |
| Proverbs with multiple senses | 1,094 |

## Load it

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
df = ds.to_pandas()
```

Or load the parquet directly:

```python
import pandas as pd
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo_id}", "data/korean_proverbs_en.parquet", repo_type="dataset")
df = pd.read_parquet(path)
```

## Columns

| Column | Description |
|---|---|
| `id` | Urimalsaem `target_code` for this sense |
| `headword_raw` | Raw headword text, including Urimalsaem's `[A/B]` / `(C)` variant notation |
| `ko` | Canonical form of the proverb (variant notation stripped) |
| `sense_no` | Sense number within this proverb, as assigned by Urimalsaem (`"001"`, `"002"`, …) |
| `type` | Urimalsaem's usage-category label for this sense (`일반어` is the standard-usage code; see Urimalsaem for the other codes) |
| `definition_ko` | Dictionary definition (synonym tag and following synonym list are stripped out) |
| `synonyms_ko` | The stripped synonym list: proverb-length strings that are their own dictionary entries |
| `proverb_en_nonthinking` | English translation of the proverb, thinking off |
| `definition_en_nonthinking` | English translation of the definition, thinking off |
| `proverb_en_thinking` | English translation of the proverb, thinking on |
| `definition_en_thinking` | English translation of the definition, thinking on |
| `reasoning` | The model's reasoning for the thinking-on translations |

## Companion embeddings

Vector embeddings of the Korean and English texts, computed with two models, are shipped as separate parquet files:

```
embeddings/
  kanana.parquet         # 1,792-dim
  qwen3.parquet          # 4,096-dim
```

Each row carries `id` plus six `list[float32]` columns: `ko_embedding`, `definition_ko_embedding`, `proverb_en_nonthinking_embedding`, `definition_en_nonthinking_embedding`, `proverb_en_thinking_embedding`, `definition_en_thinking_embedding`. Join on `id`:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

df  = pd.read_parquet(hf_hub_download("{repo_id}", "data/korean_proverbs_en.parquet", repo_type="dataset"))
emb = pd.read_parquet(hf_hub_download("{repo_id}", "embeddings/kanana.parquet",       repo_type="dataset"))
joined = df.merge(emb, on="id")
```

Both the proverb side and the definition side are embedded with the same prompt, so they share one space (useful for cosine similarity between a proverb and its own meaning). For Qwen3-Embedding the prompt is:

```
Instruct: Given a Korean or English text, retrieve texts with the same meaning
Query:
```

## Source

- 우리말샘 (Urimalsaem), National Institute of Korean Language (국립국어원).
- Dump dated 2026-06-02, via the GitHub mirror [`spellcheck-ko/korean-dict-nikl`](https://github.com/spellcheck-ko/korean-dict-nikl), pinned to commit `42c0d01`, folder `opendict/` only.

## License

- The **dataset** (Korean proverbs, Korean definitions, English translations, reasoning): **CC-BY-SA-2.0-KR**, matching Urimalsaem. See <https://creativecommons.org/licenses/by-sa/2.0/kr/>. Attribution: 국립국어원 (NIKL), 우리말샘.
- `embeddings/kanana.parquet`: **CC-BY-NC-4.0** (non-commercial only), inherited from `kakaocorp/kanana-nano-2.1b-embedding`.
- `embeddings/qwen3.parquet`: **Apache-2.0**, inherited from `Qwen/Qwen3-Embedding-8B`.

## Translation setup

- Model: `nvidia/Qwen3.5-122B-A10B-NVFP4` served by vLLM {vllm_version} on NVIDIA DGX Spark.
- Every row was translated twice with the same prompt: once with thinking off (`*_nonthinking` columns) and once with thinking on (`*_thinking` columns).
- Sampling with thinking off (`enable_thinking: false`): `temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1.0, max_tokens=1024`.
- Sampling with thinking on (`enable_thinking: true`): `temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1.0, max_tokens=20480, thinking_token_budget=16384`.
- Seed per row: `int(id)`.
- Output constrained to a Pydantic JSON schema via `response_format`.
- No thinking-off translation hit the `max_tokens=1024` ceiling. The longest completion seen was 156 tokens.
- No thinking-on request hit the thinking budget. The longest reasoning trace was 11,488 tokens.

## Limitations

- Machine-translated; no human review.
- Multiple rows share the same `ko` when a proverb has multiple senses. Dedupe on `headword_raw` if you want unique proverb strings (see "Structure" above).
- Only the canonical form of each proverb is translated; variant notation in `headword_raw` is not expanded.

## Citation

If you use this dataset, please cite the source dictionary:

> 국립국어원 (National Institute of Korean Language). 우리말샘 (Urimalsaem).
> Available via [`spellcheck-ko/korean-dict-nikl`](https://github.com/spellcheck-ko/korean-dict-nikl).
"""
    (stage / "README.md").write_text(card, encoding="utf-8")
    print(f"[hf]   wrote {stage/'README.md'}", file=sys.stderr)


def write_embeddings_readme(stage: Path) -> None:
    emb_md = stage / "embeddings" / "README.md"
    emb_md.parent.mkdir(parents=True, exist_ok=True)
    emb_md.write_text("""# Embeddings

Per-model parquet files. Each file carries one row per proverb-sense (`id`
joins back to the main dataset), with six `list[float32]` columns:

- `ko_embedding`: embedding of the Korean proverb (`ko`)
- `definition_ko_embedding`: embedding of the Korean definition
- `proverb_en_nonthinking_embedding`: embedding of the English translation (`proverb_en_nonthinking`)
- `definition_en_nonthinking_embedding`: embedding of the English definition (`definition_en_nonthinking`)
- `proverb_en_thinking_embedding`: embedding of the English translation (`proverb_en_thinking`)
- `definition_en_thinking_embedding`: embedding of the English definition (`definition_en_thinking`)

| File | Model | Dim | License |
|---|---|---|---|
| `kanana.parquet` | `kakaocorp/kanana-nano-2.1b-embedding` | 1,792 | **CC-BY-NC-4.0** (non-commercial) |
| `qwen3.parquet`  | `Qwen/Qwen3-Embedding-8B`              | 4,096 | Apache-2.0 |

All texts are embedded with the model's query prompt applied identically on
both sides (both proverb and definition are treated as queries), so a
same-row proverb and definition live in the same embedding space.

Both models receive the **same** instruction string, applied verbatim as a
prefix to every text (Korean and English, proverb and definition):

```
Instruct: Given a Korean or English text, retrieve texts with the same meaning
Query: 
```

Per-model specifics:

- `kanana.parquet` (`kakaocorp/kanana-nano-2.1b-embedding`): the prefix
  above is prepended to every input; pooling uses `include_prompt=false`
  (mean-pool over non-prompt tokens), per the Kanana model card.
- `qwen3.parquet` (`Qwen/Qwen3-Embedding-8B`): the prefix above is passed
  as the model's `custom_prompt` (the Qwen3-Embedding `Instruct: … \\nQuery: `
  template), applied identically to every input.

Vectors are L2-normalized float32.
""", encoding="utf-8")
    print(f"[hf]   wrote {emb_md}", file=sys.stderr)


def stage_everything(stage: Path, repo_id: str, models: list[str], regenerate_card: bool) -> None:
    stage.mkdir(parents=True, exist_ok=True)

    # Dataset parquet
    src = config.OUTPUT / "korean_proverbs_en.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing. Run `make package` first.")
    dst = stage / "data" / "korean_proverbs_en.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f"[hf]   staged {dst}", file=sys.stderr)

    # Per-model embedding parquets
    for key in models:
        if key not in CANDIDATES:
            raise ValueError(f"unknown model key: {key}")
        build_embedding_parquet(key, stage / "embeddings" / f"{key}.parquet")

    # Embeddings README and dataset card: both hand-maintained, so preserve
    # them unless --regenerate-card is passed.
    emb_md = stage / "embeddings" / "README.md"
    if emb_md.exists() and not regenerate_card:
        print(f"[hf]   keeping existing {emb_md} (pass --regenerate-card to overwrite)",
              file=sys.stderr)
    else:
        write_embeddings_readme(stage)

    card_path = stage / "README.md"
    if card_path.exists() and not regenerate_card:
        print(f"[hf]   keeping existing {card_path} (pass --regenerate-card to overwrite)",
              file=sys.stderr)
    else:
        write_card(stage, repo_id)


def upload(repo_id: str, stage: Path, private: bool, data_only: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"[hf] create_repo warning (ok if the repo already exists): {e}", file=sys.stderr)

    # Only ever upload data when --data-only is set, so prose stays behind
    # a human review gate.
    allow_patterns = ["*.parquet"] if data_only else None
    scope = "parquet files only" if data_only else "full staging dir"
    print(f"[hf] uploading {stage} → {repo_id} ({scope}, private={private})", file=sys.stderr)
    msg = ("Dataset + kanana/qwen3 embeddings (parquet only)"
           if data_only else "Dataset + kanana/qwen3 embeddings")
    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        commit_message=msg,
    )
    print(f"[hf] done. https://huggingface.co/datasets/{repo_id}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HF repo id, e.g. your-handle/korean-proverbs-en")
    p.add_argument("--models", nargs="+", default=["kanana", "qwen3"],
                   help="Which models' embeddings to ship (default: both).")
    p.add_argument("--stage", default="hf_stage",
                   help="Local staging directory; built then uploaded.")
    p.add_argument("--private", action="store_true", help="Create as private repo.")
    p.add_argument("--dry-run", action="store_true", help="Only stage locally; don't upload.")
    p.add_argument("--regenerate-card", action="store_true",
                   help="Overwrite hf_stage/README.md and hf_stage/embeddings/README.md with "
                        "the autogenerated templates. Default is to preserve both (keep your edits).")
    p.add_argument("--data-only", action="store_true",
                   help="Upload parquet files only; skip README.md and embeddings/README.md. "
                        "Use this to push data without pushing any human-reviewable prose.")
    args = p.parse_args()

    stage = Path(args.stage).resolve()
    stage_everything(stage, args.repo, args.models, regenerate_card=args.regenerate_card)
    if args.dry_run:
        print(f"[hf] dry run complete; staged files in {stage}", file=sys.stderr)
        return 0
    upload(args.repo, stage, private=args.private, data_only=args.data_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
