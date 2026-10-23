"""Step 6: Package the final dataset.

Writes:
- data/output/korean_proverbs_en.jsonl
- data/output/korean_proverbs_en.csv
- data/output/korean_proverbs_en.parquet

README.md is hand-maintained; this script no longer generates it.
"""
from __future__ import annotations

import csv
import json
import sys

from . import config
from .jsonl import read_jsonl, write_jsonl


COLUMNS = [
    "id", "headword_raw", "ko", "sense_no", "type",
    "definition_ko", "synonyms_ko",
    "proverb_en_nonthinking", "definition_en_nonthinking",
    "proverb_en_thinking", "definition_en_thinking", "reasoning",
]

LIST_COLUMNS = {"synonyms_ko"}


def build_rows() -> list[dict]:
    source = {r["id"]: r for r in read_jsonl(config.PROVERBS_JSONL)}
    translations = {r["id"]: r for r in read_jsonl(config.TRANSLATIONS_JSONL)}
    thinking = {r["id"]: r for r in read_jsonl(config.TRANSLATIONS_THINKING_JSONL)}
    # Every row needs both translations; a missing one means an input file is
    # absent or incomplete, so fail rather than ship a partial dataset.
    missing = [rid for rid in source if rid not in translations or rid not in thinking]
    if missing:
        raise SystemExit(f"[package] {len(missing):,} rows lack a thinking-off or thinking-on "
                         f"translation (e.g. {missing[:3]}); check {config.TRANSLATIONS_JSONL} "
                         f"and {config.TRANSLATIONS_THINKING_JSONL}.")
    rows: list[dict] = []
    for rid, src in source.items():
        tr = translations[rid]
        th = thinking[rid]
        rows.append({
            "id": rid,
            "headword_raw": src["headword_raw"],
            "ko": src["ko"],
            "sense_no": src["sense_no"],
            "type": src["type"],
            "definition_ko": src["definition_ko"],
            "synonyms_ko": src["synonyms_ko"],
            "proverb_en_nonthinking": tr["proverb_en"],
            "definition_en_nonthinking": tr["definition_en"],
            "proverb_en_thinking": th["proverb_en"],
            "definition_en_thinking": th["definition_en"],
            "reasoning": th["reasoning"],
        })
    return rows


def write_jsonl_csv() -> None:
    rows = build_rows()
    jsonl_path = config.OUTPUT / "korean_proverbs_en.jsonl"
    csv_path = config.OUTPUT / "korean_proverbs_en.csv"
    parquet_path = config.OUTPUT / "korean_proverbs_en.parquet"
    write_jsonl(jsonl_path, rows)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            row = {}
            for c in COLUMNS:
                v = r.get(c)
                if c in LIST_COLUMNS:
                    v = json.dumps(v, ensure_ascii=False)
                row[c] = v
            w.writerow(row)
    # Parquet: pandas handles list columns natively (as object arrays), so no
    # need to pre-serialize. Keep COLUMNS order.
    try:
        import pandas as pd
        df = pd.DataFrame([{c: r.get(c) for c in COLUMNS} for r in rows])
        df.to_parquet(parquet_path, index=False, compression="zstd")
        print(f"[package] wrote {len(rows):,} rows → {jsonl_path} / {csv_path} / {parquet_path}",
              file=sys.stderr)
    except Exception as e:
        print(f"[package] parquet export skipped ({e}); install pandas+pyarrow in research extras.",
              file=sys.stderr)
        print(f"[package] wrote {len(rows):,} rows → {jsonl_path} / {csv_path}", file=sys.stderr)


def main() -> int:
    config.ensure_dirs()
    write_jsonl_csv()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
