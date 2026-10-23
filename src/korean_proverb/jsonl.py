"""Small JSONL helpers used all over the pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    import sys
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # Tolerate a torn final line (e.g. power loss mid-append)
                # so resume can proceed. Log once per file to stderr.
                print(f"[read_jsonl] skipping malformed line {lineno} in {path}: {e}",
                      file=sys.stderr)
                continue


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def accepted_ids(path: Path) -> set[str]:
    """Return the set of ids with an accepted translation row."""
    return {r["id"] for r in read_jsonl(path)}
