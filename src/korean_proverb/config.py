"""Paths, environment, and constants shared across the pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_OPENDICT = RAW / "korean-dict-nikl" / "opendict"
INTERIM = DATA / "interim"
TRACES = DATA / "traces"
OUTPUT = DATA / "output"
RESEARCH = DATA / "research"
EMBED_CACHE = RESEARCH / "embeddings"

RUNS = ROOT / "runs"
PROMPTS = RUNS / "prompts"
FIGURES = RUNS / "research" / "figures"

PROVERBS_JSONL = INTERIM / "proverbs.jsonl"
TRANSLATIONS_JSONL = OUTPUT / "translations.jsonl"
TRANSLATIONS_THINKING_JSONL = OUTPUT / "translations_thinking.jsonl"
ATTEMPTS_JSONL = TRACES / "attempts.jsonl"
JUDGE_ATTEMPTS_JSONL = TRACES / "judge_attempts.jsonl"

MODE_FILE = RUNS / "mode.json"

TRANSLATION_MODEL = "nvidia/Qwen3.5-122B-A10B-NVFP4"
JUDGE_MODELS = {
    "gemma": "google/gemma-4-26B-A4B-it",
    "hyperclova": "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
}

EMBED_MODELS = ["BAAI/bge-m3", "Qwen/Qwen3-Embedding-4B"]

EXAMPLE_IDS = {"601294", "612202", "607368"}


def qwen_base_url() -> str:
    return os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")


def judge_base_url() -> str | None:
    return os.environ.get("JUDGE_BASE_URL")


def ensure_dirs() -> None:
    for p in (DATA, RAW, INTERIM, TRACES, OUTPUT, RESEARCH, EMBED_CACHE, RUNS, PROMPTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
