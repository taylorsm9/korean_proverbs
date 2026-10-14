"""Step 7 embeddings. Loads the plan's candidate models, applies the shared
instruction prefix, and caches L2-normalized vectors per row.

Candidates (from plan.md Step 7):
- sionic-ai/comsat-embed-ko-8b-preview (Qwen3-Embedding-8B fine-tune; CC-BY-NC)
- nlpai-lab/KURE-v1 (bge-m3 Korean fine-tune; MIT)
- dragonkue/snowflake-arctic-embed-l-v2.0-ko (Apache 2.0)
- kakaocorp/kanana-nano-2.1b-embedding (CC-BY-NC 4.0)
- Qwen/Qwen3-Embedding-8B (Apache 2.0; reference and fallback)

Every candidate embeds Korean and English with the **same instruction string**.
Kanana uses mean-pooling with `include_prompt=False`; the others use the
sentence-transformers default for the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .jsonl import read_jsonl

FIELDS = ("ko", "definition_ko", "proverb_en", "definition_en")


def embed_base_url() -> str | None:
    return os.environ.get("EMBED_BASE_URL")

INSTRUCTION = (
    "Instruct: Given a Korean or English text, retrieve texts with the same meaning\n"
    "Query: "
)


@dataclass
class ModelSpec:
    name: str                 # HF repo id
    license: str
    loader: str = "sentence_transformers"   # or "kanana"
    apply_instruction: bool = True
    max_length: int | None = None
    note: str = ""
    # Per plan.md Step 7, every text (query + corpus) gets the same prompt.
    # For qwen3/comsat the plan prescribes a custom instruction string, passed
    # as the server's `prompt` field. For other models we use `prompt_name` and
    # let the server apply the model-card's own query prompt.
    custom_prompt: str | None = None
    prompt_name: str = "query"


# Plan.md Step 7 custom instruction for Qwen3-Embedding and comsat (verbatim,
# including the trailing space).
QWEN_INSTRUCT = (
    "Instruct: Given a Korean or English text, retrieve texts with the same meaning\n"
    "Query: "
)

# Candidate catalog.
CANDIDATES: dict[str, ModelSpec] = {
    "kanana": ModelSpec(
        name="kakaocorp/kanana-nano-2.1b-embedding",
        license="CC-BY-NC-4.0",
        loader="kanana",
        max_length=512,
        note="Kakao bilingual (Korean + English); primary model. Server handles include_prompt=False pooling.",
    ),
    "qwen3": ModelSpec(
        name="Qwen/Qwen3-Embedding-8B",
        license="Apache-2.0",
        custom_prompt=QWEN_INSTRUCT,
        note="Used as the robustness model.",
    ),
}


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _cache_dir(model_name: str) -> Path:
    d = config.EMBED_CACHE / _slug(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _prefix(texts: list[str]) -> list[str]:
    return [INSTRUCTION + t for t in texts]


# ---------- loaders ----------

def _load_st(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def _encode_st(model, texts: list[str], apply_instruction: bool, batch_size: int, max_length: int | None) -> np.ndarray:
    inputs = _prefix(texts) if apply_instruction else texts
    kw = dict(batch_size=batch_size, normalize_embeddings=True,
              convert_to_numpy=True, show_progress_bar=True)
    if max_length is not None:
        try:
            model.max_seq_length = max_length
        except Exception:
            pass
    vecs = model.encode(inputs, **kw)
    return vecs.astype(np.float32)


def _encode_kanana(model_name: str, texts: list[str], device: str, batch_size: int, max_length: int) -> np.ndarray:
    """Kanana embedder: mean-pool, include_prompt=False, trust_remote_code.

    Kanana's model card loads via `AutoModel(..., trust_remote_code=True)` with
    a custom embedding head that supports `include_prompt=False` in its forward.
    If the custom head isn't available in this installation, we fall back to
    manual mean-pool over non-padding tokens, after stripping the instruction.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    prefixed = _prefix(texts)
    with torch.inference_mode():
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            # Try the custom forward with include_prompt=False first
            try:
                emb = model(**enc, include_prompt=False)
            except TypeError:
                hs = model(**enc, output_hidden_states=False).last_hidden_state
                # Strip the prompt tokens from the mean pool: compute how many
                # prompt tokens are in each example (same for all since prefix is fixed)
                prompt_ids = tok(INSTRUCTION, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
                mask = enc.attention_mask.clone()
                # Zero out the first `prompt_ids` positions after the special tokens
                # (approximate; good enough for a fallback).
                mask[:, :prompt_ids] = 0
                emb = (hs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out[i:i + batch_size] = emb.cpu().numpy().astype(np.float32)
    return out


def _encode(spec: ModelSpec, texts: list[str], device: str, batch_size: int) -> np.ndarray:
    if spec.loader == "kanana":
        return _encode_kanana(spec.name, texts, device=device, batch_size=batch_size,
                              max_length=spec.max_length or 512)
    model = _load_st(spec.name, device)
    return _encode_st(model, texts, apply_instruction=spec.apply_instruction,
                      batch_size=batch_size, max_length=spec.max_length)


# ---------- remote server client ----------
#
# The 4090 embed server (scripts/embed_server.py on that host) accepts:
#     POST /embed  {model, texts, prompt_name?, batch_size?, format?}
# where prompt_name is "query" | "document" | null, defaulting to "document".
# Per plan.md Step 7 we apply each model's own *query* format identically to
# every text — ko, definition_ko, proverb_en, definition_en all take
# prompt_name="query", so pairs stay comparable in a single space.
#
# Response format is `npy` to avoid float->JSON->float round-tripping on 8B
# embedders producing 4,096-dim vectors (~16 KB each in binary vs ~70 KB in
# JSON).

def _encode_remote(spec: "ModelSpec", spec_key: str, texts: list[str], batch_size: int,
                   base_url: str, chunk_size: int = 512) -> np.ndarray:
    """POST to the 4090 embed server. Per spec, either passes a custom prompt
    string (qwen3/comsat) or `prompt_name` so the server applies the
    model-card's own query prompt (kure/arctic-ko/kanana)."""
    import io

    import httpx
    url = base_url.rstrip("/") + "/embed"
    out_chunks: list[np.ndarray] = []
    base_payload: dict = {"model": spec_key, "format": "npy"}
    if spec.custom_prompt is not None:
        base_payload["prompt"] = spec.custom_prompt
    else:
        base_payload["prompt_name"] = spec.prompt_name
    with httpx.Client(timeout=1800.0) as client:
        for start in range(0, len(texts), chunk_size):
            batch = texts[start:start + chunk_size]
            payload = dict(base_payload, texts=batch)
            if batch_size:
                payload["batch_size"] = batch_size
            r = client.post(url, json=payload)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "octet-stream" in ctype or "numpy" in ctype or r.content[:6] == b"\x93NUMPY":
                arr = np.load(io.BytesIO(r.content), allow_pickle=False)
            else:
                data = r.json()
                arr = np.array(data.get("embeddings") or data.get("vectors"), dtype=np.float32)
            out_chunks.append(arr.astype(np.float32))
            print(f"[embed-remote] {spec_key} {start + len(batch):,}/{len(texts):,}",
                  file=sys.stderr, flush=True)
    return np.concatenate(out_chunks, axis=0)


def _unload_remote(base_url: str) -> None:
    import httpx
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(base_url.rstrip("/") + "/unload")
    except Exception as e:
        print(f"[embed-remote] /unload failed (non-fatal): {e}", file=sys.stderr)


# ---------- public API ----------

def _field_text(field_name: str, source_row: dict, translation_row: dict | None) -> str:
    if field_name in ("ko", "definition_ko"):
        return source_row.get(field_name, "")
    if translation_row is None:
        return ""
    return translation_row.get(field_name, "")


def _load_sidecar_ids(cache: Path, field_name: str) -> list[str]:
    p = cache / f"{field_name}.ids.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _append_vectors(cache: Path, field_name: str, new_vecs: np.ndarray, new_ids: list[str]) -> None:
    """Append `new_vecs` (M, D) and `new_ids` to the on-disk cache for this field."""
    npy_path = cache / f"{field_name}.npy"
    ids_path = cache / f"{field_name}.ids.json"
    if npy_path.exists():
        existing = np.load(npy_path)
        if existing.shape[1] != new_vecs.shape[1]:
            raise RuntimeError(f"dim mismatch for {npy_path}: {existing.shape[1]} vs {new_vecs.shape[1]}")
        combined = np.concatenate([existing, new_vecs], axis=0)
    else:
        combined = new_vecs
    existing_ids = _load_sidecar_ids(cache, field_name)
    combined_ids = existing_ids + new_ids
    assert len(combined_ids) == combined.shape[0], \
        f"id/vector count mismatch for {field_name}: {len(combined_ids)} vs {combined.shape[0]}"
    np.save(npy_path, combined)
    ids_path.write_text(json.dumps(combined_ids), encoding="utf-8")


def embed_all(spec: ModelSpec, spec_key: str, batch_size: int = 16, device: str | None = None,
              fields: tuple[str, ...] = FIELDS, remote_url: str | None = None) -> None:
    """Embed each requested field for every row in proverbs.jsonl that has
    content for that field, using translations.jsonl for English fields.

    Partial-fill resumable: for each field we maintain a `{field}.ids.json`
    sidecar listing which row ids are already embedded; subsequent calls only
    embed the delta (new translations as they arrive). The vectors array
    `{field}.npy` grows in step with the sidecar.
    """
    source = list(read_jsonl(config.PROVERBS_JSONL))
    translations = {r["id"]: r for r in read_jsonl(config.TRANSLATIONS_JSONL)}
    all_ids = [r["id"] for r in source]
    cache = _cache_dir(spec.name)
    (cache / "ids.json").write_text(json.dumps(all_ids), encoding="utf-8")
    (cache / "model.json").write_text(json.dumps({
        "name": spec.name, "license": spec.license, "loader": spec.loader,
        "custom_prompt": spec.custom_prompt,
        "prompt_name": None if spec.custom_prompt else spec.prompt_name,
        "note": spec.note,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if remote_url:
        print(f"[embed] {spec.name} via {remote_url}", file=sys.stderr)
    else:
        dev = device or _pick_device()
        print(f"[embed] {spec.name} on {dev}", file=sys.stderr)

    did_any_work = False
    for field_name in fields:
        already = set(_load_sidecar_ids(cache, field_name))
        # Build (id, text) for every row that has non-empty content and hasn't been embedded yet.
        pairs: list[tuple[str, str]] = []
        for r in source:
            if r["id"] in already:
                continue
            text = _field_text(field_name, r, translations.get(r["id"]))
            if text:
                pairs.append((r["id"], text))
        total_available = sum(1 for r in source
                              if _field_text(field_name, r, translations.get(r["id"])))
        if not pairs:
            print(f"[embed] {spec.name} {field_name}: nothing new to embed "
                  f"({len(already):,}/{total_available:,} already cached, "
                  f"{len(all_ids) - total_available:,} rows have no text yet)",
                  file=sys.stderr)
            continue
        print(f"[embed] {spec.name} {field_name}: embedding {len(pairs):,} new rows "
              f"({len(already):,} already cached; {total_available - len(already) - len(pairs):,} "
              f"still waiting for text)", file=sys.stderr)
        texts = [t for _, t in pairs]
        ids = [i for i, _ in pairs]
        if remote_url:
            vecs = _encode_remote(spec, spec_key, texts, batch_size=batch_size, base_url=remote_url)
        else:
            vecs = _encode(spec, texts, dev, batch_size)
        _append_vectors(cache, field_name, vecs, ids)
        did_any_work = True
        print(f"[embed] {spec.name} {field_name}: wrote {len(ids):,} rows → "
              f"cache now has {len(already) + len(ids):,} / {total_available:,}",
              file=sys.stderr)
    if remote_url and did_any_work:
        _unload_remote(remote_url)


def load_field(model_name: str, field: str) -> tuple[list[str], np.ndarray]:
    """Load the per-field cache. Returns (ids, vecs) where `ids[k]` is the row
    id for `vecs[k]`. Only rows with non-empty text at that field will appear.
    """
    cache = _cache_dir(model_name)
    npy_path = cache / f"{field}.npy"
    ids_path = cache / f"{field}.ids.json"
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)
    vecs = np.load(npy_path)
    if ids_path.exists():
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
    else:
        # Legacy pre-sidecar file: aligned with the top-level ids.json
        ids = json.loads((cache / "ids.json").read_text(encoding="utf-8"))
    assert len(ids) == vecs.shape[0], \
        f"ids/vectors mismatch for {model_name} {field}: {len(ids)} vs {vecs.shape[0]}"
    return ids, vecs


def load_cached(model_name: str, field: str) -> tuple[list[str], np.ndarray]:
    """Legacy loader. Prefer `load_field()` which respects per-field sidecars."""
    return load_field(model_name, field)


def embed_texts(spec_key: str, texts: list[str], ids: list[str], cache_name: str,
                 remote_url: str | None = None, batch_size: int = 32,
                 device: str | None = None) -> None:
    """Embed an arbitrary list of (id, text) pairs and cache under cache_name.

    Used for Step 7 research arms (N1, A2, …) whose texts aren't in
    proverbs.jsonl or translations.jsonl. The cache is partial-fill resumable
    the same way `embed_all` is: `{cache_name}.ids.json` records which ids
    are already embedded, so repeat calls only embed the delta.
    """
    assert len(texts) == len(ids), f"text/id length mismatch: {len(texts)} vs {len(ids)}"
    if spec_key not in CANDIDATES:
        raise ValueError(f"unknown model key: {spec_key}")
    spec = CANDIDATES[spec_key]
    cache = _cache_dir(spec.name)
    already = set(_load_sidecar_ids(cache, cache_name))
    pairs = [(i, t) for i, t in zip(ids, texts) if i not in already and t]
    if not pairs:
        print(f"[embed] {spec.name} {cache_name}: nothing new", file=sys.stderr)
        return
    new_ids = [p[0] for p in pairs]
    new_texts = [p[1] for p in pairs]
    print(f"[embed] {spec.name} {cache_name}: embedding {len(pairs):,} new rows "
          f"({len(already):,} already cached)", file=sys.stderr)
    if remote_url:
        vecs = _encode_remote(spec, spec_key, new_texts, batch_size=batch_size, base_url=remote_url)
    else:
        dev = device or _pick_device()
        vecs = _encode(spec, new_texts, dev, batch_size)
    _append_vectors(cache, cache_name, vecs, new_ids)


def load_aligned(model_name: str, field: str, target_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (vecs, mask) where vecs[i] is the embedding for target_ids[i]
    and mask[i] is True iff that id is in the field's sidecar.

    Rows without a cached embedding get a zero vector (so dot-products behave
    sensibly) and mask=False, so callers can filter.
    """
    ids, vecs = load_field(model_name, field)
    index: dict[str, int] = {rid: k for k, rid in enumerate(ids)}
    D = vecs.shape[1]
    out = np.zeros((len(target_ids), D), dtype=vecs.dtype)
    mask = np.zeros(len(target_ids), dtype=bool)
    for i, rid in enumerate(target_ids):
        k = index.get(rid)
        if k is not None:
            out[i] = vecs[k]
            mask[i] = True
    return out, mask


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["kure"],
                   help="Keys from the CANDIDATES catalog, e.g. kure arctic-ko kanana qwen3 comsat.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, help="Local device (ignored when --server/EMBED_BASE_URL set).")
    p.add_argument("--server", default=None,
                   help="Base URL of a remote embedding server (overrides EMBED_BASE_URL).")
    p.add_argument("--fields", nargs="+", default=list(FIELDS))
    p.add_argument("--list", action="store_true", help="List available models and exit.")
    args = p.parse_args()
    if args.list:
        for k, s in CANDIDATES.items():
            print(f"{k:12s} {s.name:55s} {s.license:15s} {s.note}")
        return 0
    config.ensure_dirs()
    remote = args.server or embed_base_url()
    for key in args.models:
        if key not in CANDIDATES:
            print(f"[embed] unknown model key: {key}. Options: {', '.join(CANDIDATES)}", file=sys.stderr)
            return 1
        embed_all(CANDIDATES[key], key,
                  batch_size=args.batch_size, device=args.device,
                  fields=tuple(args.fields), remote_url=remote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
