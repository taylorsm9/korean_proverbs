"""Minimal embedding HTTP server for the 4090.

Serves the Step 7 embedding candidates over an OpenAI-compatible-ish
`POST /embed` endpoint. Hot-swaps the loaded model when the request names a
different one — only one model is resident at a time, since 8B embedders
fill most of a 24 GB GPU.

Run on the 4090:
    cd ~/korean_proverb
    .venv/bin/pip install -e '.[research]' fastapi uvicorn[standard]
    .venv/bin/python scripts/embed_server.py --host 0.0.0.0 --port 8800

From the orchestrating host:
    export EMBED_BASE_URL=http://<4090 host>:8800
    make embed MODELS="kure arctic-ko kanana qwen3 comsat"
    make modelcheck

Endpoints:
    GET  /health      → {"ok": true, "loaded": "<key or null>"}
    GET  /models      → list of candidate keys and HF names
    POST /embed       → body {"model": "<key>", "texts": [...], "batch_size": 16}
                        → {"vectors": [[...], ...], "dim": N, "model": "..."}
    POST /unload      → unload the currently loaded model to free VRAM

Vectors are L2-normalized float32 arrays. The server adds the plan's shared
instruction prefix (`Instruct: Given a Korean or English text, ...`) to every
text, matching the local embed path.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Any

# Pull the catalog + instruction from the package so client and server agree.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
from korean_proverb.embed import CANDIDATES, INSTRUCTION, _encode_kanana, _load_st, _encode_st  # noqa: E402

try:
    import torch
except Exception as e:
    print("torch is required on the server:", e, file=sys.stderr)
    raise

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except Exception:
    print("Install fastapi + uvicorn on the server:\n  pip install fastapi 'uvicorn[standard]'", file=sys.stderr)
    raise


app = FastAPI(title="korean-proverb embedding server")

_STATE: dict[str, Any] = {"loaded_key": None, "model": None, "device": None}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unload() -> None:
    _STATE["model"] = None
    _STATE["loaded_key"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(key: str) -> None:
    spec = CANDIDATES[key]
    device = _device()
    print(f"[server] loading {key} ({spec.name}) on {device}", file=sys.stderr, flush=True)
    t0 = time.time()
    if spec.loader == "kanana":
        # For kanana we keep tok+model in _STATE; _encode_kanana handles it per call.
        _STATE["model"] = {"spec": spec, "mode": "kanana"}
    else:
        _STATE["model"] = {"spec": spec, "mode": "st",
                           "obj": _load_st(spec.name, device)}
    _STATE["loaded_key"] = key
    _STATE["device"] = device
    print(f"[server] loaded {key} in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)


class EmbedRequest(BaseModel):
    model: str
    texts: list[str]
    batch_size: int = 16
    instruction: str | None = None   # defaults to plan's shared string


@app.get("/health")
def health() -> dict:
    return {"ok": True, "loaded": _STATE["loaded_key"], "device": _STATE["device"]}


@app.get("/models")
def models() -> dict:
    return {k: {"name": s.name, "license": s.license, "loader": s.loader, "note": s.note}
            for k, s in CANDIDATES.items()}


@app.post("/unload")
def unload() -> dict:
    _unload()
    return {"ok": True}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    if req.model not in CANDIDATES:
        raise HTTPException(400, f"unknown model key: {req.model}; options: {list(CANDIDATES)}")
    if _STATE["loaded_key"] != req.model:
        if _STATE["loaded_key"] is not None:
            print(f"[server] evicting {_STATE['loaded_key']}", file=sys.stderr, flush=True)
            _unload()
        _load(req.model)

    spec = CANDIDATES[req.model]
    device = _STATE["device"]
    instruction = req.instruction if req.instruction is not None else INSTRUCTION
    # Temporarily patch the module's INSTRUCTION so the helpers prefix with
    # the caller's value. (The helpers already handle both loaders.)
    import korean_proverb.embed as embed_mod
    old = embed_mod.INSTRUCTION
    embed_mod.INSTRUCTION = instruction
    try:
        t0 = time.time()
        if _STATE["model"]["mode"] == "kanana":
            vecs = _encode_kanana(spec.name, req.texts, device=device,
                                  batch_size=req.batch_size,
                                  max_length=spec.max_length or 512)
        else:
            vecs = _encode_st(_STATE["model"]["obj"], req.texts,
                              apply_instruction=spec.apply_instruction,
                              batch_size=req.batch_size,
                              max_length=spec.max_length)
        dt = time.time() - t0
    finally:
        embed_mod.INSTRUCTION = old

    print(f"[server] /embed {req.model} n={len(req.texts)} dim={vecs.shape[1]} {dt:.1f}s", file=sys.stderr, flush=True)
    return {"vectors": vecs.tolist(), "dim": int(vecs.shape[1]),
            "model": spec.name, "n": len(req.texts), "elapsed_s": dt}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8800)
    args = p.parse_args()
    print(f"[server] starting on {args.host}:{args.port}, device={_device()}", file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
