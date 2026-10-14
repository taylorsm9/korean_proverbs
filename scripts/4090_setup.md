# Setting up the embedding server on the 4090

The embedding server runs on the 4090 and loads Step 7 candidate models on demand (one at a time; the 24 GB card can hold any single 8B embedder, but not two). The orchestrating host (where this repo lives today) talks to it over HTTP via `EMBED_BASE_URL`, so the model check can rotate through every candidate without any manual back-and-forth.

## 1. Get the repo onto the 4090

From a shell on the 4090 (or using rsync from here):

```bash
mkdir -p ~/korean_proverb
rsync -av --delete \
  --exclude='.venv' --exclude='data/raw' --exclude='data/traces' \
  <orchestrating-host>:/home/scott/PycharmProjects/korean_proverb/ \
  ~/korean_proverb/
```

Only `data/interim/proverbs.jsonl` and `data/output/translations.jsonl` are needed on the 4090 — the server reads both to decide which texts to embed per request, so make sure these are in sync (re-rsync when the main translation run completes).

If HF models need to come from behind a cache, set `HF_HOME` to a disk with plenty of space (Qwen3-Embedding-8B is ~16 GB on disk).

## 2. Install deps

```bash
cd ~/korean_proverb
python3 -m venv .venv
.venv/bin/pip install -e '.[research,embed-server]'
```

If torch doesn't come with the right CUDA build automatically, override with:

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.*'
```

(Use the CUDA version that matches your drivers; `nvidia-smi` shows it.)

## 3. Start the server

```bash
cd ~/korean_proverb
.venv/bin/python scripts/embed_server.py --host 0.0.0.0 --port 8800
```

Or via Make:

```bash
make embed-server
```

Keep it running (nohup/tmux as you prefer). It idles with no model loaded; the first `/embed` request lazy-loads whatever model is asked for. A `POST /unload` or any `/embed` for a different model evicts the previous one and frees VRAM.

Sanity-check from the 4090 itself:

```bash
curl http://localhost:8800/health
curl http://localhost:8800/models | jq
```

## 4. Point this host at it

On the orchestrating host, pick the 4090's reachable address (LAN IP or hostname) and export:

```bash
export EMBED_BASE_URL=http://<4090 host or ip>:8800
```

Verify:

```bash
curl -sS "$EMBED_BASE_URL/health"
```

Then let me (Claude) know and I'll fire the model check.

## Driving the model check from this host

Once `EMBED_BASE_URL` is set, this host runs the full Step 7.0 model check:

```bash
# Embed every candidate (one request per model; the server swaps them in order)
make embed MODELS="kure arctic-ko kanana qwen3 comsat"

# Then the metric comparison
make modelcheck MODELS="kure arctic-ko kanana qwen3 comsat"
```

The embed step POSTs texts up and writes `.npy` caches here under
`data/research/embeddings/<slug>/`. Each vector is L2-normalized float32; a
full set per model is roughly 700 MB over the wire.

The model check reads only the local caches and writes `runs/research/model_check.md`
with per-model MRR, BM25 baseline, synonym AUC, and a `primary_model.json`
pick.

## Shutdown

Nothing special — just stop the server process (Ctrl-C or kill the pid).
Models unload on process exit; HF files stay in cache for next time.
