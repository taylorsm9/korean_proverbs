#!/usr/bin/env bash
# Thinking research chain (plan_reasoning_embeddings.md, F then G). Run on the 4090 in tmux:
#   tmux new -d -s research 'scripts/thinking_research.sh 2>&1 | tee -a runs/thinking_research.stdout.log'
# Each stage runs twice: the second pass picks up rows that hit a hard error in the first
# (resume skips accepted ids, so it's a no-op when the first pass was clean).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export QWEN_BASE_URL=${QWEN_BASE_URL:-http://192.168.68.52:8000/v1}

want=nvidia/Qwen3.5-122B-A10B-NVFP4
got=$(curl -s --max-time 10 "$QWEN_BASE_URL/models" | python3 -c 'import sys,json; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' || true)
[ "$got" = "$want" ] || { echo "server at $QWEN_BASE_URL serves '$got', expected '$want'" >&2; exit 1; }

run() { for pass in 1 2; do .venv/bin/python -m korean_proverb.thinking_research "$@"; done; }
run reseed --subset sample
run nodef --subset all
run nodef-meaning --subset sample
run nodef-meaning --subset all
echo "[$(date '+%F %T')] all stages done"
