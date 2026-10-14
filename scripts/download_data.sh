#!/usr/bin/env bash
# Step 1: Clone the Urimalsaem XML dump, pinned to commit 42c0d01, opendict/ only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw/korean-dict-nikl"
COMMIT="42c0d01"

mkdir -p "$ROOT/data/raw"

if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
    https://github.com/spellcheck-ko/korean-dict-nikl "$DEST"
fi

cd "$DEST"
git -c advice.detachedHead=false checkout "$COMMIT" -- opendict/

echo "Fetched opendict/ at $COMMIT:"
ls -1 opendict/ | head
echo "Total XML files: $(find opendict -name '*.xml' | wc -l)"
du -sh opendict/
