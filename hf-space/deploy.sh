#!/usr/bin/env bash
# Sync the contents of this directory to a Hugging Face Space repo.
#
# Usage:
#   ./deploy.sh <hf-username>/<space-name>
#
# Example:
#   ./deploy.sh random-walks/citeformer-demo
#
# Prerequisites (one-time):
#   - pip install huggingface_hub
#   - huggingface-cli login   (paste a write-scoped token from
#     https://huggingface.co/settings/tokens)
#   - Create the Space in the web UI at https://huggingface.co/new-space
#     with sdk=gradio, hardware=CPU basic. (This script assumes it exists.)
#
# The script copies app.py, requirements.txt, and README.md to a fresh
# clone of the Space repo, commits with the current citeformer git SHA
# embedded, and pushes. It is idempotent — rerunning just pushes the
# latest state.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <hf-username>/<space-name>" >&2
    exit 2
fi

SPACE_ID="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CITEFORMER_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "→ Cloning https://huggingface.co/spaces/${SPACE_ID}"
git clone "https://huggingface.co/spaces/${SPACE_ID}" "$WORK_DIR/space"

echo "→ Syncing app.py, requirements.txt, README.md"
cp "$HERE/app.py" "$WORK_DIR/space/"
cp "$HERE/requirements.txt" "$WORK_DIR/space/"
cp "$HERE/README.md" "$WORK_DIR/space/"

cd "$WORK_DIR/space"
git add app.py requirements.txt README.md
if git diff --cached --quiet; then
    echo "→ Nothing to commit (Space already matches local state)."
    exit 0
fi

git commit -m "sync from random-walks/citeformer@${CITEFORMER_SHA}"
echo "→ Pushing to hf.co"
git push

echo ""
echo "✓ Deployed. Space should rebuild in ~1-2 min at:"
echo "    https://huggingface.co/spaces/${SPACE_ID}"
