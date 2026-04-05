#!/usr/bin/env bash
# download_test_dataset.sh
#
# Download a small subset of the fractal (Google Robot) dataset for smoke-testing
# your setup on a new machine. Downloads only 1 chunk (~1000 episodes) instead
# of the full 87k episodes.
#
# Usage:
#   bash scripts/download_test_dataset.sh
#
# After downloading, run a quick training test:
#   bash scripts/train_balanced_from_270m.sh --test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HF_DATASET="IPEC-COMMUNITY/fractal20220817_data_lerobot"
DATASET_PATH="$REPO_ROOT/examples/SimplerEnv/fractal20220817_data_lerobot"

echo "============================================"
echo " Download test dataset (fractal, 1 chunk)"
echo "============================================"
echo "[config] Source: $HF_DATASET"
echo "[config] Dest:   $DATASET_PATH"
echo ""

# Check if full dataset already exists
if [ -f "$DATASET_PATH/meta/info.json" ]; then
    echo "[skip] Dataset already exists at $DATASET_PATH"
    echo "       To re-download, remove the directory first."
    exit 0
fi

# Ensure git-lfs is available
if ! command -v git-lfs &>/dev/null; then
    echo "[setup] Installing git-lfs..."
    sudo apt-get install -y git-lfs
    git lfs install
fi

# Use huggingface-cli to download only chunk-000 (meta + 1 data chunk + 1 video chunk)
echo "[download] Downloading metadata + chunk-000 only..."
HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download --repo-type dataset "$HF_DATASET" \
    --include "meta/*" "data/chunk-000/*" "videos/chunk-000/*" "README.md" \
    --local-dir "$DATASET_PATH"

# Patch info.json to reflect the subset
if [ -f "$DATASET_PATH/meta/info.json" ]; then
    echo "[patch] Updating info.json for subset..."
    python3 -c "
import json
info_path = '$DATASET_PATH/meta/info.json'
with open(info_path) as f:
    info = json.load(f)
# Update episode count to just chunk-000 (1000 episodes)
info['splits'] = {'train': '0:1000'}
info['_note'] = 'Subset: chunk-000 only, for testing'
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print(f'  total_episodes: {info.get(\"total_episodes\")} -> subset: 1000')
"
    # Trim episodes.jsonl to first 1000
    if [ -f "$DATASET_PATH/meta/episodes.jsonl" ]; then
        head -1000 "$DATASET_PATH/meta/episodes.jsonl" > "$DATASET_PATH/meta/episodes_subset.jsonl"
        mv "$DATASET_PATH/meta/episodes_subset.jsonl" "$DATASET_PATH/meta/episodes.jsonl"
        echo "[patch] Trimmed episodes.jsonl to 1000 entries"
    fi
fi

# Copy modality.json
MODALITY_SRC="$REPO_ROOT/examples/SimplerEnv/fractal_modality.json"
if [ -f "$MODALITY_SRC" ]; then
    cp "$MODALITY_SRC" "$DATASET_PATH/meta/modality.json"
    echo "[setup] Copied modality.json"
elif [ ! -f "$DATASET_PATH/meta/modality.json" ]; then
    echo "[warn] No modality.json found — training may fail. Check examples/SimplerEnv/fractal_modality.json"
fi

echo ""
echo "============================================"
echo " Test dataset ready!"
echo "============================================"
SIZE=$(du -sh "$DATASET_PATH" | cut -f1)
echo "  Path: $DATASET_PATH"
echo "  Size: $SIZE"
echo ""
echo "Run a smoke test:"
echo "  bash scripts/train_balanced_from_270m.sh --test"
