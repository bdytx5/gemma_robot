#!/usr/bin/env bash
# Download IPEC-COMMUNITY/fractal20220817_data_lerobot dataset.
#
# Usage: bash scripts/download_fractal.sh [DATASET_PATH] [--legacy]
#
#   Default: git lfs clone (reliable, resumes, no rate limit hell)
#   --legacy: use huggingface_hub snapshot_download
#
# Set HF_TOKEN env var for authenticated requests.

HF_DATASET="IPEC-COMMUNITY/fractal20220817_data_lerobot"
HF_REPO_URL="https://huggingface.co/datasets/$HF_DATASET"
PYTHON=${PYTHON:-"$(dirname "$0")/../.venv/bin/python"}
HANG_TIMEOUT=${HANG_TIMEOUT:-1800}

# Parse args
LEGACY=0
DATASET_PATH=""
for arg in "$@"; do
    if [ "$arg" = "--legacy" ]; then
        LEGACY=1
    elif [ -z "$DATASET_PATH" ]; then
        DATASET_PATH="$arg"
    fi
done
DATASET_PATH=${DATASET_PATH:-examples/SimplerEnv/fractal20220817_data_lerobot}

echo "[download] Saving to: $DATASET_PATH"
echo "[download] Repo: $HF_DATASET"

if [ -z "$HF_TOKEN" ]; then
    read -rp "[download] HuggingFace token (leave blank to skip): " HF_TOKEN
fi
[ -n "$HF_TOKEN" ] && echo "[download] HF_TOKEN set — authenticated mode"

# ── git lfs mode (default) ────────────────────────────────────────────────────
if [ "$LEGACY" = "0" ]; then
    echo "[download] Mode: git lfs"

    # Ensure git-lfs is available
    if ! command -v git-lfs &>/dev/null; then
        echo "[download] Installing git-lfs..."
        sudo apt-get install -y git-lfs
        git lfs install
    fi

    GIT_URL="$HF_REPO_URL"
    if [ -n "$HF_TOKEN" ]; then
        GIT_URL="https://user:${HF_TOKEN}@huggingface.co/datasets/$HF_DATASET"
    fi

    if [ ! -d "$DATASET_PATH/.git" ]; then
        if [ -d "$DATASET_PATH" ]; then
            # Exists but not a git repo (leftover from snapshot_download) — init in place
            echo "[download] Existing non-git dir found — initialising git repo in place..."
            git -C "$DATASET_PATH" init
            git -C "$DATASET_PATH" lfs install
            git -C "$DATASET_PATH" remote add origin "$GIT_URL"
            GIT_LFS_SKIP_SMUDGE=1 git -C "$DATASET_PATH" fetch origin main
            GIT_LFS_SKIP_SMUDGE=1 git -C "$DATASET_PATH" checkout -f FETCH_HEAD
        else
            echo "[download] Cloning repo (metadata only)..."
            GIT_LFS_SKIP_SMUDGE=1 git clone "$GIT_URL" "$DATASET_PATH"
        fi
    else
        echo "[download] Repo already cloned — fetching updates..."
        git -C "$DATASET_PATH" remote set-url origin "$GIT_URL"
        GIT_LFS_SKIP_SMUDGE=1 git -C "$DATASET_PATH" pull
    fi

    echo "[download] Pulling LFS files..."
    cd "$DATASET_PATH"
    while true; do
        GIT_LFS_SKIP_SMUDGE=0 git lfs pull && break
        echo "[download] git lfs pull failed — retrying..."
    done
    echo "[download] Done."
    exit 0
fi

# ── legacy snapshot_download mode ────────────────────────────────────────────
echo "[download] Mode: legacy (snapshot_download)"

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[download] Attempt $ATTEMPT — starting snapshot_download..."

    timeout "$HANG_TIMEOUT" "$PYTHON" -u - <<PYEOF
import os, sys
from huggingface_hub import HfApi, snapshot_download

token = os.environ.get("HF_TOKEN") or None
repo_id = "$HF_DATASET"

try:
    api = HfApi()
    api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
except Exception as e:
    msg = str(e)
    if any(k in msg for k in ("429", "rate limit", "quota", "too many requests")):
        print(f"[download] Rate limited on preflight: {msg[:200]}")
        sys.exit(42)
    print(f"[download] Preflight error: {msg[:300]}", file=sys.stderr)
    sys.exit(1)

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="$DATASET_PATH",
        max_workers=4,
        token=token,
    )
    print("[download] snapshot_download completed successfully.")
    sys.exit(0)
except Exception as e:
    msg = str(e)
    if any(k in msg for k in ("429", "rate limit", "quota", "too many requests")):
        print(f"[download] Rate limited: {msg[:200]}")
        sys.exit(42)
    print(f"[download] Error: {msg[:300]}", file=sys.stderr)
    sys.exit(1)
PYEOF
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
        echo "[download] Done."
        break
    elif [ $EXIT -eq 124 ]; then
        echo "[download] HUNG after ${HANG_TIMEOUT}s — retrying..."
    elif [ $EXIT -eq 42 ]; then
        echo "[download] Rate limited — retrying..."
    else
        echo "[download] Failed (exit $EXIT) — retrying..."
    fi
done
