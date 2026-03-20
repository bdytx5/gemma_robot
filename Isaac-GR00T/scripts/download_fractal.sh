#!/usr/bin/env bash
# Download IPEC-COMMUNITY/fractal20220817_data_lerobot dataset, retrying on 429.
# Usage: bash scripts/download_fractal.sh [DATASET_PATH]
set -e

DATASET_PATH=${1:-examples/SimplerEnv/fractal20220817_data_lerobot}
HF_DATASET="IPEC-COMMUNITY/fractal20220817_data_lerobot"
PYTHON=${PYTHON:-"$(dirname "$0")/../.venv/bin/python"}

echo "[download] Saving to: $DATASET_PATH"
echo "[download] Repo: $HF_DATASET"

"$PYTHON" -u - <<EOF
import time
from huggingface_hub import snapshot_download

while True:
    try:
        print("[download] Starting snapshot_download...")
        snapshot_download(
            repo_id="$HF_DATASET",
            repo_type="dataset",
            local_dir="$DATASET_PATH",
            max_workers=32,
        )
        print("[download] Done.")
        break
    except Exception as e:
        if "429" in str(e):
            print(f"[download] Rate limited. Waiting 300s...")
            time.sleep(300)
        else:
            raise
EOF
