#!/usr/bin/env bash
# setup_stage1_data.sh
# Downloads LLaVA-CC3M-Pretrain-595K, tokenizes via prepare.sh → parquet,
# then deletes raw images + HF cache. Only ~5-10GB parquet remains.
#
# Usage: bash setup_stage1_data.sh [HF_TOKEN] [TOKENIZER] [GPUS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HF_TOKEN="${1:-${HF_TOKEN:-}}"
TOKENIZER="${2:-google/gemma-3-270m}"
export GPUS="${3:-${GPUS:-1}}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN="$HF_TOKEN"

IMAGE_DIR="${SCRIPT_DIR}/local_playground/stage1_data/cc3m_595k/images"
JSONL_PATH="${SCRIPT_DIR}/local_playground/stage1_data/cc3m_595k/annotations.jsonl"
RECIPE_PATH="${SCRIPT_DIR}/local_playground/recipe/stage1.json"
PREPARED_RECIPE="${SCRIPT_DIR}/local_playground/recipe/stage1.prepared.json"

echo "========================================="
echo " Stage 1 data setup"
echo "  tokenizer : $TOKENIZER"
echo "  GPUs      : $GPUS"
echo "  images    : $IMAGE_DIR  (deleted after prepare)"
echo "  parquet   : local_playground/prepared_data/  (kept)"
echo "========================================="

mkdir -p "${IMAGE_DIR}" "$(dirname "${RECIPE_PATH}")"

# ── STEP 1: stream download → images + jsonl ─────────────────────────────────
IMAGE_COUNT=$(find "${IMAGE_DIR}" -type f 2>/dev/null | wc -l || echo 0)
if [ -f "${JSONL_PATH}" ] && [ "${IMAGE_COUNT}" -gt 1000 ]; then
    echo "[1/3] annotations.jsonl + images already exist (${IMAGE_COUNT} images) — skipping download."
else
    echo "[1/3] Streaming download from HuggingFace (no full cache written)..."
    python3 - <<EOF
import json
from pathlib import Path
from datasets import load_dataset

image_dir = Path("${IMAGE_DIR}")
jsonl_path = Path("${JSONL_PATH}")

ds = load_dataset(
    "liuhaotian/LLaVA-CC3M-Pretrain-595K",
    split="train",
    streaming=True,
)

written = 0
skipped = 0
with open(jsonl_path, "w") as f:
    for i, item in enumerate(ds):
        img = item.get("image")
        convs = item.get("conversations")
        if img is None or not convs:
            skipped += 1
            continue
        img_filename = f"{i:07d}.jpg"
        img.convert("RGB").save(image_dir / img_filename, "JPEG", quality=95)
        f.write(json.dumps({"image": img_filename, "conversations": convs}) + "\n")
        written += 1
        if written % 10000 == 0:
            print(f"  {written} written...", flush=True)

print(f"Done: {written} written, {skipped} skipped.")
EOF
    echo "[1/3] Download complete."
fi

# ── STEP 2: write raw recipe json ─────────────────────────────────────────────
if [ ! -f "${RECIPE_PATH}" ]; then
    echo "[2/3] Writing stage1.json recipe..."
    python3 -c "
import json
recipe = {'cc3m_595k': {'annotation': '${JSONL_PATH}', 'root': '${IMAGE_DIR}', 'repeat_time': 1}}
open('${RECIPE_PATH}', 'w').write(json.dumps(recipe, indent=2))
print('  Written: ${RECIPE_PATH}')
"
else
    echo "[2/3] Recipe already exists — skipping."
fi

# ── STEP 3: tokenize + tile + pack → parquet (auto-writes stage1.prepared.json)
if [ -f "${PREPARED_RECIPE}" ]; then
    echo "[3/3] Prepared recipe already exists — skipping prepare."
else
    echo "[3/3] Running prepare (tokenize + image tiling → parquet)..."
    pip install -e . -q
    bash shell/prepare.sh "${RECIPE_PATH}" 1 work_dirs/prepare_stage1 "${TOKENIZER}"
fi

# ── CLEANUP ───────────────────────────────────────────────────────────────────
echo "Cleaning up raw images and jsonl..."
rm -rf "${SCRIPT_DIR}/local_playground/stage1_data"

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/datasets/liuhaotian___llava-cc3m-pretrain-595k"
[ -d "$HF_CACHE" ] && rm -rf "$HF_CACHE" && echo "  Removed HF cache: $HF_CACHE"

echo ""
echo "========================================="
echo " Done."
echo " Prepared recipe : ${PREPARED_RECIPE}"
echo " Parquet data    : local_playground/prepared_data/"
du -sh "${SCRIPT_DIR}/local_playground/prepared_data/" 2>/dev/null || true
echo "========================================="
