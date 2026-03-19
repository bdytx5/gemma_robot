#!/usr/bin/env bash
# train_google_robot.sh
#
# End-to-end: download Fractal dataset (if needed) + finetune GR00T-N1.6-3B
# for Google robot tasks (SimplerEnv / SIMPLER evaluation).
#
# Usage:
#   bash scripts/train_google_robot.sh
#
# Overrides (env vars):
#   NUM_GPUS       Number of GPUs (default: 1)
#   OUTPUT_DIR     Where to save checkpoints (default: ./output/gr00t-fractal)
#   HF_REPO_ID     If set, push checkpoints to this HF Hub repo
#   MAX_STEPS      Training steps (default: 20000)
#   DATASET_PATH   Override dataset location (skip auto-download)

set -e

# ── Config ────────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/gr00t-fractal"}
MAX_STEPS=${MAX_STEPS:-20000}
BASE_MODEL="nvidia/GR00T-N1.6-3B"
EMBODIMENT="OXE_GOOGLE"

DATASET_PATH=${DATASET_PATH:-"examples/SimplerEnv/fractal20220817_data_lerobot"}
MODALITY_SRC="examples/SimplerEnv/fractal_modality.json"
MODALITY_DST="${DATASET_PATH}/meta/modality.json"

HF_DATASET="IPEC-COMMUNITY/fractal20220817_data_lerobot"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Step 1: Download dataset if not present ───────────────────────────────────
if [ ! -d "$DATASET_PATH/data" ]; then
    echo "[data] Dataset not found at $DATASET_PATH — downloading from HuggingFace..."
    huggingface-cli download \
        --repo-type dataset "$HF_DATASET" \
        --local-dir "$DATASET_PATH" \
        --max-workers 4
    echo "[data] Download complete."
else
    echo "[data] Dataset already present at $DATASET_PATH — skipping download."
fi

# ── Step 2: Copy modality.json if missing ─────────────────────────────────────
if [ ! -f "$MODALITY_DST" ]; then
    echo "[data] Copying modality.json..."
    mkdir -p "$(dirname "$MODALITY_DST")"
    cp "$MODALITY_SRC" "$MODALITY_DST"
else
    echo "[data] modality.json already in place."
fi

# ── Step 3: Convert AV1 videos to H.264 if needed ────────────────────────────
# Check if any video exists and probe the first one
FIRST_VIDEO=$(find "$DATASET_PATH/videos" -name "*.mp4" 2>/dev/null | head -1)
if [ -n "$FIRST_VIDEO" ]; then
    CODEC=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name \
        -of default=nw=1:nk=1 "$FIRST_VIDEO" 2>/dev/null || echo "unknown")
    if [[ "$CODEC" == "av01" || "$CODEC" == "av1" ]]; then
        echo "[data] AV1 videos detected — converting to H.264 (this may take a while)..."
        python examples/SimplerEnv/convert_av1_to_h264.py \
            "$DATASET_PATH" --jobs "$(nproc)"
        echo "[data] Video conversion complete."
    else
        echo "[data] Videos are already H.264 — skipping conversion."
    fi
else
    echo "[data] No videos found yet (may be normal if dataset has no video subfolder)."
fi

# ── Step 4: Create output dir + check HF repo access ─────────────────────────
mkdir -p "$OUTPUT_DIR"

if [ -n "$HF_REPO_ID" ]; then
    echo "[hub] Checking HuggingFace repo: $HF_REPO_ID"
    python3 - <<EOF
from huggingface_hub import HfApi
import sys
api = HfApi()
try:
    api.repo_info(repo_id="$HF_REPO_ID", repo_type="model")
    print(f"Repo '$HF_REPO_ID' exists and is accessible.")
except Exception:
    print(f"Repo '$HF_REPO_ID' not found — creating it...")
    try:
        api.create_repo(repo_id="$HF_REPO_ID", repo_type="model", private=True)
        print(f"Created repo '$HF_REPO_ID'.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
EOF
    if [ $? -ne 0 ]; then
        echo "Aborting: HuggingFace repo check failed."
        exit 1
    fi
    TRAIN_SCRIPT="scripts/finetune_with_hub_push.py"
else
    TRAIN_SCRIPT="gr00t/experiment/launch_finetune.py"
fi

# ── Step 5: Train ─────────────────────────────────────────────────────────────
echo "[train] Starting finetuning: $BASE_MODEL -> $OUTPUT_DIR"
echo "[train] GPUs=$NUM_GPUS  Steps=$MAX_STEPS  Embodiment=$EMBODIMENT"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    $TRAIN_SCRIPT \
    --base_model_path "$BASE_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag $EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps $MAX_STEPS \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 1024 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.5

echo "[done] Checkpoint saved to $OUTPUT_DIR"
