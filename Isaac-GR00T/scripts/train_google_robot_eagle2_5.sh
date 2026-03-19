#!/usr/bin/env bash
# train_google_robot_eagle2_5.sh
#
# End-to-end: download Fractal dataset (if needed) + finetune GR00T with your
# custom Eagle2.5 + Gemma3-270m backbone for Google robot tasks.
#
# Prerequisites:
#   - Eagle2.5 repo at ../Eagle/Eagle2_5 (relative to Isaac-GR00T/)
#   - Stage 2 checkpoint at HF: youngbrett48/train_stage2_gemma3_270m.sh
#     (or set BASE_MODEL_PATH to a local path)
#
# Usage:
#   bash scripts/train_google_robot_eagle2_5.sh
#
# Overrides:
#   NUM_GPUS        Number of GPUs (default: 1)
#   OUTPUT_DIR      Checkpoint output dir (default: ./output/gr00t-eagle2_5-fractal)
#   BASE_MODEL_PATH Eagle2.5 checkpoint (default: HF repo above)
#   HF_REPO_ID      If set, push checkpoints here
#   MAX_STEPS       Training steps (default: 20000)
#   DATASET_PATH    Override dataset location (skip auto-download)

set -e

# ── Config ────────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/gr00t-eagle2_5-fractal"}
MAX_STEPS=${MAX_STEPS:-20000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"youngbrett48/train_stage2_gemma3_270m.sh"}
EMBODIMENT="OXE_GOOGLE"
BACKBONE_EMBEDDING_DIM=1152  # Gemma3-270m hidden_size

DATASET_PATH=${DATASET_PATH:-"examples/SimplerEnv/fractal20220817_data_lerobot"}
MODALITY_SRC="examples/SimplerEnv/fractal_modality.json"
MODALITY_DST="${DATASET_PATH}/meta/modality.json"

HF_DATASET="IPEC-COMMUNITY/fractal20220817_data_lerobot"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Python: use uv venv if present, else find env with torch ─────────────────
PYTHON=${PYTHON:-""}
if [ -z "$PYTHON" ]; then
    # uv creates .venv in the project root
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        for candidate in \
            /home/ubuntu/anaconda3/bin/python \
            /home/ubuntu/anaconda3/envs/base/bin/python \
            /home/ubuntu/miniconda3/bin/python \
            /home/ubuntu/miniconda3/envs/base/bin/python \
            /opt/conda/bin/python \
            /opt/miniconda3/bin/python \
            "$(which python)"; do
            if [ -x "$candidate" ] && "$candidate" -c "import torch" 2>/dev/null; then
                PYTHON="$candidate"
                break
            fi
        done
        if [ -z "$PYTHON" ]; then
            PYTHON="$(which python3)"
        fi
    fi
fi
echo "[python] Using: $PYTHON ($("$PYTHON" --version 2>&1))"

# Install Isaac-GR00T package + all deps if not already installed
if ! "$PYTHON" -c "import gr00t" 2>/dev/null; then
    echo "[deps] Installing Isaac-GR00T and dependencies via uv..."
    uv sync --extra gpu
fi

# ── Step 1: Check Eagle2.5 source is accessible ───────────────────────────────
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ ! -d "$EAGLE_REPO" ]; then
    echo "ERROR: Eagle2.5 repo not found at $REPO_ROOT/../Eagle/Eagle2_5"
    echo "       Clone it there or set PYTHONPATH to include the eaglevl package."
    exit 1
fi
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
echo "[eagle] Eagle2.5 source: $EAGLE_REPO"

# ── Step 2: Download dataset if not present ───────────────────────────────────
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

# ── Step 3: Copy modality.json if missing ─────────────────────────────────────
if [ ! -f "$MODALITY_DST" ]; then
    echo "[data] Copying modality.json..."
    mkdir -p "$(dirname "$MODALITY_DST")"
    cp "$MODALITY_SRC" "$MODALITY_DST"
else
    echo "[data] modality.json already in place."
fi

# ── Step 4: Convert AV1 videos to H.264 if needed ────────────────────────────
FIRST_VIDEO=$(find "$DATASET_PATH/videos" -name "*.mp4" 2>/dev/null | head -1)
if [ -n "$FIRST_VIDEO" ]; then
    CODEC=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name \
        -of default=nw=1:nk=1 "$FIRST_VIDEO" 2>/dev/null || echo "unknown")
    if [[ "$CODEC" == "av01" || "$CODEC" == "av1" ]]; then
        echo "[data] AV1 videos detected — converting to H.264..."
        python examples/SimplerEnv/convert_av1_to_h264.py \
            "$DATASET_PATH" --jobs "$(nproc)"
        echo "[data] Video conversion complete."
    else
        echo "[data] Videos already H.264 — skipping conversion."
    fi
fi

# ── Step 5: Create output dir + check HF repo ────────────────────────────────
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

# ── Step 6: Train ─────────────────────────────────────────────────────────────
echo "[train] Starting finetuning with Eagle2.5+Gemma3 backbone"
echo "[train] Base model: $BASE_MODEL_PATH"
echo "[train] GPUs=$NUM_GPUS  Steps=$MAX_STEPS  Embodiment=$EMBODIMENT"

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    $TRAIN_SCRIPT \
    --base_model_path "$BASE_MODEL_PATH" \
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
    --global_batch_size 512 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.5 \
    --backbone_model_type eagle2_5 \
    --backbone_embedding_dim $BACKBONE_EMBEDDING_DIM \
    --tune_llm True

echo "[done] Checkpoint saved to $OUTPUT_DIR"
