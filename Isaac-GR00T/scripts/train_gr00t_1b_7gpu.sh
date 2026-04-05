#!/usr/bin/env bash
# train_gr00t_1b_7gpu.sh
#
# Finetune GR00T-1B (Eagle2.5 + Gemma3-1B fractal backbone) on the Fractal
# dataset using 7 GPUs for training and GPU 7 for the eval server.
# Only the last 4 LLM transformer layers are unfrozen (tune_top_llm_layers=4).
#
# Usage:
#   bash scripts/train_gr00t_1b_7gpu.sh [--good]
#
#   --good  Skip dataset download (use whatever is already on disk)
#
# Overrides:
#   OUTPUT_DIR      Checkpoint dir    (default: ./output/gr00t-1b-fractal-7gpu)
#   BASE_MODEL_PATH 1B checkpoint     (default: youngbrett48/train_stage2_gemma3_1b.sh)
#   MAX_STEPS       Training steps    (default: 20000)
#   HF_REPO_ID      If set, push checkpoints here

set -e

# Ensure uv is on PATH (installed to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

GOOD_TO_GO=0
for arg in "$@"; do
    [ "$arg" = "--good" ] && GOOD_TO_GO=1
done

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_GPUS="0,1,2,3,4,5,6"   # 7 GPUs for training
NUM_GPUS=7
EVAL_GPU=7                     # 8th GPU for eval server

OUTPUT_DIR=${OUTPUT_DIR:-"./output/gr00t-1b-fractal-7gpu"}
MAX_STEPS=${MAX_STEPS:-20000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"youngbrett48/train_stage2_gemma3_1b.sh"}
EMBODIMENT="OXE_GOOGLE"
BACKBONE_PROJ_DIM=1152         # Gemma3-1B hidden_size (backbone output)
BACKBONE_EMBEDDING_DIM=2048    # Match pretrained action head (nvidia/GR00T-N1.6-fractal)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET_PATH="examples/SimplerEnv/fractal20220817_data_lerobot"
MODALITY_SRC="examples/SimplerEnv/fractal_modality.json"
MODALITY_DST="${DATASET_PATH}/meta/modality.json"

SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
export HF_REPO_ID=${HF_REPO_ID:-"youngbrett48/$SCRIPT_NAME"}

# ── Step 1: Use system Python (pyenv) ─────────────────────────────────────────
PYTHON=$(which python3)
echo "[deps] Using system python (no venv)."
echo "[python] Using: $PYTHON ($("$PYTHON" --version 2>&1))"

# ── Step 2: Eagle2.5 source ───────────────────────────────────────────────────
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ ! -d "$EAGLE_REPO" ]; then
    echo "ERROR: Eagle2.5 repo not found at $REPO_ROOT/../Eagle/Eagle2_5"
    exit 1
fi
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[eagle] Eagle2.5 source: $EAGLE_REPO"

# ── Step 3: Download small test sample of fractal dataset ─────────────────────
if [ "$GOOD_TO_GO" = "1" ]; then
    echo "[data] --good flag set — skipping download."
elif [ -f "$DATASET_PATH/meta/info.json" ]; then
    echo "[data] Dataset already present at $DATASET_PATH — skipping download."
else
    echo "[data] Downloading small fractal sample (chunk-000 only)..."
    HF_HUB_ENABLE_HF_TRANSFER=0 bash "$REPO_ROOT/scripts/download_all_datasets.sh" --test
fi

# Ensure modality.json is in place
if [ ! -f "$MODALITY_DST" ] && [ -f "$MODALITY_SRC" ]; then
    echo "[data] Copying modality.json..."
    mkdir -p "$(dirname "$MODALITY_DST")"
    cp "$MODALITY_SRC" "$MODALITY_DST"
fi

# ── Step 4: Convert AV1 → H.264 if needed ────────────────────────────────────
if [ "$GOOD_TO_GO" != "1" ]; then
    FIRST_VIDEO=$(find "$DATASET_PATH/videos" -name "*.mp4" 2>/dev/null | head -1)
    if [ -n "$FIRST_VIDEO" ]; then
        CODEC=$(ffprobe -v error -select_streams v:0 \
            -show_entries stream=codec_name \
            -of default=nw=1:nk=1 "$FIRST_VIDEO" 2>/dev/null || echo "unknown")
        if [[ "$CODEC" == "av01" || "$CODEC" == "av1" ]]; then
            echo "[data] AV1 detected — converting to H.264..."
            "$PYTHON" examples/SimplerEnv/convert_av1_to_h264.py \
                "$DATASET_PATH" --jobs "$(nproc)"
        else
            echo "[data] Videos already H.264."
        fi
    fi
fi

# ── Step 5: Output dir + HF repo check ───────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

if [ -n "$HF_REPO_ID" ]; then
    echo "[hub] Checking HuggingFace repo: $HF_REPO_ID"
    "$PYTHON" - <<EOF
from huggingface_hub import HfApi
import sys
api = HfApi()
try:
    api.repo_info(repo_id="$HF_REPO_ID", repo_type="model")
    print(f"Repo '$HF_REPO_ID' exists.")
except Exception:
    try:
        api.create_repo(repo_id="$HF_REPO_ID", repo_type="model", private=True)
        print(f"Created repo '$HF_REPO_ID'.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
EOF
    TRAIN_SCRIPT="scripts/finetune_with_hub_push.py"
else
    TRAIN_SCRIPT="gr00t/experiment/launch_finetune.py"
fi

# ── Step 6: Start eval server on GPU 7 in background ─────────────────────────
echo "[eval] Starting eval watcher on GPU $EVAL_GPU..."
CUDA_VISIBLE_DEVICES=$EVAL_GPU \
    bash scripts/eval_watcher.sh "$OUTPUT_DIR" \
    > "${OUTPUT_DIR}/eval_watcher.log" 2>&1 &
EVAL_PID=$!
echo "[eval] Watcher PID: $EVAL_PID  (log: ${OUTPUT_DIR}/eval_watcher.log)"

# ── Step 7: Train on GPUs 0-6 ────────────────────────────────────────────────
echo "[train] Starting 1B training — last 4 LLM layers unfrozen"
echo "[train] Base model : $BASE_MODEL_PATH"
echo "[train] GPUs       : $TRAIN_GPUS (NUM_GPUS=$NUM_GPUS)"
echo "[train] Steps      : $MAX_STEPS"
echo "[train] Embodiment : $EMBODIMENT"

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=${MASTER_PORT:-29503} \
    $TRAIN_SCRIPT \
    --base_model_path "$BASE_MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag $EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 200 \
    --save_total_limit 5 \
    --max_steps $MAX_STEPS \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 28 \
    --gradient_accumulation_steps 16 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.5 \
    --backbone_model_type eagle2_5 \
    --backbone_embedding_dim $BACKBONE_EMBEDDING_DIM \
    --backbone_proj_dim $BACKBONE_PROJ_DIM \
    --tune_top_llm_layers 4 \
    --load_pretrained_action_head nvidia/GR00T-N1.6-fractal

echo "[done] Checkpoint saved to $OUTPUT_DIR"

# Kill eval watcher when training finishes
if [ -n "$EVAL_PID" ] && kill -0 $EVAL_PID 2>/dev/null; then
    echo "[eval] Stopping eval watcher (PID $EVAL_PID)..."
    kill $EVAL_PID 2>/dev/null
fi
