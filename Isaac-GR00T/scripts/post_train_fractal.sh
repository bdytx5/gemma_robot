#!/usr/bin/env bash
# post_train_fractal.sh
#
# Post-training: fine-tune on fractal dataset only with last 4 LLM layers
# unfrozen, starting from the latest balanced-270m checkpoint.
#
# Usage:
#   bash scripts/post_train_fractal.sh
#   bash scripts/post_train_fractal.sh --checkpoint /path/to/checkpoint
#   bash scripts/post_train_fractal.sh --test   # smoke test (50 steps)
#
# Overrides:
#   CHECKPOINT        Explicit checkpoint path (default: latest from balanced training)
#   OUTPUT_DIR        Where to save (default: ./output/post-train-fractal)
#   MAX_STEPS         Training steps (default: 5000)
#   LR                Learning rate (default: 5e-5)
#   TOP_LAYERS        Number of LLM layers to unfreeze (default: 4)
#   SAVE_STEPS        Save interval (default: 500)
#   WANDB_PROJECT     W&B project (default: finetune-gr00t-n1d6)
#   EXPERIMENT_NAME   W&B run name (default: post-train-fractal)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR=${OUTPUT_DIR:-"./output/post-train-fractal"}
MAX_STEPS=${MAX_STEPS:-5000}
LR=${LR:-5e-5}
TOP_LAYERS=${TOP_LAYERS:-4}
SAVE_STEPS=${SAVE_STEPS:-1000}
WANDB_PROJECT=${WANDB_PROJECT:-"finetune-gr00t-n1d6"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"post-train-fractal"}
NUM_GPUS=${NUM_GPUS:-1}

# Parse flags
EXTRA_FLAGS=""
CHECKPOINT_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT_ARG="$2"; shift ;;
        --test)       EXTRA_FLAGS="$EXTRA_FLAGS --test" ;;
        --resume)     EXTRA_FLAGS="$EXTRA_FLAGS --resume" ;;
    esac
    shift
done

# ── Auto-detect latest balanced checkpoint ────────────────────────────────────
if [ -z "$CHECKPOINT_ARG" ] && [ -z "${CHECKPOINT:-}" ]; then
    BALANCED_DIR="./output/balanced-from-270m/balanced-270m-train"
    CHECKPOINT=$(find "$BALANCED_DIR" -maxdepth 1 -type d -name "checkpoint-*" \
        | sort -t'-' -k2 -n | tail -1 2>/dev/null || true)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found in $BALANCED_DIR"
        echo "       Set CHECKPOINT or pass --checkpoint /path/to/checkpoint"
        exit 1
    fi
elif [ -n "$CHECKPOINT_ARG" ]; then
    CHECKPOINT="$CHECKPOINT_ARG"
fi
echo "[config] Checkpoint: $CHECKPOINT"
echo "[config] Output:     $OUTPUT_DIR"
echo "[config] Steps:      $MAX_STEPS"
echo "[config] LR:         $LR"
echo "[config] Top layers: $TOP_LAYERS"

# ── Deps ──────────────────────────────────────────────────────────────────────
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
echo "[python] Using: $PYTHON ($("$PYTHON" --version 2>&1))"

EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ ! -d "$EAGLE_REPO" ]; then
    echo "ERROR: Eagle2.5 repo not found at $REPO_ROOT/../Eagle/Eagle2_5"
    exit 1
fi
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Verify fractal dataset ───────────────────────────────────────────────────
FRACTAL="examples/SimplerEnv/fractal20220817_data_lerobot"
if [ ! -f "$REPO_ROOT/$FRACTAL/meta/info.json" ]; then
    echo "ERROR: Fractal dataset not found at $FRACTAL"
    echo "       Run: bash scripts/download_test_dataset.sh"
    exit 1
fi
echo "[data] Fractal dataset OK."

# ── Output dir + HF repo ─────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
export HF_REPO_ID=${HF_REPO_ID:-"youngbrett48/gr00t-post-train-fractal-270m"}

if [ -n "$HF_REPO_ID" ]; then
    echo "[hub] Checking HuggingFace repo: $HF_REPO_ID"
    python3 - <<EOF
from huggingface_hub import HfApi
import sys
api = HfApi()
try:
    api.repo_info(repo_id="$HF_REPO_ID", repo_type="model")
    print(f"Repo '$HF_REPO_ID' exists.")
except Exception:
    print(f"Creating repo '$HF_REPO_ID'...")
    try:
        api.create_repo(repo_id="$HF_REPO_ID", repo_type="model", private=True)
        print(f"Created.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
EOF
    if [ $? -ne 0 ]; then
        echo "Aborting: HuggingFace repo check failed."
        exit 1
    fi
fi

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Post-training: fractal only, top $TOP_LAYERS LLM layers"
echo "============================================"

# Start eval watcher in parallel with training
CKPT_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"
mkdir -p "$CKPT_DIR"
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo "[eval] Starting eval watcher in background (every ${EVAL_EVERY:-1000} steps)..."
    EVAL_EVERY=${EVAL_EVERY:-1000} WANDB_PROJECT="$WANDB_PROJECT" WANDB_RUN_ID="post-train-fractal-eval" \
        bash scripts/eval_watcher.sh "$CKPT_DIR" > "${OUTPUT_DIR}/eval.log" 2>&1 &
    EVAL_PID=$!
    echo "[eval] Eval watcher PID: $EVAL_PID"
fi

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=${MASTER_PORT:-29502} \
    scripts/finetune_balanced_270m.py \
    --checkpoint_path "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --save_steps $SAVE_STEPS \
    --tune_top_llm_layers $TOP_LAYERS \
    --no_tune_llm \
    --wandb_project "$WANDB_PROJECT" \
    --experiment_name "$EXPERIMENT_NAME" \
    --fractal_only \
    $EXTRA_FLAGS

echo "[done] Post-training complete. Checkpoints in $OUTPUT_DIR"

# Wait for eval watcher to finish remaining evals
if [ -n "${EVAL_PID:-}" ]; then
    echo "[eval] Waiting for eval watcher to finish..."
    wait $EVAL_PID 2>/dev/null || true
fi
