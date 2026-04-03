#!/usr/bin/env bash
# train_balanced_from_270m.sh
#
# Fine-tune on the balanced 8-dataset mixture starting from the latest
# 270m pretrained checkpoint. After training completes, runs SimplerEnv
# eval on each checkpoint (single GPU — train then eval sequentially).
#
# Prerequisites:
#   - All 8 datasets downloaded via scripts/download_all_datasets.sh
#   - Eagle2.5 repo at ../Eagle/Eagle2_5
#   - 270m fractal checkpoint in output/gr00t-eagle2_5-fractal/
#
# Usage:
#   bash scripts/train_balanced_from_270m.sh
#
# Overrides:
#   CHECKPOINT_270M   Explicit checkpoint path (default: auto-detect latest)
#   OUTPUT_DIR        Where to save (default: ./output/balanced-from-270m)
#   MAX_STEPS         Training steps (default: 20000)
#   SKIP_EVAL         Set to 1 to skip post-training eval (default: 0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Config ────────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/balanced-from-270m"}
MAX_STEPS=${MAX_STEPS:-20000}

# ── Auto-detect latest 270m checkpoint ────────────────────────────────────────
PREV_OUTPUT="./output/gr00t-eagle2_5-fractal"
if [ -z "${CHECKPOINT_270M:-}" ]; then
    CHECKPOINT_270M=$(find "$PREV_OUTPUT" -maxdepth 1 -type d -name "checkpoint-*" \
        | sort -t'-' -k2 -n | tail -1)
    if [ -z "$CHECKPOINT_270M" ]; then
        echo "ERROR: No checkpoint found in $PREV_OUTPUT"
        echo "       Set CHECKPOINT_270M explicitly."
        exit 1
    fi
fi
echo "[config] Starting from checkpoint: $CHECKPOINT_270M"

# ── Deps ──────────────────────────────────────────────────────────────────────
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    echo "[deps] Running uv sync --extra gpu..."
    uv sync --extra gpu
fi
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
echo "[python] Using: $PYTHON ($("$PYTHON" --version 2>&1))"

EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ ! -d "$EAGLE_REPO" ]; then
    echo "ERROR: Eagle2.5 repo not found at $REPO_ROOT/../Eagle/Eagle2_5"
    exit 1
fi
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[eagle] Eagle2.5 source: $EAGLE_REPO"

# ── Verify datasets exist ────────────────────────────────────────────────────
echo "[data] Checking datasets..."
READY=0
MISSING=0
for ds in \
    "examples/SimplerEnv/fractal20220817_data_lerobot" \
    "examples/SimplerEnv/bridge_orig_lerobot" \
    "examples/LIBERO/libero_10_no_noops_1.0.0_lerobot" \
    "examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot" \
    "examples/LIBERO/libero_object_no_noops_1.0.0_lerobot" \
    "examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot" \
    "examples/DROID/droid_subset" \
    "examples/robocasa/robocasa_mg_gr00t_300"; do
    if [ ! -f "$REPO_ROOT/$ds/meta/info.json" ]; then
        echo "  MISSING: $ds"
        MISSING=$((MISSING + 1))
    else
        echo "  OK: $(basename $ds)"
        READY=$((READY + 1))
    fi
done
# Check NVIDIA single_panda_gripper tasks (optional — training skips missing ones)
for task in \
    single_panda_gripper-OpenDrawer single_panda_gripper-OpenDoubleDoor \
    single_panda_gripper-OpenSingleDoor single_panda_gripper-CloseDrawer \
    single_panda_gripper-PnPCabToCounter single_panda_gripper-PnPCounterToCab \
    single_panda_gripper-PnPCounterToMicrowave single_panda_gripper-PnPMicrowaveToCounter; do
    ds="examples/nvidia-panda/$task"
    if [ -f "$REPO_ROOT/$ds/meta/info.json" ]; then
        echo "  OK: $task"
        READY=$((READY + 1))
    else
        echo "  NOT YET: $task (will be skipped in training)"
    fi
done
echo "[data] $READY datasets ready."
if [ "$READY" -lt 8 ]; then
    echo "ERROR: Need at least the 8 base datasets. Run scripts/download_all_datasets.sh first."
    exit 1
fi

# ── Output dir + HF repo ─────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
export HF_REPO_ID=${HF_REPO_ID:-"youngbrett48/$SCRIPT_NAME"}

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
fi

# ── Train on balanced 8-dataset mixture ───────────────────────────────────────
echo ""
echo "============================================"
echo " Balanced 8-dataset training from 270m ckpt"
echo "============================================"
echo "[train] Checkpoint: $CHECKPOINT_270M"
echo "[train] GPUs=$NUM_GPUS  Steps=$MAX_STEPS"
echo "[train] Output: $OUTPUT_DIR"
echo ""

EXTRA_ARGS=""
if [ -n "${PRETRAINED_ACTION_HEAD:-}" ]; then
    echo "[train] Loading pretrained action head from: $PRETRAINED_ACTION_HEAD"
    EXTRA_ARGS="--load_pretrained_action_head $PRETRAINED_ACTION_HEAD"
fi

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=${MASTER_PORT:-29501} \
    scripts/finetune_balanced_270m.py \
    --checkpoint_path "$CHECKPOINT_270M" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $MAX_STEPS \
    --wandb_project "finetune-gr00t-n1d6" \
    --experiment_name "balanced-270m-train" \
    $EXTRA_ARGS

echo "[done] Training complete. Checkpoints in $OUTPUT_DIR"

# ── Post-training eval (SimplerEnv on same GPU) ─────────────────────────────
# Actual checkpoint dir is output_dir/experiment_name when experiment_name is set
CKPT_DIR="$OUTPUT_DIR/balanced-270m-train"
if [ ! -d "$CKPT_DIR" ]; then
    CKPT_DIR="$OUTPUT_DIR"
fi

if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo ""
    echo "[eval] Running SimplerEnv eval on all checkpoints in $CKPT_DIR..."
    bash scripts/eval_watcher.sh "$CKPT_DIR" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
    echo "[eval] Eval skipped (SKIP_EVAL=1). Run manually:"
    echo "       bash scripts/eval_watcher.sh $CKPT_DIR"
fi
