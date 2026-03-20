#!/usr/bin/env bash
# Watch output dir for new checkpoints and run eval on each one.
#
# Usage:
#   bash scripts/eval_watcher.sh [OUTPUT_DIR]
#
# Overrides:
#   OUTPUT_DIR      Dir to watch (default: ./output/gr00t-eagle2_5-fractal)
#   N_EPISODES      Episodes per env (default: 5)
#   WANDB_PROJECT   W&B project (default: finetune-gr00t-n1d6)
#   POLL_INTERVAL   Seconds between scans (default: 60)
#   EVAL_PORT       Policy server port (default: 5556)
#   RESULTS_FILE    JSONL results log (default: eval_results.jsonl)

set -e

OUTPUT_DIR=${1:-${OUTPUT_DIR:-"./output/gr00t-eagle2_5-fractal"}}
N_EPISODES=${N_EPISODES:-5}
WANDB_PROJECT=${WANDB_PROJECT:-"finetune-gr00t-n1d6"}
POLL_INTERVAL=${POLL_INTERVAL:-60}
EVAL_PORT=${EVAL_PORT:-5556}
RESULTS_FILE=${RESULTS_FILE:-"eval_results.jsonl"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Stable W&B run ID derived from the output dir — all checkpoints log to one run
WANDB_RUN_ID="eval-$(basename "$OUTPUT_DIR")"

PYTHON="$REPO_ROOT/.venv/bin/python"
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ -d "$EAGLE_REPO" ]; then
    export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[watcher] Watching: $OUTPUT_DIR"
echo "[watcher] Episodes/env: $N_EPISODES  |  Poll: ${POLL_INTERVAL}s  |  Port: $EVAL_PORT"
echo "[watcher] W&B run ID: $WANDB_RUN_ID (all checkpoints → one run)"
echo "[watcher] Results: $RESULTS_FILE"

EVALUATED=""  # space-separated list of already-evaluated checkpoint paths

while true; do
    # Find all checkpoint-N dirs, sorted by step number
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "[watcher] Output dir not found yet, waiting..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    CHECKPOINTS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" \
        | sort -t'-' -k2 -n)

    NEW=""
    for CKPT in $CHECKPOINTS; do
        if echo "$EVALUATED" | grep -qw "$CKPT"; then
            continue
        fi
        # Make sure the checkpoint has a config.json (i.e. fully written)
        if [ ! -f "$CKPT/config.json" ]; then
            echo "[watcher] $CKPT incomplete (no config.json), skipping for now"
            continue
        fi
        NEW="$NEW $CKPT"
    done

    if [ -z "$NEW" ]; then
        echo "[watcher] No new checkpoints. Next scan in ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    for CKPT in $NEW; do
        STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
        echo ""
        echo "[watcher] =========================================="
        echo "[watcher] New checkpoint: $CKPT  (step $STEP)"
        echo "[watcher] =========================================="

        "$PYTHON" scripts/eval_google_robot.py \
            --checkpoint_path "$CKPT" \
            --step "$STEP" \
            --n_episodes "$N_EPISODES" \
            --port "$EVAL_PORT" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_id "$WANDB_RUN_ID" \
            --results_file "$RESULTS_FILE" \
            || echo "[watcher] Eval failed for $CKPT — continuing."

        EVALUATED="$EVALUATED $CKPT"
        echo "[watcher] Done with $CKPT"
    done

    echo "[watcher] Next scan in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done
