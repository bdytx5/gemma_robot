#!/usr/bin/env bash
# eval_baseline_n16.sh
#
# Run the pretrained nvidia/GR00T-N1.6-fractal model on the same SimplerEnv
# Google robot eval tasks and log results to W&B as "baseline-gr00t".
#
# Usage:
#   bash scripts/eval_baseline_n16.sh
#
# Overrides:
#   N_EPISODES      Episodes per env (default: 5)
#   EVAL_PORT       Policy server port (default: 5557 — different from training eval)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="nvidia/GR00T-N1.6-fractal"
N_EPISODES=${N_EPISODES:-5}
EVAL_PORT=${EVAL_PORT:-5557}  # Use different port from training eval (5556)
WANDB_PROJECT="finetune-gr00t-n1d6"
WANDB_RUN_ID="baseline-gr00t"
SEED=42

PYTHON="$REPO_ROOT/.venv/bin/python"
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ -d "$EAGLE_REPO" ]; then
    export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo " Baseline eval: $MODEL_ID"
echo "============================================"
echo "[config] Episodes/env: $N_EPISODES"
echo "[config] Port: $EVAL_PORT"
echo "[config] W&B run: $WANDB_RUN_ID"
echo "[config] Seed: $SEED"
echo ""

"$PYTHON" scripts/eval_google_robot.py \
    --checkpoint_path "$MODEL_ID" \
    --step 0 \
    --n_episodes "$N_EPISODES" \
    --port "$EVAL_PORT" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_id "$WANDB_RUN_ID" \
    --results_file "eval_results_baseline.jsonl" \
    --seed "$SEED"

echo ""
echo "[done] Baseline eval complete. Results in eval_results_baseline.jsonl"
echo "[done] W&B run: $WANDB_RUN_ID"
