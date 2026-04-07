#!/usr/bin/env bash
# eval_single.sh — run a single episode for the demo worker
# Usage: bash scripts/eval_single.sh <env_name> <output_dir>
#
# Expects a GR00T server already running on PORT (default 5556).
# Outputs: <output_dir>/result.json, <output_dir>/*.mp4

set -e

ENV_NAME=${1:?"Usage: eval_single.sh <env_name> <output_dir>"}
OUT_DIR=${2:?"Usage: eval_single.sh <env_name> <output_dir>"}
PORT=${PORT:-5556}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROLLOUT_SCRIPT="$REPO_ROOT/gr00t/eval/rollout_policy.py"
SIMPLER_PYTHON="$REPO_ROOT/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
ROLLOUT_PYTHON="${SIMPLER_PYTHON:-python3}"

mkdir -p "$OUT_DIR"

echo "[eval_single] env=$ENV_NAME out=$OUT_DIR port=$PORT"

env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    "$ROLLOUT_PYTHON" "$ROLLOUT_SCRIPT" \
    --env_name "$ENV_NAME" \
    --n_episodes 1 \
    --n_envs 1 \
    --n_action_steps 8 \
    --max_episode_steps 504 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port "$PORT" \
    --output_json "$OUT_DIR/result.json" \
    --video_dir "$OUT_DIR"

echo "[eval_single] Done. result: $(cat $OUT_DIR/result.json 2>/dev/null || echo 'no result')"
