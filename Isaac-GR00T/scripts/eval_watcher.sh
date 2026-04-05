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
WANDB_RUN_ID="270m-balanced-eval"

PYTHON="$REPO_ROOT/.venv/bin/python"
SIMPLER_VENV="$REPO_ROOT/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv"
SIMPLER_REPO="$REPO_ROOT/external_dependencies/SimplerEnv"
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ -d "$EAGLE_REPO" ]; then
    export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Virtual display for SAPIEN/Vulkan rendering on headless servers
# ---------------------------------------------------------------------------
if [ -z "${DISPLAY:-}" ] && command -v Xvfb &>/dev/null; then
    echo "[setup] Starting Xvfb virtual display..."
    Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
    XVFB_PID=$!
    export DISPLAY=:99
    echo "[setup] DISPLAY=$DISPLAY (Xvfb PID $XVFB_PID)"
fi

# ---------------------------------------------------------------------------
# Dependency check: ensure SimplerEnv venv exists and simpler_env is importable
# ---------------------------------------------------------------------------
setup_simpler_env() {
    echo "[setup] SimplerEnv venv not ready — setting up now..."

    # Clone SimplerEnv if missing
    if [ ! -f "$SIMPLER_REPO/setup.py" ]; then
        echo "[setup] Cloning SimplerEnv..."
        mkdir -p "$REPO_ROOT/external_dependencies"
        git clone https://github.com/squarefk/SimplerEnv.git "$SIMPLER_REPO"
    fi

    # Clone ManiSkill2_real2sim submodule if missing
    if [ ! -f "$SIMPLER_REPO/ManiSkill2_real2sim/setup.py" ]; then
        echo "[setup] Cloning ManiSkill2_real2sim..."
        rm -rf "$SIMPLER_REPO/ManiSkill2_real2sim"
        git clone https://github.com/youliangtan/ManiSkill2_real2sim \
            "$SIMPLER_REPO/ManiSkill2_real2sim"
    fi

    # Create venv
    echo "[setup] Creating venv..."
    rm -rf "$SIMPLER_VENV"
    uv venv "$SIMPLER_VENV" --python 3.10

    # Install all deps
    echo "[setup] Installing dependencies..."
    uv pip install --python "$SIMPLER_VENV" \
        gymnasium==0.29.1 \
        json-numpy>=2.1.1 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        ray==2.48.0

    uv pip install --python "$SIMPLER_VENV" -e "$SIMPLER_REPO/ManiSkill2_real2sim"
    uv pip install --python "$SIMPLER_VENV" -e "$SIMPLER_REPO"
    uv pip install --python "$SIMPLER_VENV" --editable "$REPO_ROOT" --no-deps
    uv pip install --python "$SIMPLER_VENV" \
        tianshou==0.5.1 pydantic av zmq torchvision==0.22.0 transformers==4.51.3

    # Fix pkg_resources (sapien needs setuptools<=69)
    uv pip install --python "$SIMPLER_VENV" --reinstall setuptools==69.5.1

    echo "[setup] SimplerEnv ready."
}

echo "[setup] Checking SimplerEnv venv..."
if ! "$SIMPLER_VENV/bin/python" -c "import simpler_env" 2>/dev/null; then
    setup_simpler_env
    # Verify
    if ! "$SIMPLER_VENV/bin/python" -c "import simpler_env" 2>/dev/null; then
        echo "[setup] ERROR: SimplerEnv setup failed — aborting." >&2
        exit 1
    fi
fi
echo "[setup] SimplerEnv OK."
# ---------------------------------------------------------------------------

# Cleanup trap: kill any orphaned eval processes on exit
cleanup_eval_processes() {
    echo "[watcher] Cleaning up eval processes..."
    pkill -f "run_gr00t_server.py.*--port $EVAL_PORT" 2>/dev/null || true
    pkill -f "rollout_policy.py.*--policy_client_port $EVAL_PORT" 2>/dev/null || true
    sleep 1
    pkill -9 -f "run_gr00t_server.py.*--port $EVAL_PORT" 2>/dev/null || true
    pkill -9 -f "rollout_policy.py.*--policy_client_port $EVAL_PORT" 2>/dev/null || true
    echo "[watcher] Cleanup done."
}
trap cleanup_eval_processes EXIT

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

        MAX_RETRIES=10
        ATTEMPT=0
        SUCCESS=0
        while [ $ATTEMPT -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; do
            ATTEMPT=$((ATTEMPT + 1))
            echo "[watcher] Attempt $ATTEMPT/$MAX_RETRIES for $CKPT"

            "$PYTHON" scripts/eval_google_robot.py \
                --checkpoint_path "$CKPT" \
                --step "$STEP" \
                --n_episodes "$N_EPISODES" \
                --port "$EVAL_PORT" \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_run_id "$WANDB_RUN_ID" \
                --results_file "$RESULTS_FILE" \
                --device cuda \
                && SUCCESS=1

            # Always clean up between attempts
            cleanup_eval_processes

            if [ $SUCCESS -eq 0 ]; then
                echo "[watcher] Attempt $ATTEMPT failed for $CKPT — retrying in 30s..."
                sleep 30
            fi
        done

        if [ $SUCCESS -eq 0 ]; then
            echo "[watcher] All $MAX_RETRIES attempts failed for $CKPT — skipping."
        fi

        EVALUATED="$EVALUATED $CKPT"
        echo "[watcher] Done with $CKPT"
    done

    echo "[watcher] Next scan in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done
