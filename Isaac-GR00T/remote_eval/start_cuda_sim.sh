#!/usr/bin/env bash
# start_cuda_sim.sh — launch policy server + sim server, then optionally tunnel via ngrok
#
# Usage:
#   ./remote_eval/start_cuda_sim.sh \
#       --checkpoint output/post-train-fractal/post-train-fractal/checkpoint-1000 \
#       --env simpler_env_google/google_robot_close_drawer \
#       [--policy_port 5557] [--sim_port 8080] [--ngrok]
#
# Client flow (one round trip per step):
#   POST /reset              → {obs, done, success}
#   POST /step_cuda {obs}    → {obs, reward, done, success, action}   (CUDA predicts + steps)
#   POST /step_cuda {obs}    → ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLICY_PORT=5557
SIM_PORT=8080
CHECKPOINT=""
ENV_NAME="simpler_env_google/google_robot_close_drawer"
USE_NGROK=0
EMBODIMENT="OXE_GOOGLE"
MAX_EPISODE_STEPS=300
N_ACTION_STEPS=8
SEED=42
VIDEO_DIR=""

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2";         shift 2 ;;
        --env)          ENV_NAME="$2";           shift 2 ;;
        --policy_port)  POLICY_PORT="$2";        shift 2 ;;
        --sim_port)     SIM_PORT="$2";           shift 2 ;;
        --embodiment)   EMBODIMENT="$2";         shift 2 ;;
        --seed)         SEED="$2";               shift 2 ;;
        --video_dir)    VIDEO_DIR="$2";          shift 2 ;;
        --ngrok)        USE_NGROK=1;             shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required"
    echo "Usage: $0 --checkpoint <path> [--env <name>] [--policy_port 5557] [--sim_port 8080] [--ngrok]"
    exit 1
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
POLICY_SCRIPT="$REPO_ROOT/gr00t/eval/run_gr00t_server.py"
SIM_SCRIPT="$REPO_ROOT/remote_eval/sim_server.py"
LOG_POLICY="/tmp/gr00t_policy_server.log"
LOG_SIM="/tmp/gr00t_sim_server.log"

# ── cleanup on exit ───────────────────────────────────────────────────────────
POLICY_PID=""
SIM_PID=""
NGROK_PID=""

cleanup() {
    echo ""
    echo "[start_cuda_sim] Shutting down..."
    [[ -n "$POLICY_PID" ]] && kill "$POLICY_PID" 2>/dev/null || true
    [[ -n "$SIM_PID"    ]] && kill "$SIM_PID"    2>/dev/null || true
    [[ -n "$NGROK_PID"  ]] && kill "$NGROK_PID"  2>/dev/null || true
    wait 2>/dev/null || true
    echo "[start_cuda_sim] Done."
}
trap cleanup EXIT INT TERM

# ── 1. Start policy server ────────────────────────────────────────────────────
echo "[start_cuda_sim] Starting policy server (checkpoint: $CHECKPOINT, port: $POLICY_PORT)..."

EAGLE_REPO="$REPO_ROOT/../Eagle/Eagle2_5"
PYTHONPATH_EXTRA=""
if [[ -d "$EAGLE_REPO" ]]; then
    PYTHONPATH_EXTRA="$EAGLE_REPO:"
fi

PYTHONPATH="${PYTHONPATH_EXTRA}${PYTHONPATH:-}" \
PYTHONPATH="$REPO_ROOT:${PYTHONPATH}" \
    "$VENV_PYTHON" "$POLICY_SCRIPT" \
    --model_path "$CHECKPOINT" \
    --embodiment_tag "$EMBODIMENT" \
    --use_sim_policy_wrapper \
    --port "$POLICY_PORT" \
    --device cuda \
    > "$LOG_POLICY" 2>&1 &
POLICY_PID=$!
echo "[start_cuda_sim] Policy server PID=$POLICY_PID, log: $LOG_POLICY"

# ── 2. Wait for policy server ZMQ port ───────────────────────────────────────
echo "[start_cuda_sim] Waiting for policy server on port $POLICY_PORT..."
DEADLINE=$((SECONDS + 180))
while [[ $SECONDS -lt $DEADLINE ]]; do
    if ! kill -0 "$POLICY_PID" 2>/dev/null; then
        echo "[start_cuda_sim] ERROR: Policy server died. Last lines:"
        tail -20 "$LOG_POLICY"
        exit 1
    fi
    if ss -tlnp 2>/dev/null | grep -q ":$POLICY_PORT "; then
        echo "[start_cuda_sim] Policy server up on port $POLICY_PORT"
        break
    fi
    sleep 3
done
if [[ $SECONDS -ge $DEADLINE ]]; then
    echo "[start_cuda_sim] ERROR: Policy server did not come up in 180s"
    tail -20 "$LOG_POLICY"
    exit 1
fi

# ── 3. Start sim server ───────────────────────────────────────────────────────
echo "[start_cuda_sim] Starting sim server (env: $ENV_NAME, port: $SIM_PORT)..."

SIM_EXTRA_ARGS=""
[[ -n "$VIDEO_DIR" ]] && SIM_EXTRA_ARGS="--video_dir $VIDEO_DIR"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    "$VENV_PYTHON" "$SIM_SCRIPT" \
    --env_name "$ENV_NAME" \
    --port "$SIM_PORT" \
    --seed "$SEED" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --n_action_steps "$N_ACTION_STEPS" \
    --policy_host localhost \
    --policy_port "$POLICY_PORT" \
    $SIM_EXTRA_ARGS \
    > "$LOG_SIM" 2>&1 &
SIM_PID=$!
echo "[start_cuda_sim] Sim server PID=$SIM_PID, log: $LOG_SIM"

# ── 4. Wait for sim server HTTP port ─────────────────────────────────────────
echo "[start_cuda_sim] Waiting for sim server on port $SIM_PORT..."
DEADLINE=$((SECONDS + 60))
while [[ $SECONDS -lt $DEADLINE ]]; do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo "[start_cuda_sim] ERROR: Sim server died. Last lines:"
        tail -20 "$LOG_SIM"
        exit 1
    fi
    if ss -tlnp 2>/dev/null | grep -q ":$SIM_PORT "; then
        echo "[start_cuda_sim] Sim server up on port $SIM_PORT"
        break
    fi
    sleep 2
done

# ── 5. Optional ngrok tunnel ──────────────────────────────────────────────────
if [[ "$USE_NGROK" -eq 1 ]]; then
    echo "[start_cuda_sim] Starting ngrok tunnel on port $SIM_PORT..."
    ngrok http "$SIM_PORT" > /tmp/ngrok_cuda_sim.log 2>&1 &
    NGROK_PID=$!
    sleep 4
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print(t[0]['public_url'])" 2>/dev/null || echo "")
    if [[ -n "$NGROK_URL" ]]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "  Sim server URL: $NGROK_URL"
        echo "  Endpoints:"
        echo "    POST $NGROK_URL/reset"
        echo "    POST $NGROK_URL/step_cuda   ← CUDA predict + step"
        echo "    POST $NGROK_URL/step        ← client-provided action"
        echo "    POST $NGROK_URL/predict     ← action only (no step)"
        echo "    GET  $NGROK_URL/info"
        echo "╚══════════════════════════════════════════════════════╝"
    else
        echo "[start_cuda_sim] ngrok tunnel started (check http://localhost:4040)"
    fi
fi

echo ""
echo "[start_cuda_sim] Both servers running. Ctrl+C to stop."
echo "  Policy log: $LOG_POLICY"
echo "  Sim log:    $LOG_SIM"
echo ""
echo "  Local sim server: http://localhost:$SIM_PORT"
echo "  Endpoints:"
echo "    POST /reset"
echo "    POST /step_cuda   ← CUDA predict + step (one round trip)"
echo "    POST /step        ← client-provided action"
echo "    POST /predict     ← action only (no step)"
echo "    GET  /info"

# ── 6. Wait forever (cleanup on Ctrl+C) ──────────────────────────────────────
wait "$POLICY_PID" "$SIM_PID"
