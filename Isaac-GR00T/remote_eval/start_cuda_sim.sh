#!/usr/bin/env bash
# start_cuda_sim.sh — launch policy server + control server (which manages sim_server), + ngrok
#
# Usage:
#   ./remote_eval/start_cuda_sim.sh \
#       --checkpoint output/post-train-fractal/post-train-fractal/checkpoint-1000 \
#       --env simpler_env_google/google_robot_close_drawer \
#       [--policy_port 5557] [--sim_port 8080] [--control_port 8090] [--ngrok] [--no_policy]
#
# Architecture:
#   ngrok → control_server:8090 → (proxies to) sim_server:8080
#                               → manages sim_server lifecycle (restart/status/logs)
#   policy_server:5557 ← sim_server calls this for /step_cuda and /predict
#
# Client endpoints (all via control_server/ngrok URL):
#   GET  /info                  — sim env + step count
#   POST /reset                 — reset env, returns obs
#   POST /step {action}         — client-provided action
#   POST /step_cuda {obs}       — CUDA predict + step (one round trip)
#   POST /predict {obs}         — CUDA action only (no step)
#   GET  /control/status        — is sim running? env? uptime?
#   POST /control/restart       — restart sim_server (no ngrok reconnect needed)
#   GET  /control/logs?n=100    — tail of sim_server log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLICY_PORT=5557
SIM_PORT=8080
CONTROL_PORT=8090
CHECKPOINT=""
ENV_NAME="simpler_env_google/google_robot_close_drawer"
USE_NGROK=0
NO_POLICY=0
EMBODIMENT="OXE_GOOGLE"
MAX_EPISODE_STEPS=300
N_ACTION_STEPS=8
SEED=42
VIDEO_DIR=""

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
        --env)           ENV_NAME="$2";      shift 2 ;;
        --policy_port)   POLICY_PORT="$2";   shift 2 ;;
        --sim_port)      SIM_PORT="$2";      shift 2 ;;
        --control_port)  CONTROL_PORT="$2";  shift 2 ;;
        --embodiment)    EMBODIMENT="$2";    shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --video_dir)     VIDEO_DIR="$2";     shift 2 ;;
        --ngrok)         USE_NGROK=1;        shift ;;
        --no_policy)     NO_POLICY=1;        shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" && "$NO_POLICY" -eq 0 ]]; then
    echo "ERROR: --checkpoint is required (or use --no_policy)"
    echo "Usage: $0 --checkpoint <path> [--env <name>] [--ngrok] [--no_policy]"
    exit 1
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
POLICY_SCRIPT="$REPO_ROOT/gr00t/eval/run_gr00t_server.py"
SIM_SCRIPT="$REPO_ROOT/remote_eval/sim_server.py"
CONTROL_SCRIPT="$REPO_ROOT/remote_eval/control_server.py"
LOG_POLICY="/tmp/gr00t_policy_server.log"
LOG_CONTROL="/tmp/gr00t_control_server.log"
# sim_server log managed by control_server → /tmp/gr00t_sim_server.log

# ── cleanup on exit ───────────────────────────────────────────────────────────
POLICY_PID=""
CONTROL_PID=""
NGROK_PID=""

cleanup() {
    echo ""
    echo "[start_cuda_sim] Shutting down..."
    [[ -n "$POLICY_PID"  ]] && kill "$POLICY_PID"  2>/dev/null || true
    [[ -n "$CONTROL_PID" ]] && kill "$CONTROL_PID" 2>/dev/null || true
    [[ -n "$NGROK_PID"   ]] && kill "$NGROK_PID"   2>/dev/null || true
    wait 2>/dev/null || true
    echo "[start_cuda_sim] Done."
}
trap cleanup EXIT INT TERM

# ── 1. Start policy server (if needed) ───────────────────────────────────────
if [[ "$NO_POLICY" -eq 0 ]]; then
    echo "[start_cuda_sim] Starting policy server (checkpoint: $CHECKPOINT, port: $POLICY_PORT)..."

    EAGLE_REPO="$REPO_ROOT/../Eagle/Eagle2_5"
    PYTHONPATH_EXTRA=""
    [[ -d "$EAGLE_REPO" ]] && PYTHONPATH_EXTRA="$EAGLE_REPO:"

    PYTHONPATH="${PYTHONPATH_EXTRA}${REPO_ROOT}:${PYTHONPATH:-}" \
        "$VENV_PYTHON" "$POLICY_SCRIPT" \
        --model_path "$CHECKPOINT" \
        --embodiment_tag "$EMBODIMENT" \
        --use_sim_policy_wrapper \
        --port "$POLICY_PORT" \
        --device cuda \
        > "$LOG_POLICY" 2>&1 &
    POLICY_PID=$!
    echo "[start_cuda_sim] Policy server PID=$POLICY_PID, log: $LOG_POLICY"

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
fi

# ── 2. Build sim_server command (passed to control_server for restart support) ─
SIM_CMD="$VENV_PYTHON $SIM_SCRIPT --env_name $ENV_NAME --port $SIM_PORT --seed $SEED --max_episode_steps $MAX_EPISODE_STEPS --n_action_steps $N_ACTION_STEPS"
if [[ "$NO_POLICY" -eq 1 ]]; then
    SIM_CMD="$SIM_CMD --no_policy"
else
    SIM_CMD="$SIM_CMD --policy_host localhost --policy_port $POLICY_PORT"
fi
[[ -n "$VIDEO_DIR" ]] && SIM_CMD="$SIM_CMD --video_dir $VIDEO_DIR"

# ── 3. Start control server (manages sim_server as subprocess) ────────────────
echo "[start_cuda_sim] Starting control server on port $CONTROL_PORT..."

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    "$VENV_PYTHON" "$CONTROL_SCRIPT" \
    --sim_cmd "$SIM_CMD" \
    --sim_port "$SIM_PORT" \
    --port "$CONTROL_PORT" \
    > "$LOG_CONTROL" 2>&1 &
CONTROL_PID=$!
echo "[start_cuda_sim] Control server PID=$CONTROL_PID, log: $LOG_CONTROL"

# Wait for control server HTTP port
DEADLINE=$((SECONDS + 60))
while [[ $SECONDS -lt $DEADLINE ]]; do
    if ! kill -0 "$CONTROL_PID" 2>/dev/null; then
        echo "[start_cuda_sim] ERROR: Control server died. Last lines:"
        tail -20 "$LOG_CONTROL"
        exit 1
    fi
    if ss -tlnp 2>/dev/null | grep -q ":$CONTROL_PORT "; then
        echo "[start_cuda_sim] Control server up on port $CONTROL_PORT"
        break
    fi
    sleep 2
done

# ── 4. Optional ngrok tunnel → control_server ─────────────────────────────────
NGROK_URL=""
if [[ "$USE_NGROK" -eq 1 ]]; then
    echo "[start_cuda_sim] Starting ngrok tunnel on port $CONTROL_PORT..."
    pkill -f "ngrok http" 2>/dev/null || true
    ngrok http "$CONTROL_PORT" > /tmp/ngrok_cuda_sim.log 2>&1 &
    NGROK_PID=$!
    # Poll for tunnel URL
    for i in $(seq 1 10); do
        sleep 2
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
            | python3 -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print(t[0]['public_url'])" 2>/dev/null || echo "")
        [[ -n "$NGROK_URL" ]] && break
    done
fi

# ── 5. Print summary ──────────────────────────────────────────────────────────
BASE_URL="${NGROK_URL:-http://localhost:$CONTROL_PORT}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  GemmaRobot Sim Stack"
[[ -n "$NGROK_URL" ]] && echo "  URL: $NGROK_URL" || echo "  URL: http://localhost:$CONTROL_PORT  (no ngrok)"
echo ""
echo "  Sim endpoints:"
echo "    GET  $BASE_URL/info"
echo "    POST $BASE_URL/reset"
echo "    POST $BASE_URL/step_cuda    ← CUDA predict + step"
echo "    POST $BASE_URL/step         ← client action"
echo "    POST $BASE_URL/predict      ← action only"
echo ""
echo "  Control endpoints:"
echo "    GET  $BASE_URL/control/status"
echo "    POST $BASE_URL/control/restart"
echo "    GET  $BASE_URL/control/logs"
echo ""
echo "  Logs:"
[[ "$NO_POLICY" -eq 0 ]] && echo "    Policy:  $LOG_POLICY"
echo "    Sim:     /tmp/gr00t_sim_server.log"
echo "    Control: $LOG_CONTROL"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "[start_cuda_sim] Running. Ctrl+C to stop all."

# ── 6. Wait ───────────────────────────────────────────────────────────────────
wait "$CONTROL_PID"
