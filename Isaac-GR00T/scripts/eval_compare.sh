#!/usr/bin/env bash
# eval_compare.sh — Head-to-head: our model vs stock N1.6
# Runs all 6 envs in parallel against each model server.
#
# Usage:
#   bash scripts/eval_compare.sh
#   bash scripts/eval_compare.sh --ours /path/to/checkpoint
#   N_EPISODES=10 bash scripts/eval_compare.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

N_EPISODES=${N_EPISODES:-10}
PORT=${PORT:-5556}
RESULTS_FILE=${RESULTS_FILE:-"eval_results_compare.jsonl"}
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
SIMPLER_PYTHON="$REPO_ROOT/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
ROLLOUT_SCRIPT="$REPO_ROOT/gr00t/eval/rollout_policy.py"
SERVER_SCRIPT="$REPO_ROOT/gr00t/eval/run_gr00t_server.py"
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
BASE_SEED=${BASE_SEED:-42}

ENVS=(
    "simpler_env_google/google_robot_open_drawer"
    "simpler_env_google/google_robot_close_drawer"
    "simpler_env_google/google_robot_place_in_closed_drawer"
    "simpler_env_google/google_robot_pick_coke_can"
    "simpler_env_google/google_robot_pick_object"
    "simpler_env_google/google_robot_move_near"
)

ROLLOUT_PYTHON="$SIMPLER_PYTHON"
if [ ! -f "$ROLLOUT_PYTHON" ]; then
    ROLLOUT_PYTHON="$PYTHON"
fi

# Parse flags
OURS_CKPT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ours) OURS_CKPT="$2"; shift ;;
    esac
    shift
done

if [ -z "$OURS_CKPT" ]; then
    OURS_CKPT=$(find ./output/post-train-fractal/post-train-fractal -maxdepth 1 -type d -name "checkpoint-*" \
        | sort -t'-' -k2 -n | tail -1 2>/dev/null || true)
    if [ -z "$OURS_CKPT" ]; then
        echo "ERROR: No post-train checkpoint found. Pass --ours /path/to/checkpoint"
        exit 1
    fi
fi

STEP=$(basename "$OURS_CKPT" | sed 's/checkpoint-//')
STOCK="nvidia/GR00T-N1.6-fractal"
VIDEO_ROOT="/tmp/gr00t_eval_compare"
rm -rf "$VIDEO_ROOT"
rm -f "$RESULTS_FILE"

start_server() {
    local model_path=$1 port=$2
    $PYTHON $SERVER_SCRIPT \
        --model_path "$model_path" --embodiment_tag OXE_GOOGLE \
        --use_sim_policy_wrapper --port "$port" --device cuda > /tmp/gr00t_server_${port}.log 2>&1 &
    SERVER_PID=$!
    echo "[server] Started PID $SERVER_PID on port $port, waiting..."
    for i in $(seq 1 180); do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',$port)); s.close()" 2>/dev/null; then
            echo "[server] Port $port ready."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Server on port $port failed to start"
    return 1
}

kill_server() {
    local pid=$1
    kill $pid 2>/dev/null; sleep 1; kill -9 $pid 2>/dev/null || true
}

run_all_envs() {
    local model_name=$1 port=$2 step=$3 video_dir=$4
    local pids=()

    for env_idx in "${!ENVS[@]}"; do
        local env="${ENVS[$env_idx]}"
        local env_short=$(basename "$env")
        local seed=$((BASE_SEED + env_idx))
        local out_dir="$video_dir/$env_short"
        mkdir -p "$out_dir"

        echo "  [$model_name] Starting $env_short (seed=$seed)..."
        env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
            "$ROLLOUT_PYTHON" "$ROLLOUT_SCRIPT" \
            --env_name "$env" --n_episodes "$N_EPISODES" --n_envs 1 \
            --n_action_steps 8 --max_episode_steps 300 \
            --policy_client_host 127.0.0.1 --policy_client_port "$port" \
            --output_json "$out_dir/result.json" --video_dir "$out_dir" --seed "$seed" \
            > "$out_dir/stdout.log" 2>&1 &
        pids+=($!)
    done

    echo "  [$model_name] Waiting for ${#pids[@]} envs to finish..."
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done
    echo "  [$model_name] All envs done."

    # Collect results
    for env_idx in "${!ENVS[@]}"; do
        local env="${ENVS[$env_idx]}"
        local env_short=$(basename "$env")
        local json="$video_dir/$env_short/result.json"
        if [ -f "$json" ]; then
            python3 -c "
import json
r = json.load(open('$json'))
r['step'] = $step; r['model'] = '$model_name'
print(json.dumps(r))
" >> "$RESULTS_FILE"
            local sr=$(python3 -c "import json; print(json.load(open('$json'))['success_rate'])")
            echo "  [$model_name] $env_short: $sr"
        else
            echo "  [$model_name] $env_short: FAIL (no result)"
        fi
    done
}

echo "============================================"
echo " Head-to-head: $N_EPISODES episodes/task"
echo " All 6 envs in parallel per model"
echo "============================================"
echo "[stock] $STOCK"
echo "[ours]  $OURS_CKPT (step $STEP)"
echo ""

# ── Stock ────────────────────────────────────────────────────────────────────
echo "==============================="
echo " STOCK N1.6"
echo "==============================="
start_server "$STOCK" $PORT
run_all_envs "stock" $PORT 0 "$VIDEO_ROOT/stock"
kill_server $SERVER_PID
echo ""

# ── Ours ─────────────────────────────────────────────────────────────────────
echo "==============================="
echo " OURS (step $STEP)"
echo "==============================="
start_server "$OURS_CKPT" $PORT
run_all_envs "ours" $PORT $STEP "$VIDEO_ROOT/ours"
kill_server $SERVER_PID
echo ""

# ── Comparison table ─────────────────────────────────────────────────────────
echo "============================================"
echo " RESULTS ($N_EPISODES episodes/task)"
echo "============================================"
python3 -c "
import json

results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        results.append(json.loads(line))

stock = {r['env_name'].split('/')[-1]: r for r in results if r.get('model') == 'stock'}
ours = {r['env_name'].split('/')[-1]: r for r in results if r.get('model') == 'ours'}

print(f'{\"Task\":<40} {\"Stock\":>8} {\"Ours\":>8} {\"Diff\":>8}')
print('─' * 68)
s_sum, o_sum, n = 0, 0, 0
for env in sorted(set(list(stock.keys()) + list(ours.keys()))):
    s = stock.get(env, {}).get('success_rate', float('nan'))
    o = ours.get(env, {}).get('success_rate', float('nan'))
    d = o - s
    sign = '+' if d > 0 else ''
    print(f'{env:<40} {s:>7.0%} {o:>7.0%} {sign}{d:>7.0%}')
    s_sum += s; o_sum += o; n += 1
print('─' * 68)
sm, om = s_sum/n, o_sum/n
d = om - sm
sign = '+' if d > 0 else ''
print(f'{\"MEAN\":<40} {sm:>7.0%} {om:>7.0%} {sign}{d:>7.0%}')
"
