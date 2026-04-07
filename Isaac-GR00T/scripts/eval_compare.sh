#!/usr/bin/env bash
# eval_compare.sh — Head-to-head: our model vs stock N1.6
# Runs all 6 envs in parallel against each model server.
# Stock model uses ORIGINAL unmodified NVIDIA code (flash_attn, default settings).
#
# Usage:
#   bash scripts/eval_compare.sh
#   bash scripts/eval_compare.sh --ours /path/to/checkpoint
#   N_EPISODES=200 N_ENVS=20 bash scripts/eval_compare.sh

set +e  # Don't exit on error — we handle failures per-task

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

N_EPISODES=${N_EPISODES:-10}
N_ENVS=${N_ENVS:-1}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-504}  # NVIDIA default
PORT=${PORT:-5556}
RESULTS_FILE=${RESULTS_FILE:-"eval_results_compare.jsonl"}
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
SIMPLER_PYTHON="$REPO_ROOT/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
ROLLOUT_SCRIPT="$REPO_ROOT/gr00t/eval/rollout_policy.py"
SERVER_SCRIPT="$REPO_ROOT/gr00t/eval/run_gr00t_server.py"
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
BASE_SEED=${BASE_SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-"finetune-gr00t-n1d6"}

# Stock model uses unmodified NVIDIA code from a clean clone
STOCK_REPO="/home/ubuntu/gemma_robot/gr00t_stock"
STOCK_SERVER_SCRIPT="$STOCK_REPO/gr00t/eval/run_gr00t_server.py"

ALL_ENVS=(
    "simpler_env_google/google_robot_open_drawer"
    "simpler_env_google/google_robot_close_drawer"
    "simpler_env_google/google_robot_place_in_closed_drawer"
    "simpler_env_google/google_robot_pick_coke_can"
    "simpler_env_google/google_robot_pick_object"
    "simpler_env_google/google_robot_move_near"
)
if [ -n "${ENVS_FILTER:-}" ]; then
    # ENVS_FILTER: comma-separated substring(s) to match against task names
    ENVS=()
    IFS=',' read -ra FILTERS <<< "$ENVS_FILTER"
    for e in "${ALL_ENVS[@]}"; do
        for f in "${FILTERS[@]}"; do
            if [[ "$e" == *"$f"* ]]; then
                ENVS+=("$e")
                break
            fi
        done
    done
elif [ -n "${N_TASKS:-}" ] && [ "$N_TASKS" -lt "${#ALL_ENVS[@]}" ]; then
    ENVS=("${ALL_ENVS[@]:0:$N_TASKS}")
else
    ENVS=("${ALL_ENVS[@]}")
fi

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
mkdir -p "$VIDEO_ROOT"
# Rebuild RESULTS_FILE from any existing result.json files (resume support, no duplicates)
rm -f "$RESULTS_FILE"
for model_tag in stock ours; do
    model_name="$model_tag"
    step_val=0
    [ "$model_tag" = "ours" ] && step_val="$STEP"
    for env in "${ALL_ENVS[@]}"; do
        env_short=$(basename "$env")
        json="$VIDEO_ROOT/$model_tag/$env_short/result.json"
        if [ -f "$json" ]; then
            python3 -c "
import json
r = json.load(open('$json'))
r['step'] = $step_val; r['model'] = '$model_name'
print(json.dumps(r))
" >> "$RESULTS_FILE"
        fi
    done
done

start_server() {
    local model_path=$1 port=$2 use_stock=${3:-0}
    if [ "$use_stock" = "1" ]; then
        echo "[server] Using STOCK gr00t code from $STOCK_REPO"
        PYTHONPATH="$STOCK_REPO:$EAGLE_REPO:${PYTHONPATH:-}" $PYTHON \
            "$STOCK_REPO/run_stock.py" "$STOCK_SERVER_SCRIPT" \
            --model_path "$model_path" --embodiment_tag OXE_GOOGLE \
            --use_sim_policy_wrapper --port "$port" --device cuda > /tmp/gr00t_server_${port}.log 2>&1 &
    else
        $PYTHON $SERVER_SCRIPT \
            --model_path "$model_path" --embodiment_tag OXE_GOOGLE \
            --use_sim_policy_wrapper --port "$port" --device cuda > /tmp/gr00t_server_${port}.log 2>&1 &
    fi
    SERVER_PID=$!
    echo "[server] Started PID $SERVER_PID on port $port, waiting..."
    for i in $(seq 1 600); do
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
    pkill -9 -f run_gr00t_server 2>/dev/null || true
    pkill -9 -f rollout_policy 2>/dev/null || true
    sleep 2
}

run_all_envs() {
    local model_name=$1 port=$2 step=$3 video_dir=$4

    # Run tasks ONE AT A TIME (sequential) to avoid Vulkan device lost errors,
    # but each task uses N_ENVS parallel sub-envs for speed.
    for env_idx in "${!ENVS[@]}"; do
        local env="${ENVS[$env_idx]}"
        local env_short=$(basename "$env")
        local seed=$((BASE_SEED + env_idx))
        local out_dir="$video_dir/$env_short"
        mkdir -p "$out_dir"

        # Skip if already completed (result already in RESULTS_FILE from startup rebuild)
        if [ -f "$out_dir/result.json" ]; then
            local sr=$(python3 -c "import json; print(json.load(open('$out_dir/result.json'))['success_rate'])")
            echo "  [$model_name] $env_short: already done ($sr), skipping"
            continue
        fi

        local max_retries=10
        local attempt=0
        local success=false

        while [ $attempt -lt $max_retries ] && [ "$success" = "false" ]; do
            attempt=$((attempt + 1))
            echo "  [$model_name] Running $env_short attempt $attempt/$max_retries (seed=$seed, n_episodes=$N_EPISODES, n_envs=$N_ENVS, max_steps=$MAX_EPISODE_STEPS)..."
            rm -f "$out_dir/result.json"
            env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
                "$ROLLOUT_PYTHON" "$ROLLOUT_SCRIPT" \
                --env_name "$env" --n_episodes "$N_EPISODES" --n_envs "$N_ENVS" \
                --n_action_steps 8 --max_episode_steps "$MAX_EPISODE_STEPS" \
                --policy_client_host 127.0.0.1 --policy_client_port "$port" \
                --output_json "$out_dir/result.json" --video_dir "$out_dir" --seed "$seed" \
                > "$out_dir/stdout.log" 2>&1
            local rc=$?

            if [ -f "$out_dir/result.json" ]; then
                python3 -c "
import json
r = json.load(open('$out_dir/result.json'))
r['step'] = $step; r['model'] = '$model_name'
print(json.dumps(r))
" >> "$RESULTS_FILE"
                local sr=$(python3 -c "import json; print(json.load(open('$out_dir/result.json'))['success_rate'])")
                echo "  [$model_name] $env_short: $sr (exit=$rc)"
                success=true
            else
                echo "  [$model_name] $env_short: FAIL attempt $attempt (exit=$rc, no result)"
                if [ $attempt -lt $max_retries ]; then
                    echo "  [$model_name] $env_short: Retrying in 5s..."
                    sleep 5
                else
                    echo "  [$model_name] $env_short: FAILED after $max_retries attempts"
                fi
            fi
        done
    done
    echo "  [$model_name] All envs done."
}

echo "============================================"
echo " Head-to-head: $N_EPISODES episodes/task"
echo " n_envs=$N_ENVS  max_episode_steps=$MAX_EPISODE_STEPS"
echo " All 6 envs in parallel per model"
echo "============================================"
echo "[stock] $STOCK (ORIGINAL NVIDIA code)"
echo "[ours]  $OURS_CKPT (step $STEP)"
echo ""

# ── Stock ────────────────────────────────────────────────────────────────────
echo "==============================="
echo " STOCK N1.6 (using ORIGINAL NVIDIA code)"
echo "==============================="
start_server "$STOCK" $PORT 1
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

# ── Upload to W&B as comparison table with videos ────────────────────────────
echo ""
echo "[wandb] Uploading comparison table..."
$PYTHON -c "
import json, wandb, os, glob

results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        results.append(json.loads(line))

stock = {r['env_name'].split('/')[-1]: r for r in results if r.get('model') == 'stock'}
ours = {r['env_name'].split('/')[-1]: r for r in results if r.get('model') == 'ours'}
all_envs = sorted(set(list(stock.keys()) + list(ours.keys())))

# Create W&B run
run = wandb.init(
    project='$WANDB_PROJECT',
    name='compare-stock-vs-ours-step$STEP',
    id='compare-stock-vs-ours-step$STEP',
    resume='allow',
)

# ── Summary metrics ──
s_rates, o_rates = [], []
for env in all_envs:
    s = stock.get(env, {}).get('success_rate', 0)
    o = ours.get(env, {}).get('success_rate', 0)
    run.summary[f'stock/{env}'] = s
    run.summary[f'ours/{env}'] = o
    run.summary[f'diff/{env}'] = o - s
    s_rates.append(s)
    o_rates.append(o)

run.summary['stock/mean'] = sum(s_rates) / len(s_rates)
run.summary['ours/mean'] = sum(o_rates) / len(o_rates)
run.summary['diff/mean'] = run.summary['ours/mean'] - run.summary['stock/mean']

# ── Comparison table: one row per env ──
columns = ['task', 'stock_success_rate', 'ours_success_rate', 'diff', 'stock_episodes', 'ours_episodes']
table = wandb.Table(columns=columns)
for env in all_envs:
    s = stock.get(env, {})
    o = ours.get(env, {})
    sr_s = s.get('success_rate', 0)
    sr_o = o.get('success_rate', 0)
    ep_s = len(s.get('episode_successes', [])) if s else 0
    ep_o = len(o.get('episode_successes', [])) if o else 0
    table.add_data(env, sr_s, sr_o, sr_o - sr_s, ep_s, ep_o)
# Mean row
table.add_data('MEAN', sum(s_rates)/len(s_rates), sum(o_rates)/len(o_rates),
               sum(o_rates)/len(o_rates) - sum(s_rates)/len(s_rates),
               sum(len(stock.get(e, {}).get('episode_successes', [])) for e in all_envs),
               sum(len(ours.get(e, {}).get('episode_successes', [])) for e in all_envs))
run.log({'comparison_table': table})

# ── Per-episode table with videos (side-by-side) ──
ep_columns = ['task', 'episode', 'stock_success', 'stock_video', 'ours_success', 'ours_video']
ep_table = wandb.Table(columns=ep_columns)
video_root = '$VIDEO_ROOT'
for env in all_envs:
    s_r = stock.get(env, {})
    o_r = ours.get(env, {})
    s_vid_dir = os.path.join(video_root, 'stock', env)
    o_vid_dir = os.path.join(video_root, 'ours', env)
    s_vids = sorted(glob.glob(os.path.join(s_vid_dir, '*.mp4'))) if os.path.isdir(s_vid_dir) else []
    o_vids = sorted(glob.glob(os.path.join(o_vid_dir, '*.mp4'))) if os.path.isdir(o_vid_dir) else []
    n_eps = max(len(s_r.get('episode_successes', [])), len(o_r.get('episode_successes', [])))
    s_succs = s_r.get('episode_successes', [])
    o_succs = o_r.get('episode_successes', [])
    for i in range(n_eps):
        s_succ = int(s_succs[i]) if i < len(s_succs) else None
        o_succ = int(o_succs[i]) if i < len(o_succs) else None
        s_vid = wandb.Video(s_vids[i], format='mp4') if i < len(s_vids) else None
        o_vid = wandb.Video(o_vids[i], format='mp4') if i < len(o_vids) else None
        ep_table.add_data(env, i, s_succ, s_vid, o_succ, o_vid)
run.log({'episode_table': ep_table})

run.finish()
print('[wandb] Done.')
"

# ── Upload to Weave for comparison ─────────────────────────────────────────
echo ""
echo "[weave] Logging results to Weave..."
$PYTHON -c "
import json, os, glob, weave, cv2
import numpy as np
from PIL import Image

def video_to_grid(video_path, n_frames=6):
    \"\"\"Extract n_frames from a video (first, last, and evenly spaced in between)
    and return a PIL Image grid.\"\"\"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Pick frame indices: first, last, and evenly spaced in between
    if total <= n_frames:
        indices = list(range(total))
    else:
        indices = [0] + [int(round(i * (total - 1) / (n_frames - 1))) for i in range(1, n_frames - 1)] + [total - 1]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    # Build grid: 2 rows x 3 columns
    cols = 3
    rows = 2
    gap = 4
    h, w = frames[0].shape[:2]
    # Pad frames list to fill grid if needed
    while len(frames) < rows * cols:
        frames.append(np.ones((h, w, 3), dtype=np.uint8) * 40)
    grid_w = cols * w + (cols - 1) * gap
    grid_h = rows * h + (rows - 1) * gap
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40
    for idx, f in enumerate(frames[:rows * cols]):
        r, c = divmod(idx, cols)
        y = r * (h + gap)
        x = c * (w + gap)
        grid[y:y+h, x:x+w] = f

    return Image.fromarray(grid)


weave.init('byyoung3/$WANDB_PROJECT')

results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        results.append(json.loads(line))

stock = {r['env_name'].split('/')[-1]: r for r in results if r.get('model') == 'stock'}
ours = {r['env_name'].split('/')[-1]: r for r in results if r.get('model') == 'ours'}
all_envs = sorted(set(list(stock.keys()) + list(ours.keys())))
video_root = '$VIDEO_ROOT'

# Publish dataset (one row per task+episode)
rows = []
for env in all_envs:
    s_r = stock.get(env, {})
    o_r = ours.get(env, {})
    n_eps = max(len(s_r.get('episode_successes', [])), len(o_r.get('episode_successes', [])))
    for i in range(n_eps):
        rows.append({'task': env, 'episode': i})
dataset = weave.Dataset(name='compare-stock-vs-ours-step$STEP', rows=rows)
weave.publish(dataset)

# Log evaluations for each model
for model_name, model_results in [('stock-n1.6', stock), ('ours-step$STEP', ours)]:
    tag = 'stock' if 'stock' in model_name else 'ours'
    ev = weave.EvaluationLogger(
        name='compare-stock-vs-ours-step$STEP',
        model=model_name,
        dataset='compare-stock-vs-ours-step$STEP',
        scorers=['success'],
    )
    for env in all_envs:
        r = model_results.get(env, {})
        vid_dir = os.path.join(video_root, tag, env)
        vids = sorted(glob.glob(os.path.join(vid_dir, '*.mp4'))) if os.path.isdir(vid_dir) else []
        for i, succ in enumerate(r.get('episode_successes', [])):
            grid_img = video_to_grid(vids[i]) if i < len(vids) else None
            pred = ev.log_prediction(
                inputs={'task': env, 'episode': i},
                output={'success': bool(succ), 'frames': grid_img},
            )
            pred.log_score(scorer='success', score=bool(succ))
            pred.finish()

    # Summary
    rates = [model_results.get(e, {}).get('success_rate', 0) for e in all_envs]
    summary = {e: model_results.get(e, {}).get('success_rate', 0) for e in all_envs}
    summary['mean_success_rate'] = sum(rates) / len(rates) if rates else 0
    ev.log_summary(summary)

print('[weave] Done. Compare models in Weave Evals tab.')
"
