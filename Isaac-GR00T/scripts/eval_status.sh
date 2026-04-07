#!/usr/bin/env bash
# eval_status.sh — Show live progress of eval_compare.sh run
# Usage: bash scripts/eval_status.sh [VIDEO_ROOT]

VIDEO_ROOT=${1:-/tmp/gr00t_eval_compare}

clear 2>/dev/null || true
echo "========================================"
echo " Eval Progress — $(date '+%H:%M:%S')"
echo "========================================"

for model in stock ours; do
    model_dir="$VIDEO_ROOT/$model"
    [ -d "$model_dir" ] || continue
    echo ""
    echo "  [$model]"
    printf "  %-35s %6s %10s %8s %8s %7s\n" "TASK" "EP" "STEPS" "SUCCESS" "RATE" "TIME"
    printf "  %-35s %6s %10s %8s %8s %7s\n" "---" "--" "-----" "-------" "----" "----"

    for task_dir in "$model_dir"/*/; do
        [ -d "$task_dir" ] || continue
        task=$(basename "$task_dir")
        progress="$task_dir/progress.json"
        result="$task_dir/result.json"

        if [ -f "$result" ]; then
            # Completed
            rate=$(python3 -c "import json; r=json.load(open('$result')); print(f\"{r['success_rate']:.0%}\")" 2>/dev/null || echo "?")
            eps=$(python3 -c "import json; r=json.load(open('$result')); print(r['n_episodes'])" 2>/dev/null || echo "?")
            printf "  %-35s %4s/%-1s %10s %8s %8s %7s\n" "$task" "$eps" "$eps" "done" "-" "$rate" "done"
        elif [ -f "$progress" ]; then
            # In progress
            info=$(python3 -c "
import json
p = json.load(open('$progress'))
ep = p['completed_episodes']
n = p['n_episodes']
step = p['step']
mx = p['max_steps_per_episode']
succ = p['successes_so_far']
elapsed = p['elapsed']
ep_steps = p.get('current_ep_steps', [])
cur = '/'.join(str(s) for s in ep_steps)
rate = f'{succ}/{ep}' if ep > 0 else '-'
mins = int(elapsed // 60)
secs = int(elapsed % 60)
print(f'{ep}|{n}|{cur}/{mx}|{succ}|{rate}|{mins}m{secs:02d}s')
" 2>/dev/null)
            if [ -n "$info" ]; then
                IFS='|' read -r ep n steps succ rate elapsed <<< "$info"
                printf "  %-35s %4s/%-1s %10s %8s %8s %7s\n" "$task" "$ep" "$n" "$steps" "$succ" "$rate" "$elapsed"
            else
                printf "  %-35s %6s\n" "$task" "starting..."
            fi
        else
            printf "  %-35s %6s\n" "$task" "waiting"
        fi
    done
done

echo ""
# Show overall phase
if [ -f /tmp/eval_pick_object_test.log ]; then
    phase=$(grep -oE "(STOCK|OURS|RESULTS|wandb|weave)" /tmp/eval_pick_object_test.log | tail -1)
    echo "  Phase: $phase"
fi
echo "========================================"
