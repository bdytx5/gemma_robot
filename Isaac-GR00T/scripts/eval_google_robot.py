#!/usr/bin/env python3
"""
Run eval on all 6 Google robot SimplerEnv tasks, capture videos, log to W&B.

Usage:
    python scripts/eval_google_robot.py \
        --checkpoint_path ./output/gr00t-eagle2_5-fractal/checkpoint-200 \
        --wandb_project finetune-gr00t-n1d6 \
        --n_episodes 5

The step index is auto-detected from the checkpoint folder name (checkpoint-<N>)
or can be set explicitly with --step.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import tempfile
from pathlib import Path

GOOGLE_ENVS = [
    "simpler_env_google/google_robot_open_drawer",
    "simpler_env_google/google_robot_close_drawer",
    "simpler_env_google/google_robot_place_in_closed_drawer",
    "simpler_env_google/google_robot_pick_coke_can",
    "simpler_env_google/google_robot_pick_object",
    "simpler_env_google/google_robot_move_near",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
SIMPLER_PYTHON = REPO_ROOT / "gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
SERVER_SCRIPT = REPO_ROOT / "gr00t/eval/run_gr00t_server.py"
ROLLOUT_SCRIPT = REPO_ROOT / "gr00t/eval/rollout_policy.py"


def wait_for_server(port: int, timeout: int = 180, server_proc: subprocess.Popen = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        # If the server process already died (e.g. OOM), bail immediately
        if server_proc and server_proc.poll() is not None:
            print(f"[eval] Server process died (exit code {server_proc.returncode}) — not waiting.")
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(3)
    return False


def start_server(checkpoint_path: str, port: int, embodiment: str, device: str = "cuda") -> subprocess.Popen:
    cmd = [
        str(VENV_PYTHON),
        str(SERVER_SCRIPT),
        "--model_path", checkpoint_path,
        "--embodiment_tag", embodiment,
        "--use_sim_policy_wrapper",
        "--port", str(port),
        "--device", device,
    ]
    env = os.environ.copy()
    eagle_repo = str(REPO_ROOT.parent / "Eagle" / "Eagle2_5")
    if os.path.isdir(eagle_repo):
        env["PYTHONPATH"] = f"{eagle_repo}:{env.get('PYTHONPATH', '')}"
    print(f"[eval] Starting server: {' '.join(cmd)}")
    # Start in new process group so we can kill the entire tree on cleanup
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def kill_process_tree(pid: int):
    """Kill a process and all its descendants."""
    import signal as _signal
    # Collect all descendant PIDs via /proc
    def _get_children(parent_pid):
        try:
            out = subprocess.check_output(
                ["pgrep", "-P", str(parent_pid)], text=True, stderr=subprocess.DEVNULL
            )
            return [int(p) for p in out.strip().split("\n") if p.strip()]
        except (subprocess.CalledProcessError, ValueError):
            return []

    def _collect_tree(root):
        pids = []
        for child in _get_children(root):
            pids.extend(_collect_tree(child))
        pids.append(root)
        return pids

    all_pids = _collect_tree(pid)
    # SIGTERM first
    for p in all_pids:
        try:
            os.kill(p, _signal.SIGTERM)
        except OSError:
            pass
    time.sleep(2)
    # SIGKILL stragglers
    for p in all_pids:
        try:
            os.kill(p, _signal.SIGKILL)
        except OSError:
            pass


def kill_server(server: subprocess.Popen):
    """Kill the server and all its children (torch compile workers, etc.)."""
    print(f"[eval] Killing server tree (root PID {server.pid})...")
    kill_process_tree(server.pid)
    try:
        server.wait(timeout=5)
    except Exception:
        pass
    # Also kill any orphaned rollout_policy.py or run_gr00t_server.py on our port
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_gr00t_server.py"], text=True, stderr=subprocess.DEVNULL
        )
        for pid_str in out.strip().split("\n"):
            if pid_str.strip():
                kill_process_tree(int(pid_str.strip()))
    except (subprocess.CalledProcessError, ValueError):
        pass
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "rollout_policy.py"], text=True, stderr=subprocess.DEVNULL
        )
        for pid_str in out.strip().split("\n"):
            if pid_str.strip():
                kill_process_tree(int(pid_str.strip()))
    except (subprocess.CalledProcessError, ValueError):
        pass
    print("[eval] Server process cleaned up.")


def run_env(env_name: str, port: int, n_episodes: int, video_dir: str, result_json: str, seed: int = 42, n_envs: int = 1) -> dict:
    rollout_python = str(SIMPLER_PYTHON) if SIMPLER_PYTHON.exists() else str(VENV_PYTHON)
    cmd = [
        rollout_python,
        str(ROLLOUT_SCRIPT),
        "--env_name", env_name,
        "--n_episodes", str(n_episodes),
        "--n_envs", str(n_envs),
        "--n_action_steps", "8",
        "--max_episode_steps", "300",
        "--policy_client_host", "127.0.0.1",
        "--policy_client_port", str(port),
        "--output_json", result_json,
        "--video_dir", video_dir,
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    print(f"[eval] Running {env_name} ({n_episodes} episodes)...")
    t0 = time.time()
    timeout = max(600, n_episodes * 120)  # 2 min per episode minimum
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    elapsed = time.time() - t0
    if proc.stdout:
        print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(f"[eval] ERROR for {env_name}:\n{proc.stderr[-1000:]}", file=sys.stderr)
        return {"env_name": env_name, "success_rate": None, "error": proc.stderr[-500:], "elapsed": elapsed}

    if Path(result_json).exists():
        with open(result_json) as f:
            result = json.load(f)
        result["elapsed"] = elapsed
        return result
    return {"env_name": env_name, "success_rate": None, "error": "no output json", "elapsed": elapsed}


def log_to_wandb(results: list[dict], video_root: str, step: int, project: str, run_name: str, run_id: str):
    try:
        import wandb
    except ImportError:
        print("[eval] wandb not installed, skipping logging")
        return

    # Always resume the same run so all checkpoints appear on one timeline
    run = wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume="allow",
    )

    log = {}
    table_data = []
    for r in results:
        env_short = r["env_name"].split("/")[-1]
        sr = r.get("success_rate")
        if sr is not None:
            log[f"eval/{env_short}/success_rate"] = sr
        if r.get("episode_successes"):
            log[f"eval/{env_short}/n_success"] = sum(r["episode_successes"])
            log[f"eval/{env_short}/n_episodes"] = len(r["episode_successes"])
        n_requested = r.get("n_episodes_requested") or r.get("n_episodes")
        n_completed = len(r["episode_successes"]) if r.get("episode_successes") else 0
        if n_requested and n_completed < n_requested:
            log[f"eval/{env_short}/episodes_missing"] = n_requested - n_completed
        # Log task description so we can verify determinism across checkpoints
        if r.get("task_description"):
            log[f"eval/{env_short}/task_description"] = r["task_description"]
        table_data.append([step, env_short, sr, r.get("task_description", "")])

        # One video panel per env — log each episode video under eval/video/<env>
        env_video_dir = Path(video_root) / env_short
        ep_videos = sorted(env_video_dir.rglob("*.mp4"))
        for i, vf in enumerate(ep_videos):
            log[f"eval/video/{env_short}/ep{i:02d}"] = wandb.Video(str(vf), fps=10, format="mp4")

    # Aggregate mean across envs
    rates = [r["success_rate"] for r in results if r.get("success_rate") is not None]
    if rates:
        log["eval/mean_success_rate"] = sum(rates) / len(rates)

    # Summary table for this checkpoint
    table = wandb.Table(columns=["step", "env", "success_rate", "task_description"], data=table_data)
    log["eval/results_table"] = table

    run.log(log, step=step)
    run.finish()
    print(f"[eval] Logged to W&B run '{run_id}': step={step}, mean_success={log.get('eval/mean_success_rate')}")
    for row in table_data:
        print(f"  {row[1]}: {row[2]:.1%} — \"{row[3]}\"" if row[2] is not None else f"  {row[1]}: ERROR")


def detect_step(checkpoint_path: str) -> int:
    m = re.search(r"checkpoint-(\d+)", checkpoint_path)
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True, help="Path to GR00T checkpoint dir")
    parser.add_argument("--step", type=int, default=None, help="Training step (auto-detected from checkpoint name if not set)")
    parser.add_argument("--n_episodes", type=int, default=5, help="Episodes per env")
    parser.add_argument("--n_envs", type=int, default=1, help="Parallel envs per task")
    parser.add_argument("--no_server", action="store_true", help="Skip launching policy server (use existing one)")
    parser.add_argument("--port", type=int, default=5556, help="Policy server port")
    parser.add_argument("--wandb_project", default="finetune-gr00t-n1d6", help="W&B project name")
    parser.add_argument("--video_root", default="/tmp/gr00t_eval_videos", help="Root dir for video output")
    parser.add_argument("--results_file", default="eval_results.jsonl", help="JSONL file to append results to")
    parser.add_argument("--envs", nargs="+", default=GOOGLE_ENVS, help="Override env list")
    parser.add_argument("--wandb_run_id", default=None,
                        help="Fixed W&B run ID to resume (all checkpoints log to one run). "
                             "Auto-derived from output dir name if not set.")
    parser.add_argument("--no_wandb", action="store_true", help="Skip W&B logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for deterministic eval (each env gets base_seed + env_index)")
    parser.add_argument("--device", default="cuda",
                        help="Device for model inference (cuda or cpu)")
    args = parser.parse_args()

    step = args.step if args.step is not None else detect_step(args.checkpoint_path)
    video_root = os.path.join(args.video_root, f"step-{step:06d}")
    # run_name is human-readable; run_id is the stable identity W&B uses for resume
    run_name = "270m-balanced-eval"
    run_id = args.wandb_run_id or "270m-balanced-eval"

    print(f"[eval] Checkpoint: {args.checkpoint_path}")
    print(f"[eval] Step: {step}  |  Envs: {len(args.envs)}  |  Episodes/env: {args.n_episodes}")
    print(f"[eval] Videos → {video_root}")

    if args.no_server:
        server = None
        print(f"[eval] Using existing policy server on port {args.port}")
    else:
        server = start_server(args.checkpoint_path, args.port, "OXE_GOOGLE", device=args.device)
        print(f"[eval] Waiting for server on port {args.port}...")
        if not wait_for_server(args.port, timeout=180, server_proc=server):
            raise RuntimeError("Policy server did not come up in time (crashed or timed out)")
        print("[eval] Server ready.")
    try:

        all_results = []
        for env_idx, env_name in enumerate(args.envs):
            env_short = env_name.split("/")[-1]
            env_video_dir = os.path.join(video_root, env_short)
            result_json = os.path.join(env_video_dir, "result.json")
            env_seed = args.seed + env_idx  # deterministic per-env seed
            try:
                result = run_env(env_name, args.port, args.n_episodes, env_video_dir, result_json, seed=env_seed, n_envs=args.n_envs)
            except subprocess.TimeoutExpired:
                result = {"env_name": env_name, "success_rate": None, "error": "timeout"}
            result["step"] = step
            all_results.append(result)

            sr = result.get("success_rate")
            status = f"{sr:.1%}" if sr is not None else f"ERROR: {result.get('error', '?')}"
            print(f"[result] {env_short}: {status}")

        # Only log if we got at least one real result
        good_results = [r for r in all_results if r.get("success_rate") is not None]
        if not good_results:
            print("[eval] No successful env results — skipping JSONL and W&B logging.")
            sys.exit(1)

        # Append to JSONL (only results with actual data)
        with open(args.results_file, "a") as f:
            for r in good_results:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] {len(good_results)} results appended to {args.results_file}")

        if not args.no_wandb:
            log_to_wandb(good_results, video_root, step, args.wandb_project, run_name, run_id)

    finally:
        if server is not None:
            print("[eval] Shutting down server...")
            kill_server(server)

    rates = [r["success_rate"] for r in all_results if r.get("success_rate") is not None]
    if rates:
        print(f"\n[eval] Mean success rate: {sum(rates)/len(rates):.1%} across {len(rates)} envs")


if __name__ == "__main__":
    main()
