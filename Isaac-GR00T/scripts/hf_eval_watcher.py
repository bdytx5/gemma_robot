"""
Polls a HuggingFace model repo for new checkpoints, downloads each one,
and runs SimplerEnv evaluation. Results are logged to a JSONL file.

Designed to run on a separate eval server from training.

Usage:
    python scripts/hf_eval_watcher.py \
        --repo_id your-org/gr00t-widowx-finetune \
        --embodiment OXE_WIDOWX \
        --env_name simpler_env_widowx/widowx_spoon_on_towel \
        --n_episodes 20 \
        --n_envs 5 \
        --poll_interval 120 \
        --results_file eval_results.jsonl

Requirements on eval server:
    - Isaac-GR00T repo cloned
    - SimplerEnv set up: bash gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh
    - huggingface_hub installed: pip install huggingface_hub
    - HF_TOKEN env var set if repo is private
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SIMPLER_PYTHON = "gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"


def get_remote_checkpoints(api: HfApi, repo_id: str) -> list[str]:
    """Return sorted list of checkpoint subfolder names on the Hub."""
    try:
        items = api.list_repo_tree(repo_id, repo_type="model", recursive=False)
        folders = [item.path for item in items if item.path.startswith("checkpoint-")]
        return sorted(folders, key=lambda x: int(x.split("-")[-1]))
    except Exception as e:
        print(f"[Watcher] Error listing repo: {e}")
        return []


def download_checkpoint(repo_id: str, checkpoint_name: str, local_dir: str) -> str:
    """Download a single checkpoint subfolder. Returns local path to the checkpoint."""
    dest = os.path.join(local_dir, checkpoint_name)
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"[Watcher] {checkpoint_name} already downloaded, skipping")
        return dest

    print(f"[Watcher] Downloading {repo_id}/{checkpoint_name} → {dest}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=dest,
        allow_patterns=f"{checkpoint_name}/*",
    )
    # snapshot_download nests files under checkpoint_name/ inside local_dir
    nested = os.path.join(dest, checkpoint_name)
    if os.path.isdir(nested):
        return nested
    return dest


def wait_for_server(host: str, port: int, timeout: int = 120) -> bool:
    """Poll until the policy server is accepting connections."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(3)
    return False


def run_eval(checkpoint_path: str, args) -> dict:
    """Start policy server, run rollouts, shut server down. Returns result dict."""
    print(f"\n=== Evaluating {checkpoint_path} ===")

    server_proc = subprocess.Popen(
        [
            "uv", "run", "--extra=gpu",
            "python", "gr00t/eval/run_gr00t_server.py",
            "--model-path", checkpoint_path,
            "--embodiment-tag", args.embodiment,
            "--use-sim-policy-wrapper",
            "--port", str(args.server_port),
        ]
    )

    result = {
        "checkpoint": checkpoint_path,
        "env_name": args.env_name,
        "n_episodes": args.n_episodes,
        "success_rate": None,
        "episode_successes": None,
        "episode_infos": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        print(f"[Watcher] Waiting for server on port {args.server_port} ...")
        if not wait_for_server("127.0.0.1", args.server_port, timeout=180):
            raise RuntimeError("Policy server did not become ready in time")

        result_json = f"/tmp/eval_{os.path.basename(checkpoint_path)}_{int(time.time())}.json"

        proc = subprocess.run(
            [
                SIMPLER_PYTHON, "gr00t/eval/rollout_policy.py",
                "--n_episodes", str(args.n_episodes),
                "--env_name", args.env_name,
                "--n_envs", str(args.n_envs),
                "--n_action_steps", str(args.n_action_steps),
                "--max_episode_steps", str(args.max_episode_steps),
                "--policy_client_host", "127.0.0.1",
                "--policy_client_port", str(args.server_port),
                "--output_json", result_json,
            ],
            timeout=1200,
            capture_output=True,
            text=True,
        )

        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr[-1000:], file=sys.stderr)

        if proc.returncode == 0 and os.path.exists(result_json):
            with open(result_json) as f:
                eval_data = json.load(f)
            result.update(eval_data)
        else:
            result["error"] = proc.stderr[-500:] if proc.stderr else f"returncode={proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = "rollout timeout"
    except Exception as e:
        result["error"] = str(e)
    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. your-org/gr00t-widowx-finetune")
    parser.add_argument("--local_dir", default="/tmp/hf_checkpoints", help="Where to download checkpoints")
    parser.add_argument("--embodiment", default="OXE_WIDOWX")
    parser.add_argument("--env_name", default="simpler_env_widowx/widowx_spoon_on_towel")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--n_envs", type=int, default=5)
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--server_port", type=int, default=5555)
    parser.add_argument("--poll_interval", type=int, default=120, help="Seconds between HF polls")
    parser.add_argument("--results_file", default="eval_results.jsonl", help="JSONL file to append results to")
    args = parser.parse_args()

    api = HfApi()
    evaluated: set[str] = set()
    os.makedirs(args.local_dir, exist_ok=True)

    print(f"[Watcher] Polling https://huggingface.co/{args.repo_id} every {args.poll_interval}s")
    print(f"[Watcher] Results → {args.results_file}")

    while True:
        checkpoints = get_remote_checkpoints(api, args.repo_id)
        new = [c for c in checkpoints if c not in evaluated]

        for ckpt_name in new:
            local_ckpt = download_checkpoint(args.repo_id, ckpt_name, args.local_dir)
            result = run_eval(local_ckpt, args)

            with open(args.results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            sr = result.get("success_rate")
            status = f"{sr:.1%}" if sr is not None else f"ERROR: {result.get('error')}"
            print(f"[Result] {ckpt_name}: {status}")

            evaluated.add(ckpt_name)

        if new:
            print(f"[Watcher] Processed {len(new)} new checkpoint(s). Waiting {args.poll_interval}s ...")
        else:
            print(f"[Watcher] No new checkpoints. Waiting {args.poll_interval}s ...")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
