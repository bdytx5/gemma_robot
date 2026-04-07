#!/usr/bin/env python3
"""
GR00T Demo Worker — polls the API for queued jobs, runs eval, reports back.
Waits if the GPU is busy (another eval is running) before starting a new job.

Usage:
    API_URL=https://your-api.com RESEND_API_KEY=xxx python scripts/demo_worker.py
"""
import os
import sys
import time
import json
import subprocess
import requests
import glob

API_URL = os.environ.get("API_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))  # seconds between polls
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(REPO_ROOT, "scripts", "eval_single.sh")
VIDEO_ROOT = os.environ.get("VIDEO_ROOT", "/tmp/gr00t_demo")

# Map API task names to env names used by eval_single.sh
TASK_TO_ENV = {
    "open_drawer":             "simpler_env_google/google_robot_open_drawer",
    "close_drawer":            "simpler_env_google/google_robot_close_drawer",
    "place_in_closed_drawer":  "simpler_env_google/google_robot_place_in_closed_drawer",
    "pick_coke_can":           "simpler_env_google/google_robot_pick_coke_can",
    "pick_object":             "simpler_env_google/google_robot_pick_object",
    "move_near":               "simpler_env_google/google_robot_move_near",
}


def gpu_is_busy():
    """Return True if another eval process is already using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
        return len(pids) > 0
    except Exception:
        return False


def run_eval(job_id: str, task: str) -> tuple[bool, str | None]:
    """
    Run eval_single.sh for the given task.
    Returns (success, video_url).
    """
    env_name = TASK_TO_ENV.get(task)
    if not env_name:
        print(f"[worker] Unknown task: {task}")
        return False, None

    out_dir = os.path.join(VIDEO_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

    print(f"[worker] Running eval for job {job_id} task={task}")
    proc = subprocess.run(
        ["bash", EVAL_SCRIPT, env_name, out_dir],
        env=env,
        capture_output=True,
        text=True,
    )

    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(f"[worker] eval_single.sh failed (exit={proc.returncode}): {proc.stderr[-500:]}")
        return False, None

    # Read result
    result_file = os.path.join(out_dir, "result.json")
    if not os.path.exists(result_file):
        print(f"[worker] No result.json found in {out_dir}")
        return False, None

    with open(result_file) as f:
        result = json.load(f)

    success = bool(result.get("episode_successes", [False])[0])

    # Find video file
    videos = glob.glob(os.path.join(out_dir, "*.mp4"))
    video_url = None
    if videos:
        # If serving via tunnel, build URL; otherwise just return local path
        base_url = os.environ.get("VIDEO_BASE_URL", "")
        rel = os.path.relpath(videos[0], VIDEO_ROOT)
        video_url = f"{base_url}/videos/{rel}" if base_url else videos[0]

    return success, video_url


def main():
    print(f"[worker] Starting. API={API_URL}, poll every {POLL_INTERVAL}s")
    while True:
        try:
            # Wait if GPU is busy
            if gpu_is_busy():
                print("[worker] GPU busy, waiting...")
                time.sleep(POLL_INTERVAL)
                continue

            resp = requests.get(f"{API_URL}/next_job", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            job = data.get("job")

            if not job:
                time.sleep(POLL_INTERVAL)
                continue

            job_id = job["id"]
            task = job["task"]
            print(f"[worker] Claimed job {job_id} task={task} email={job['email']}")

            success, video_url = run_eval(job_id, task)

            if video_url:
                requests.patch(
                    f"{API_URL}/jobs/{job_id}/complete",
                    json={"result_url": video_url, "success": success},
                    timeout=10,
                )
                print(f"[worker] Job {job_id} done: success={success} url={video_url}")
            else:
                requests.patch(f"{API_URL}/jobs/{job_id}/fail", timeout=10)
                print(f"[worker] Job {job_id} failed")

        except KeyboardInterrupt:
            print("[worker] Shutting down.")
            break
        except Exception as e:
            print(f"[worker] Error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
