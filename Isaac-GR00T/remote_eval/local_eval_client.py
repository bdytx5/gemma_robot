"""
local_eval_client.py — runs on YOUR LOCAL MACHINE

Connects to sim_server.py running on H100 (exposed via ngrok),
loads GR00T model locally, drives the eval loop.

Usage:
    python remote_eval/local_eval_client.py \
        --server_url https://abc123.ngrok-free.app \
        --model_path /path/to/checkpoint-2000 \
        --env_name simpler_env_google/google_robot_pick_coke_can \
        --n_episodes 20 \
        --seed 42

Requirements on local machine:
    pip install requests msgpack numpy gr00t  (or install the Isaac-GR00T package)
"""

import argparse
import io
import time
from typing import Any

import msgpack
import numpy as np
import requests


# ── serialization (mirrors sim_server.py) ─────────────────────────────────────

def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray__": True, "data": buf.getvalue()}
    raise TypeError(f"Cannot encode {type(obj)}")

def _decode(obj):
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["data"]), allow_pickle=False)
    return obj

def pack(data) -> bytes:
    return msgpack.packb(data, default=_encode, use_bin_type=True)

def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)

HEADERS = {"Content-Type": "application/msgpack"}


# ── sim server client ──────────────────────────────────────────────────────────

class SimClient:
    def __init__(self, server_url: str, timeout: int = 60):
        self.base = server_url.rstrip("/")
        self.timeout = timeout

    def reset(self, env_name=None, seed=None, max_episode_steps=None,
              n_action_steps=None, video_dir=None):
        body = {}
        if env_name:         body["env_name"] = env_name
        if seed is not None: body["seed"] = seed
        if max_episode_steps: body["max_episode_steps"] = max_episode_steps
        if n_action_steps:    body["n_action_steps"] = n_action_steps
        if video_dir:         body["video_dir"] = video_dir
        r = requests.post(f"{self.base}/reset", data=pack(body),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def step(self, action: dict):
        r = requests.post(f"{self.base}/step", data=pack({"action": action}),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def ping(self):
        return requests.get(f"{self.base}/info", timeout=10).json()


# ── policy loader ──────────────────────────────────────────────────────────────

def load_policy(model_path: str, embodiment_tag: str = "OXE_GOOGLE"):
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag[embodiment_tag]
    print(f"[local] Loading model from {model_path} ...")
    policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(
            embodiment_tag=tag,
            model_path=model_path,
            device=0,
        )
    )
    print("[local] Model loaded.")
    return policy


# ── eval loop ─────────────────────────────────────────────────────────────────

def run_eval(
    server_url: str,
    model_path: str,
    env_name: str,
    n_episodes: int,
    seed: int,
    max_episode_steps: int,
    n_action_steps: int,
    video_dir: str = None,
    embodiment_tag: str = "OXE_GOOGLE",
):
    client = SimClient(server_url)

    print(f"[local] Pinging server {server_url} ...")
    info = client.ping()
    print(f"[local] Server info: {info}")

    policy = load_policy(model_path, embodiment_tag)

    successes = []
    ep_steps_list = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 1000
        print(f"\n[ep {ep+1}/{n_episodes}] Resetting env (seed={ep_seed}) ...")
        result = client.reset(
            env_name=env_name,
            seed=ep_seed,
            max_episode_steps=max_episode_steps,
            n_action_steps=n_action_steps,
            video_dir=video_dir,
        )
        obs = result["obs"]
        policy.reset()

        done = False
        step_count = 0
        t0 = time.time()

        while not done:
            action = policy.get_action(obs)
            result = client.step(action)
            obs = result["obs"]
            done = result["done"]
            success = result["success"]
            step_count = result["episode_steps"]

        elapsed = time.time() - t0
        successes.append(success)
        ep_steps_list.append(step_count)
        print(f"  → done in {step_count} steps ({elapsed:.1f}s) | success={success}")

    success_rate = sum(successes) / len(successes)
    print(f"\n{'='*50}")
    print(f"  Results: {sum(successes)}/{n_episodes} success  ({success_rate*100:.1f}%)")
    print(f"  Avg steps: {sum(ep_steps_list)/len(ep_steps_list):.1f}")
    print(f"{'='*50}")

    return successes


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", required=True,
                        help="ngrok URL of sim_server, e.g. https://abc123.ngrok-free.app")
    parser.add_argument("--model_path", required=True,
                        help="Path to GR00T checkpoint on local machine")
    parser.add_argument("--env_name", default="simpler_env_google/google_robot_pick_coke_can")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--video_dir", default=None,
                        help="If set, sim_server will save videos here (H100 path)")
    parser.add_argument("--embodiment_tag", default="OXE_GOOGLE")
    args = parser.parse_args()

    run_eval(
        server_url=args.server_url,
        model_path=args.model_path,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        embodiment_tag=args.embodiment_tag,
    )
