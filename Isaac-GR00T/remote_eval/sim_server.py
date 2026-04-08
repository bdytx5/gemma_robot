"""
sim_server.py — runs on H100

Wraps SimplerEnv in a FastAPI HTTP server so a remote client (local machine)
can drive the eval loop over ngrok.

Usage:
    cd Isaac-GR00T
    PYTHONPATH=. VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        .venv/bin/python remote_eval/sim_server.py \
        --env_name simpler_env_google/google_robot_pick_coke_can \
        --port 8080 --seed 42

Then expose via ngrok:
    ngrok http 8080
"""

import argparse
import io
import sys
import os
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = REPO_ROOT / "external_dependencies/SimplerEnv"
MANISKILL = EXTERNAL / "ManiSkill2_real2sim"
SIMPLER_SITE = REPO_ROOT / "gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/lib/python3.10/site-packages"

# Prepend repo/sim paths so gr00t takes priority; append simpler_uv site-packages
# last so simpler_env is found without shadowing main-venv packages (e.g. transformers)
for p in [str(EXTERNAL), str(MANISKILL), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
if str(SIMPLER_SITE) not in sys.path:
    sys.path.append(str(SIMPLER_SITE))

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

# ── serialization ─────────────────────────────────────────────────────────────
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

def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)

def msgpack_response(data) -> Response:
    return Response(content=pack(data), media_type="application/msgpack")

# ── global state ──────────────────────────────────────────────────────────────
app = FastAPI()
_env = None
_episode_steps = 0
_max_steps = 504
_policy_client = None   # ZMQ PolicyClient connecting to gr00t inference server


def _connect_policy_client(host: str, port: int):
    global _policy_client
    print(f"[sim_server] Connecting to gr00t policy server at {host}:{port} ...")
    from gr00t.policy.server_client import PolicyClient
    _policy_client = PolicyClient(host=host, port=port)
    print("[sim_server] Policy client connected.")


def _build_env(env_name: str, seed: int, max_episode_steps: int, n_action_steps: int,
               video_dir: Optional[str]):
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    from gr00t.eval.rollout_policy import create_eval_env, WrapperConfigs, VideoConfig, MultiStepConfig

    register_simpler_envs()

    video_cfg = VideoConfig(
        video_dir=video_dir,
        max_episode_steps=max_episode_steps,
    )
    ms_cfg = MultiStepConfig(
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
    )
    wrapper_configs = WrapperConfigs(video=video_cfg, multistep=ms_cfg)

    env = create_eval_env(
        env_name=env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )
    obs, info = env.reset(seed=seed)
    return env, obs, info


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, (bool, int, float, str)):
        return obj
    return None  # drop sapien.Pose etc.


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/info")
def info():
    return {"status": "ready", "env": _args.env_name if _args else None, "step": _episode_steps}


@app.post("/reset")
async def reset(request: Request):
    global _env, _episode_steps, _max_steps

    body = unpack(await request.body()) if request.headers.get("content-type") == "application/msgpack" else {}
    env_name = body.get("env_name", _args.env_name)
    seed = body.get("seed", _args.seed)
    max_episode_steps = body.get("max_episode_steps", _args.max_episode_steps)
    n_action_steps = body.get("n_action_steps", _args.n_action_steps)
    video_dir = body.get("video_dir", _args.video_dir)

    _max_steps = max_episode_steps

    if _env is None or env_name != _args.env_name:
        _env, obs, info = _build_env(env_name, seed, max_episode_steps, n_action_steps, video_dir)
        _args.env_name = env_name
    else:
        obs, info = _env.reset(seed=seed)

    _episode_steps = 0

    return msgpack_response({
        "obs": _to_serializable(obs),
        "done": False,
        "success": False,
        "episode_steps": 0,
    })


@app.post("/step")
async def step(request: Request):
    global _env, _episode_steps

    if _env is None:
        return msgpack_response({"error": "call /reset first"})

    body = unpack(await request.body())
    action = body["action"]

    obs, reward, done, truncated, info = _env.step(action)
    _episode_steps += 1

    raw_success = info.get("success", False)
    success = bool(np.any(raw_success)) if isinstance(raw_success, np.ndarray) else bool(raw_success)
    terminated = done or truncated or _episode_steps >= _max_steps

    return msgpack_response({
        "obs": _to_serializable(obs),
        "reward": float(reward) if not isinstance(reward, dict) else 0.0,
        "done": bool(terminated),
        "success": success,
        "episode_steps": _episode_steps,
    })


def _batch_obs(obs: dict) -> dict:
    """Add batch dim (B=1) to obs arrays so Gr00tSimPolicyWrapper sees (B, T, H, W, C)."""
    batched = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            batched[k] = v[np.newaxis]   # (T,...) → (1, T, ...)
        elif isinstance(v, str):
            batched[k] = (v,)            # str → tuple[str] with B=1
        else:
            batched[k] = v
    return batched


@app.post("/predict")
async def predict(request: Request):
    """Forward obs to gr00t ZMQ policy server, return action for MLX delta comparison."""
    if _policy_client is None:
        return msgpack_response({"error": "no policy server — start with --policy_host/--policy_port"})

    body = unpack(await request.body())
    obs = _batch_obs(body["obs"])

    action = _policy_client.get_action(obs)
    return msgpack_response({"action": _to_serializable(action)})


# ── main ──────────────────────────────────────────────────────────────────────
_args = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="simpler_env_google/google_robot_pick_coke_can")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--policy_host", default="localhost",
                        help="Host of the gr00t ZMQ policy server (default: localhost)")
    parser.add_argument("--policy_port", type=int, default=5556,
                        help="Port of the gr00t ZMQ policy server (default: 5556). "
                             "Enables /predict endpoint for MLX vs CUDA action delta comparison.")
    parser.add_argument("--no_policy", action="store_true",
                        help="Skip connecting to policy server (sim-only mode)")
    _args = parser.parse_args()

    print(f"[sim_server] env={_args.env_name}  port={_args.port}  seed={_args.seed}")
    if not _args.no_policy:
        _connect_policy_client(_args.policy_host, _args.policy_port)
    uvicorn.run(app, host="0.0.0.0", port=_args.port)
