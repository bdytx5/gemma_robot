"""
sim_server.py — runs on H100

Wraps SimplerEnv in a FastAPI HTTP server so a remote client (local machine)
can drive the eval loop over ngrok.

Usage:
    cd Isaac-GR00T
    python remote_eval/sim_server.py --env_name simpler_env_google/google_robot_pick_coke_can \
        --port 8080 --seed 42 --max_episode_steps 504 --n_action_steps 8

Then expose via ngrok:
    ngrok http 8080
"""

import argparse
import io
import sys
import os
import threading
import time
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = REPO_ROOT / "external_dependencies/SimplerEnv"
MANISKILL = EXTERNAL / "ManiSkill2_real2sim"

for p in [str(EXTERNAL), str(MANISKILL), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

# ── serialization (reuse MsgSerializer pattern from server_client.py) ─────────
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
_latest_frame = None        # current frame as BGR numpy array for MJPEG stream
_episode_frames = []        # frames collected this episode for Weave logging
_episode_count = 0
_weave_inited = False

def _init_weave():
    global _weave_inited
    if not _weave_inited:
        try:
            import weave
            weave.init("gr00t-remote-eval")
            _weave_inited = True
        except Exception as e:
            print(f"[weave] init failed: {e}")


def _build_env(env_name: str, seed: int, max_episode_steps: int, n_action_steps: int,
               video_dir: Optional[str]):
    import gymnasium as gym
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    from gr00t.eval.rollout_policy import create_eval_env, WrapperConfigs
    from gr00t.eval.rollout_policy import VideoConfig, MultiStepConfig
    from functools import partial

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


def _extract_frame(obs) -> Optional[np.ndarray]:
    """Pull the image frame out of obs dict for streaming/logging."""
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim >= 3:
                arr = v
                while arr.ndim > 3:
                    arr = arr[-1]
                return arr.astype(np.uint8)
            result = _extract_frame(v)
            if result is not None:
                return result
    return None

def _obs_to_serializable(obs):
    """Recursively convert obs dict to msgpack-serializable form."""
    if isinstance(obs, dict):
        return {k: _obs_to_serializable(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return obs
    if isinstance(obs, (list, tuple)):
        return [_obs_to_serializable(x) for x in obs]
    if isinstance(obs, (bool, int, float, str)):
        return obs
    return None


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/info")
def info():
    return {"status": "ready", "env": str(_env), "episode": _episode_count, "step": _episode_steps}


def _mjpeg_generator():
    while True:
        frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        # encode as JPEG
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1/15)  # ~15fps


@app.get("/stream")
def stream():
    """MJPEG stream — open in browser to watch live."""
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def _log_episode_to_weave(frames, success, episode_idx, env_name, seed):
    """Save episode frames as video and log to Weave."""
    try:
        import weave
        if not frames:
            return
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        h, w = frames[0].shape[:2]
        w = w - (w % 2); h = h - (h % 2)
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for fr in frames:
            bgr = cv2.cvtColor(fr[:h, :w], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        weave.log({
            "episode": episode_idx,
            "env_name": env_name,
            "seed": seed,
            "success": success,
            "n_steps": len(frames),
            "video": weave.Video(tmp_path),
        })
        print(f"[weave] logged episode {episode_idx} success={success} frames={len(frames)}")
    except Exception as e:
        print(f"[weave] logging failed: {e}")


@app.post("/reset")
async def reset(request: Request):
    global _env, _episode_steps, _max_steps, _latest_frame, _episode_frames, _episode_count

    body = unpack(await request.body()) if request.headers.get("content-type") == "application/msgpack" else {}
    env_name = body.get("env_name", _args.env_name)
    seed = body.get("seed", _args.seed)
    max_episode_steps = body.get("max_episode_steps", _args.max_episode_steps)
    n_action_steps = body.get("n_action_steps", _args.n_action_steps)
    video_dir = body.get("video_dir", _args.video_dir)

    _max_steps = max_episode_steps
    _episode_frames = []
    _episode_count += 1
    _init_weave()

    if _env is None or env_name != _args.env_name:
        _env, obs, info = _build_env(env_name, seed, max_episode_steps, n_action_steps, video_dir)
        _args.env_name = env_name
    else:
        obs, info = _env.reset(seed=seed)

    _episode_steps = 0
    frame = _extract_frame(obs)
    if frame is not None:
        _latest_frame = frame
        _episode_frames.append(frame.copy())

    payload = {
        "obs": _obs_to_serializable(obs),
        "done": False,
        "success": False,
        "episode_steps": 0,
    }
    return msgpack_response(payload)


@app.post("/step")
async def step(request: Request):
    global _env, _episode_steps

    if _env is None:
        return msgpack_response({"error": "call /reset first"})

    body = unpack(await request.body())
    action = body["action"]  # dict of action arrays

    obs, reward, done, truncated, info = _env.step(action)
    _episode_steps += 1

    frame = _extract_frame(obs)
    if frame is not None:
        _latest_frame = frame
        _episode_frames.append(frame.copy())

    raw_success = info.get("success", False)
    success = bool(np.any(raw_success)) if isinstance(raw_success, np.ndarray) else bool(raw_success)
    terminated = done or truncated or _episode_steps >= _max_steps

    # log completed episode to Weave in background thread
    if terminated:
        frames_copy = list(_episode_frames)
        ep_idx = _episode_count
        ep_seed = _args.seed
        env_name = _args.env_name
        threading.Thread(
            target=_log_episode_to_weave,
            args=(frames_copy, success, ep_idx, env_name, ep_seed),
            daemon=True,
        ).start()

    payload = {
        "obs": _obs_to_serializable(obs),
        "reward": float(reward) if not isinstance(reward, dict) else 0.0,
        "done": bool(terminated),
        "success": success,
        "truncated": bool(truncated),
        "episode_steps": _episode_steps,
    }
    return msgpack_response(payload)


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
    _args = parser.parse_args()

    print(f"[sim_server] env={_args.env_name}  port={_args.port}  seed={_args.seed}")
    uvicorn.run(app, host="0.0.0.0", port=_args.port)
