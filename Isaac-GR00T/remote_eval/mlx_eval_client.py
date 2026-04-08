"""
mlx_eval_client.py — runs on your LOCAL Mac

Connects to sim_server.py running on GPU instance (exposed via ngrok),
loads GemmaVLA (native MLX) locally, drives the eval loop.

Features:
  - Live cv2 window showing the robot's view in real time
  - W&B Weave tracing: per-step inference logged as @weave.op,
    per-episode summary + video logged at episode end

Usage:
    python remote_eval/mlx_eval_client.py \
        --server_url https://corridored-gruffly-myesha.ngrok-free.dev \
        --weave_project gemma-robot-eval \
        --n_episodes 5 \
        --seed 42
"""

import argparse
import io
import sys
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import requests
import weave
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent          # Isaac-GR00T/
MLX_DIR   = REPO_ROOT.parent / "mlx_gr00t"                 # mlx_gr00t/

if str(MLX_DIR) not in sys.path:
    sys.path.insert(0, str(MLX_DIR))


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

def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)

HEADERS = {"Content-Type": "application/msgpack"}


# ── sim server client ──────────────────────────────────────────────────────────
class SimClient:
    def __init__(self, server_url: str, timeout: int = 120):
        self.base = server_url.rstrip("/")
        self.timeout = timeout

    def ping(self):
        return requests.get(f"{self.base}/info", timeout=10).json()

    def reset(self, env_name=None, seed=None, max_episode_steps=None,
              n_action_steps=None, video_dir=None):
        body = {}
        if env_name:              body["env_name"] = env_name
        if seed is not None:      body["seed"] = seed
        if max_episode_steps:     body["max_episode_steps"] = max_episode_steps
        if n_action_steps:        body["n_action_steps"] = n_action_steps
        if video_dir:             body["video_dir"] = video_dir
        r = requests.post(f"{self.base}/reset", data=pack(body),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def step(self, action: dict):
        r = requests.post(f"{self.base}/step", data=pack({"action": action}),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def predict(self, obs: dict):
        """Get CUDA model action from server for delta comparison."""
        r = requests.post(f"{self.base}/predict", data=pack({"obs": obs}),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)


# ── obs parsing ───────────────────────────────────────────────────────────────
STATE_KEYS = ["state.x", "state.y", "state.z",
              "state.rx", "state.ry", "state.rz", "state.rw",
              "state.gripper"]

def obs_to_inputs(obs: dict):
    """Extract PIL image, robot_state (8D float32), instruction, and raw frame array."""
    img_arr = obs["video.image"]
    while img_arr.ndim > 3:
        img_arr = img_arr[-1]
    image = Image.fromarray(img_arr.astype(np.uint8))

    state_parts = []
    for k in STATE_KEYS:
        v = np.asarray(obs[k]).flatten()
        state_parts.append(v[-1])
    robot_state = np.array(state_parts, dtype=np.float32)

    instruction = obs.get("annotation.human.action.task_description", "pick up the object")
    if isinstance(instruction, (list, np.ndarray)):
        instruction = str(instruction[-1])

    return image, robot_state, instruction, img_arr.astype(np.uint8)


# ── action formatting ─────────────────────────────────────────────────────────
ACTION_KEYS = ["action.x", "action.y", "action.z",
               "action.roll", "action.pitch", "action.yaw",
               "action.gripper"]

def format_action(actions: np.ndarray, n_action_steps: int) -> dict:
    acts = actions[0, :n_action_steps]         # (n_action_steps, 7)
    return {key: acts[:, i:i+1] for i, key in enumerate(ACTION_KEYS)}


# ── episode video → weave ──────────────────────────────────────────────────────
@weave.op
def log_episode(ep: int, seed: int, instruction: str, success: bool,
                steps: int, elapsed_s: float, video_path: str):
    """Log episode summary + video to Weave. Returns VideoFileClip so Weave uploads it."""
    try:
        from moviepy.editor import VideoFileClip  # moviepy 1.x
    except ImportError:
        from moviepy import VideoFileClip          # moviepy 2.x
    return {
        "episode": ep,
        "seed": seed,
        "instruction": instruction,
        "success": success,
        "steps": steps,
        "elapsed_s": round(elapsed_s, 2),
        "video": VideoFileClip(video_path),
    }


# ── eval loop ─────────────────────────────────────────────────────────────────
def run_eval(
    server_url: str,
    gr00t_ckpt: str,
    checkpoint: str | None,
    eagle_repo: str,
    mlx_llm_path: str,
    hf_token: str | None,
    env_name: str,
    n_episodes: int,
    seed: int,
    max_episode_steps: int,
    n_action_steps: int,
    n_diffusion_steps: int,
    video_dir: str | None,
    weave_project: str,
    video_fps: int,
    cuda_delta: bool,
):
    weave.init(weave_project)

    client = SimClient(server_url)
    print(f"[mlx_eval] Pinging {server_url} ...")
    info = client.ping()
    print(f"[mlx_eval] Server: {info}")

    from gemma_vla import GemmaVLA
    hf_tok = hf_token or True
    print(f"[mlx_eval] Loading GemmaVLA (n_diffusion_steps={n_diffusion_steps}) ...")
    vla = GemmaVLA.from_pretrained(
        gr00t_repo=gr00t_ckpt,
        checkpoint=checkpoint,
        eagle_repo=eagle_repo,
        mlx_llm_path=mlx_llm_path,
        hf_token=hf_tok,
        n_diffusion_steps=n_diffusion_steps,
    )
    print(vla)

    cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot View", 640, 480)

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
        done = False
        success = False
        step_count = 0
        t0 = time.time()
        ep_frames = []
        instruction = "pick up the object"

        while not done:
            image, robot_state, instruction, frame_rgb = obs_to_inputs(obs)

            # Live display (BGR for cv2)
            ep_frames.append(frame_rgb)
            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f"ep {ep+1}  step {step_count}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Robot View", display)
            cv2.waitKey(1)

            # Print instruction on first step of each episode
            if step_count == 0:
                print(f"\n  [prompt] \"{instruction}\"")
                print(f"  [state]  {robot_state}")

            t_infer = time.time()
            raw_actions = vla.get_action(image=image, robot_state=robot_state,
                                         instruction=instruction)
            dt_infer = (time.time() - t_infer) * 1000

            # Optionally fetch CUDA action from server and print deltas
            cuda_acts = None
            if cuda_delta:
                try:
                    cuda_result = client.predict(obs)
                    if "action" in cuda_result:
                        cuda_act_dict = cuda_result["action"]
                        cuda_acts = np.stack(
                            [np.asarray(cuda_act_dict[k]).flatten() for k in ACTION_KEYS], axis=-1
                        )  # (n_action_steps, 7)
                except Exception as e:
                    print(f"\n  [cuda_delta] error: {e}")

            # Print actions every 8 steps so we can sanity-check
            if step_count % 8 == 0:
                acts = raw_actions[0, :n_action_steps]   # (n_action_steps, 7)
                labels = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
                print(f"\n  [actions @ step {step_count}]  shape={acts.shape}")
                for i, lbl in enumerate(labels):
                    print(f"    {lbl:5s}: {acts[:, i]}")
                if cuda_acts is not None:
                    mlx_7 = acts[:, :7]   # MLX outputs 128 dims; first 7 are the actions
                    delta = mlx_7 - cuda_acts[:n_action_steps]
                    print(f"\n  [cuda_delta @ step {step_count}]  (mlx - cuda)")
                    for i, lbl in enumerate(labels):
                        mae = np.abs(delta[:, i]).mean()
                        print(f"    {lbl:5s}: {delta[:, i]}  mae={mae:.4f}")

            action = format_action(raw_actions, n_action_steps)
            result = client.step(action)

            obs        = result["obs"]
            done       = result["done"]
            success    = result["success"]
            step_count = result["episode_steps"]

            print(f"  step {step_count:3d}  infer={dt_infer:.0f}ms  success={success}", end="\r")

        elapsed = time.time() - t0
        successes.append(success)
        ep_steps_list.append(step_count)
        print(f"\n  → done in {step_count} steps ({elapsed:.1f}s) | success={success}")

        # Save episode video to disk, then log to Weave
        try:
            from moviepy.editor import ImageSequenceClip  # moviepy 1.x
        except ImportError:
            from moviepy import ImageSequenceClip          # moviepy 2.x
        video_path = f"/tmp/gr00t_ep{ep+1}_seed{ep_seed}.mp4"
        ImageSequenceClip(list(ep_frames), fps=video_fps).write_videofile(
            video_path, logger=None
        )
        log_episode(
            ep=ep + 1,
            seed=ep_seed,
            instruction=instruction,
            success=success,
            steps=step_count,
            elapsed_s=elapsed,
            video_path=video_path,
        )

    cv2.destroyAllWindows()

    success_rate = sum(successes) / len(successes)
    print(f"\n{'='*50}")
    print(f"  Results: {sum(successes)}/{n_episodes} ({success_rate*100:.1f}%)")
    print(f"  Avg steps: {sum(ep_steps_list)/len(ep_steps_list):.1f}")
    print(f"{'='*50}")
    return successes


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEFAULT_MLX_LLM = str(MLX_DIR / "gr00t_llm_mlx")

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", required=True,
                        help="ngrok URL, e.g. https://abc123.ngrok-free.app")
    parser.add_argument("--gr00t_ckpt", default="youngbrett48/gr00t-post-train-fractal-270m")
    parser.add_argument("--checkpoint", default="checkpoint-2000")
    parser.add_argument("--eagle_repo", default="youngbrett48/train_stage2_gemma3_270m.sh")
    parser.add_argument("--mlx_llm_path", default=DEFAULT_MLX_LLM)
    parser.add_argument("--hf_token", default=None)
    #simpler_env_google/google_robot_pick_coke_can
    parser.add_argument("--env_name", default="simpler_env_google/google_robot_pick_coke_can")
    # parser.add_argument("--env_name", default="simper_env_google/googlel_robot_close_drawer")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--n_diffusion_steps", type=int, default=4)
    parser.add_argument("--video_dir", default=None,
                        help="Server-side path for video saving (H100 path)")
    parser.add_argument("--weave_project", default="gemma-robot-eval",
                        help="W&B Weave project name")
    parser.add_argument("--video_fps", type=int, default=10,
                        help="FPS for episode videos logged to Weave")
    parser.add_argument("--cuda_delta", action="store_true",
                        help="Call /predict on server each step to compare CUDA vs MLX actions")
    args = parser.parse_args()

    run_eval(
        server_url=args.server_url,
        gr00t_ckpt=args.gr00t_ckpt,
        checkpoint=args.checkpoint,
        eagle_repo=args.eagle_repo,
        mlx_llm_path=args.mlx_llm_path,
        hf_token=args.hf_token,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        n_diffusion_steps=args.n_diffusion_steps,
        video_dir=args.video_dir,
        weave_project=args.weave_project,
        video_fps=args.video_fps,
        cuda_delta=args.cuda_delta,
    )
