"""
GemmaRobot Eval — macOS GUI app

Connects to a sim_server running on GPU via ngrok,
loads GemmaVLA locally on Apple Silicon, and runs the eval loop
with a live video feed and action display.

Run:
    python app.py
"""

import sys
import os
import threading
import queue
import time
import io
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, font as tkfont
import msgpack
import numpy as np
import requests

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ISAAC = HERE.parent / "Isaac-GR00T"
EAGLE = HERE.parent / "Eagle" / "Eagle2_5"
for p in [str(HERE), str(ISAAC), str(EAGLE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── constants ─────────────────────────────────────────────────────────────────
BG       = "#1a1a1a"
BG2      = "#242424"
BG3      = "#2e2e2e"
ACCENT   = "#00d4aa"
TEXT     = "#e8e8e8"
TEXT_DIM = "#888888"
RED      = "#ff5555"
GREEN    = "#50fa7b"
YELLOW   = "#f1fa8c"
FONT     = "SF Pro Display"

ACTION_KEYS = ["action.x", "action.y", "action.z",
               "action.roll", "action.pitch", "action.yaw",
               "action.gripper"]

STATE_KEYS = ["state.x", "state.y", "state.z",
              "state.rx", "state.ry", "state.rz", "state.rw",
              "state.gripper"]

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

def pack(data):
    return msgpack.packb(data, default=_encode, use_bin_type=True)

def unpack(data):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)

HEADERS = {"Content-Type": "application/msgpack"}


# ── sim client ────────────────────────────────────────────────────────────────
class SimClient:
    def __init__(self, url, timeout=120):
        self.base = url.rstrip("/")
        self.timeout = timeout

    def ping(self):
        return requests.get(f"{self.base}/info", timeout=8).json()

    def reset(self, env_name=None, seed=None, max_episode_steps=None,
              n_action_steps=None):
        body = {}
        if env_name:          body["env_name"] = env_name
        if seed is not None:  body["seed"] = seed
        if max_episode_steps: body["max_episode_steps"] = max_episode_steps
        if n_action_steps:    body["n_action_steps"] = n_action_steps
        r = requests.post(f"{self.base}/reset", data=pack(body),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def step(self, action):
        r = requests.post(f"{self.base}/step",
                          data=pack({"action": action}),
                          headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)


# ── obs helpers ───────────────────────────────────────────────────────────────
def obs_to_inputs(obs):
    from PIL import Image as PILImage
    img_arr = obs["video.image"]
    while img_arr.ndim > 3:
        img_arr = img_arr[-1]
    image = PILImage.fromarray(img_arr.astype(np.uint8))

    state_parts = []
    for k in STATE_KEYS:
        v = np.asarray(obs[k]).flatten()
        state_parts.append(v[-1])
    robot_state = np.array(state_parts, dtype=np.float32)

    instruction = obs.get("annotation.human.action.task_description",
                          "pick up the object")
    if isinstance(instruction, (list, np.ndarray)):
        instruction = str(instruction[-1])

    return image, robot_state, instruction, img_arr


def format_action(actions, n_action_steps):
    acts = actions[0][:n_action_steps]
    return {key: acts[:, i:i+1] for i, key in enumerate(ACTION_KEYS)}


# ── main app ──────────────────────────────────────────────────────────────────
class GemmaRobotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GemmaRobot Eval")
        self.root.configure(bg=BG)
        self.root.geometry("960x720")
        self.root.minsize(860, 640)

        self._q = queue.Queue()
        self._stop_event = threading.Event()
        self._vla = None
        self._running = False

        self._build_ui()
        self.root.after(50, self._drain_queue)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ---- top bar ----
        top = tk.Frame(self.root, bg=BG, pady=10, padx=16)
        top.pack(fill="x")

        tk.Label(top, text="GemmaRobot", bg=BG, fg=ACCENT,
                 font=(FONT, 18, "bold")).pack(side="left")
        tk.Label(top, text="  MLX · Apple Silicon", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 12)).pack(side="left")

        # status dot
        self._status_dot = tk.Label(top, text="●", bg=BG, fg=RED,
                                     font=(FONT, 14))
        self._status_dot.pack(side="right", padx=(0, 4))
        self._status_lbl = tk.Label(top, text="Disconnected", bg=BG,
                                     fg=TEXT_DIM, font=(FONT, 11))
        self._status_lbl.pack(side="right")

        sep = tk.Frame(self.root, bg=BG3, height=1)
        sep.pack(fill="x")

        # ---- config row ----
        cfg = tk.Frame(self.root, bg=BG2, pady=10, padx=16)
        cfg.pack(fill="x")

        tk.Label(cfg, text="Server URL", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._url_var = tk.StringVar(value="https://")
        url_entry = tk.Entry(cfg, textvariable=self._url_var, bg=BG3,
                             fg=TEXT, insertbackground=TEXT, relief="flat",
                             font=(FONT, 12), width=42)
        url_entry.grid(row=0, column=1, padx=(0, 12))

        tk.Label(cfg, text="Episodes", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=2, sticky="w", padx=(0, 6))
        self._ep_var = tk.StringVar(value="5")
        tk.Entry(cfg, textvariable=self._ep_var, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=(FONT, 12), width=4).grid(row=0, column=3, padx=(0, 12))

        tk.Label(cfg, text="Seed", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=4, sticky="w", padx=(0, 6))
        self._seed_var = tk.StringVar(value="42")
        tk.Entry(cfg, textvariable=self._seed_var, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=(FONT, 12), width=6).grid(row=0, column=5, padx=(0, 16))

        self._run_btn = tk.Button(cfg, text="▶  Run Eval", bg=ACCENT,
                                   fg="#000000", font=(FONT, 12, "bold"),
                                   relief="flat", padx=16, pady=4,
                                   cursor="hand2",
                                   command=self._on_run_stop)
        self._run_btn.grid(row=0, column=6, padx=(0, 8))

        self._setup_btn = tk.Button(cfg, text="⚙  Setup Env", bg=BG3,
                                     fg=TEXT, font=(FONT, 11),
                                     relief="flat", padx=12, pady=4,
                                     cursor="hand2",
                                     command=self._on_setup_env)
        self._setup_btn.grid(row=0, column=7)

        sep2 = tk.Frame(self.root, bg=BG3, height=1)
        sep2.pack(fill="x")

        # ---- main content ----
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=16, pady=12)

        # left: video
        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="LIVE FEED", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w")

        self._video_lbl = tk.Label(left, bg="#000000",
                                    width=480, height=360)
        self._video_lbl.pack(pady=(4, 12))

        # episode stats bar
        stats_bar = tk.Frame(left, bg=BG2, pady=8, padx=12)
        stats_bar.pack(fill="x")

        self._ep_lbl   = self._stat(stats_bar, "Episode", "—", 0)
        self._step_lbl = self._stat(stats_bar, "Step",    "—", 1)
        self._sr_lbl   = self._stat(stats_bar, "Success", "—", 2)
        self._inf_lbl  = self._stat(stats_bar, "Infer",   "—", 3)

        # right: actions + log
        right = tk.Frame(content, bg=BG, padx=(16, 0))
        right.pack(side="left", fill="both")

        tk.Label(right, text="PREDICTED ACTIONS", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w")

        self._action_frame = tk.Frame(right, bg=BG2, pady=8, padx=12)
        self._action_frame.pack(fill="x", pady=(4, 0))

        self._action_bars = {}
        for i, k in enumerate(["x", "y", "z", "roll", "pitch", "yaw", "grip"]):
            self._action_bars[k] = self._action_row(self._action_frame, k, i)

        tk.Label(right, text="LOG", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w", pady=(16, 0))

        log_frame = tk.Frame(right, bg=BG2)
        log_frame.pack(fill="both", expand=True, pady=(4, 0))

        self._log = tk.Text(log_frame, bg=BG2, fg=TEXT, font=("SF Mono", 10),
                             relief="flat", state="disabled", wrap="word",
                             width=34, height=14)
        self._log.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        sb = ttk.Scrollbar(log_frame, command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.config(yscrollcommand=sb.set)

        self._log.tag_config("ok",  foreground=GREEN)
        self._log.tag_config("err", foreground=RED)
        self._log.tag_config("dim", foreground=TEXT_DIM)
        self._log.tag_config("hi",  foreground=ACCENT)

    def _stat(self, parent, label, value, col):
        f = tk.Frame(parent, bg=BG2)
        f.grid(row=0, column=col, padx=16)
        tk.Label(f, text=label.upper(), bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 8, "bold")).pack()
        lbl = tk.Label(f, text=value, bg=BG2, fg=TEXT,
                        font=(FONT, 16, "bold"))
        lbl.pack()
        return lbl

    def _action_row(self, parent, name, row):
        tk.Label(parent, text=name.upper(), bg=BG2, fg=TEXT_DIM,
                 font=("SF Mono", 10), width=5, anchor="w"
                 ).grid(row=row, column=0, sticky="w", pady=2)
        val_lbl = tk.Label(parent, text="  0.0000", bg=BG2, fg=TEXT,
                            font=("SF Mono", 11), width=9, anchor="e")
        val_lbl.grid(row=row, column=1, padx=(4, 8))

        canvas = tk.Canvas(parent, bg=BG3, height=12, width=160,
                            highlightthickness=0)
        canvas.grid(row=row, column=2, sticky="ew", pady=2)
        return {"val": val_lbl, "canvas": canvas}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log_msg(self, msg, tag=""):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _set_status(self, text, color):
        self._status_lbl.config(text=text)
        self._status_dot.config(fg=color)

    def _update_video(self, img_arr):
        from PIL import Image as PILImage, ImageTk
        img = PILImage.fromarray(img_arr.astype(np.uint8))
        img = img.resize((480, 360), PILImage.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._video_lbl.config(image=photo, width=480, height=360)
        self._video_lbl._photo = photo   # prevent GC

    def _update_actions(self, actions):
        # actions: (1, horizon, 7+)
        a = actions[0, 0, :7]
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
        for i, k in enumerate(keys):
            v = float(a[i])
            self._action_bars[k]["val"].config(text=f"{v:+.4f}")
            # bar: map [-0.3, 0.3] → width
            c = self._action_bars[k]["canvas"]
            c.delete("all")
            mid = 80
            bar_w = min(abs(v) / 0.3 * 75, 75)
            color = ACCENT if v >= 0 else "#ff79c6"
            x0 = mid if v >= 0 else mid - bar_w
            c.create_rectangle(x0, 2, x0 + bar_w, 10, fill=color,
                                outline="")

    # ── setup env ─────────────────────────────────────────────────────────────

    def _on_setup_env(self):
        self._setup_btn.config(state="disabled", text="⚙  Setting up…")
        t = threading.Thread(target=self._setup_env_thread, daemon=True)
        t.start()

    def _setup_env_thread(self):
        import subprocess
        def q(fn, *a, **kw): self._q.put((fn, a, kw))

        def done(ok):
            color = ACCENT if ok else BG3
            q(self._setup_btn.config, state="normal",
              text="✓  Env Ready" if ok else "⚙  Setup Env",
              bg=color, fg="#000000" if ok else TEXT)

        q(self._log_msg, "\n── Setting up conda env ──", "hi")
        q(self._set_status, "Setting up…", YELLOW)

        # find conda
        conda = None
        for c in [
            os.path.expanduser("~/miniconda3/bin/conda"),
            os.path.expanduser("~/anaconda3/bin/conda"),
            os.path.expanduser("~/miniforge3/bin/conda"),
            os.path.expanduser("~/opt/miniconda3/bin/conda"),
            "/opt/homebrew/bin/conda",
        ]:
            if os.path.isfile(c):
                conda = c
                break

        if not conda:
            # try PATH
            r = subprocess.run(["which", "conda"], capture_output=True, text=True)
            if r.returncode == 0:
                conda = r.stdout.strip()

        if not conda:
            q(self._log_msg,
              "conda not found.\nInstall Miniconda: https://docs.conda.io/en/latest/miniconda.html",
              "err")
            q(self._set_status, "Error", RED)
            done(False)
            return

        q(self._log_msg, f"Found conda: {conda}", "dim")

        # check if env exists
        env_name = "mlx_robot"
        r = subprocess.run([conda, "env", "list"], capture_output=True, text=True)
        env_exists = env_name in r.stdout

        if env_exists:
            q(self._log_msg, f"Env '{env_name}' already exists.", "ok")
        else:
            q(self._log_msg, f"Creating env '{env_name}' (Python 3.10)…", "dim")
            r = subprocess.run(
                [conda, "create", "-y", "-n", env_name, "python=3.10"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                q(self._log_msg, f"Failed:\n{r.stderr[-400:]}", "err")
                done(False)
                return
            q(self._log_msg, "  Created.", "ok")

        # find pip in env
        pip = None
        for p in [
            os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/pip"),
            os.path.expanduser(f"~/anaconda3/envs/{env_name}/bin/pip"),
            os.path.expanduser(f"~/miniforge3/envs/{env_name}/bin/pip"),
            os.path.expanduser(f"~/opt/miniconda3/envs/{env_name}/bin/pip"),
        ]:
            if os.path.isfile(p):
                pip = p
                break

        if not pip:
            q(self._log_msg, f"Can't find pip in env '{env_name}'", "err")
            done(False)
            return

        packages = [
            "mlx", "mlx-lm",
            "transformers==4.49.0",
            "safetensors", "huggingface_hub",
            "Pillow", "numpy", "requests", "msgpack",
            "tokenizers",
        ]

        q(self._log_msg, f"Installing {len(packages)} packages…", "dim")
        r = subprocess.run(
            [pip, "install", "--upgrade"] + packages,
            capture_output=True, text=True
        )
        if r.returncode != 0:
            q(self._log_msg, f"Install failed:\n{r.stderr[-400:]}", "err")
            done(False)
            return

        q(self._log_msg, "All packages installed.", "ok")
        q(self._log_msg, f"\nEnv '{env_name}' is ready.\nRelaunch the app using:\n  conda activate {env_name} && python app.py",
          "hi")
        q(self._set_status, "Env Ready", GREEN)
        done(True)

    # ── run/stop ──────────────────────────────────────────────────────────────

    def _on_run_stop(self):
        if self._running:
            self._stop_event.set()
            self._run_btn.config(text="▶  Run Eval", bg=ACCENT)
            self._running = False
        else:
            self._stop_event.clear()
            self._running = True
            self._run_btn.config(text="■  Stop", bg=RED)
            t = threading.Thread(target=self._eval_thread, daemon=True)
            t.start()

    # ── eval thread ───────────────────────────────────────────────────────────

    def _eval_thread(self):
        def q(fn, *a, **kw): self._q.put((fn, a, kw))

        url    = self._url_var.get().strip()
        n_eps  = int(self._ep_var.get() or 5)
        seed   = int(self._seed_var.get() or 42)
        n_steps = 8

        q(self._log_msg, f"Connecting to {url} …", "dim")
        q(self._set_status, "Connecting…", YELLOW)

        # ping
        try:
            client = SimClient(url)
            info = client.ping()
            q(self._log_msg, f"Server: {info}", "dim")
            q(self._set_status, "Connected", GREEN)
        except Exception as e:
            q(self._log_msg, f"Connection failed: {e}", "err")
            q(self._set_status, "Disconnected", RED)
            q(self._run_btn.config, text="▶  Run Eval", bg=ACCENT)
            self._running = False
            return

        # load model (once)
        if self._vla is None:
            q(self._log_msg, "Loading GemmaVLA…", "dim")
            q(self._set_status, "Loading model…", YELLOW)
            try:
                from gemma_vla import GemmaVLA

                # prefer bundled weights (inside .app Resources or sibling dirs)
                weights_candidates = [
                    HERE / "gr00t_weights_mlx",
                    HERE.parent / "gr00t_weights_mlx",
                ]
                llm_candidates = [
                    HERE / "gr00t_llm_mlx",
                    HERE.parent / "gr00t_llm_mlx",
                ]

                weights_dir = next((p for p in weights_candidates if (p / "meta.json").exists()), None)
                llm_dir     = next((p for p in llm_candidates if p.exists() and any(p.iterdir())), None)

                if weights_dir and llm_dir:
                    q(self._log_msg, f"Using bundled weights", "dim")
                    self._vla = GemmaVLA.from_exported(
                        weights_dir=str(weights_dir),
                        mlx_llm_path=str(llm_dir),
                        n_diffusion_steps=4,
                    )
                else:
                    q(self._log_msg, "Bundled weights not found, downloading from HF…", "dim")
                    self._vla = GemmaVLA.from_pretrained(
                        gr00t_repo="youngbrett48/gr00t-post-train-fractal-270m",
                        checkpoint="checkpoint-2000",
                        eagle_repo="youngbrett48/train_stage2_gemma3_270m.sh",
                        mlx_llm_path=str(HERE / "gr00t_llm_mlx"),
                        hf_token=True,
                        n_diffusion_steps=4,
                    )
                q(self._log_msg, "Model ready ✓", "ok")
                q(self._set_status, "Ready", GREEN)
            except Exception as e:
                q(self._log_msg, f"Model load failed: {e}", "err")
                q(self._set_status, "Error", RED)
                self._running = False
                return

        successes = []

        for ep in range(n_eps):
            if self._stop_event.is_set():
                break

            ep_seed = seed + ep * 1000
            q(self._log_msg, f"\n── ep {ep+1}/{n_eps}  seed={ep_seed}", "hi")
            q(self._ep_lbl.config, text=f"{ep+1}/{n_eps}")

            try:
                result = client.reset(
                    env_name="simpler_env_google/google_robot_pick_coke_can",
                    seed=ep_seed,
                    max_episode_steps=504,
                    n_action_steps=n_steps,
                )
            except Exception as e:
                q(self._log_msg, f"Reset failed: {e}", "err")
                break

            obs = result["obs"]
            done = False
            success = False
            step = 0

            while not done and not self._stop_event.is_set():
                image, robot_state, instruction, img_arr = obs_to_inputs(obs)

                q(self._update_video, img_arr)

                t0 = time.time()
                raw = self._vla.get_action(
                    image=image,
                    robot_state=robot_state,
                    instruction=instruction,
                )
                dt = time.time() - t0

                action = format_action(raw, n_steps)

                q(self._update_actions, raw)
                q(self._inf_lbl.config, text=f"{dt*1000:.0f}ms")

                try:
                    result = client.step(action)
                except Exception as e:
                    q(self._log_msg, f"Step failed: {e}", "err")
                    break

                obs     = result["obs"]
                done    = result["done"]
                success = result["success"]
                step    = result["episode_steps"]

                q(self._step_lbl.config, text=str(step))

            successes.append(success)
            sr = sum(successes) / len(successes)
            color = "ok" if success else "err"
            q(self._log_msg,
              f"  done  steps={step}  success={success}  sr={sr:.0%}", color)
            q(self._sr_lbl.config, text=f"{sr:.0%}")

        q(self._log_msg, "\nEval complete.", "hi")
        q(self._set_status, "Done", GREEN)
        q(self._run_btn.config, text="▶  Run Eval", bg=ACCENT)
        self._running = False

    # ── queue drain ───────────────────────────────────────────────────────────

    def _drain_queue(self):
        try:
            while True:
                fn, args, kwargs = self._q.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_queue)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 2.0)    # Retina
    except Exception:
        pass
    app = GemmaRobotApp(root)
    root.mainloop()
