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
import tempfile
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

# ── remote config ────────────────────────────────────────────────────────────
_CONFIG_DOC = "https://docs.google.com/document/d/1wF4OwIRHZAGgZDBgXPliWfG589QbeTSKM46SozpsxWI/export?format=txt"

def _fetch_server_url() -> str:
    try:
        r = requests.get(_CONFIG_DOC, timeout=5)
        url = r.text.strip().lstrip("\ufeff")   # strip BOM
        if url.startswith("http"):
            return url
    except Exception:
        pass
    return "https://"

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


# ── stdout/stderr → log panel redirect ───────────────────────────────────────
class _StreamRedirect:
    def __init__(self, app, tag):
        self._app = app
        self._tag = tag
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._app._log_msg(line, self._tag)

    def flush(self):
        if self._buf:
            self._app._log_msg(self._buf, self._tag)
            self._buf = ""

    def isatty(self):
        return False


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
        self._eval_thread_ref = None

        self._build_ui()
        self.root.after(50, self._drain_queue)

        # redirect stdout/stderr → log panel
        sys.stdout = _StreamRedirect(self, "")
        sys.stderr = _StreamRedirect(self, "err")

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
        threading.Thread(target=self._load_server_url, daemon=True).start()
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

        tk.Label(cfg, text="Diff steps", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=6, sticky="w", padx=(0, 6))
        self._diff_var = tk.StringVar(value="4")
        tk.Entry(cfg, textvariable=self._diff_var, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=(FONT, 12), width=3).grid(row=0, column=7, padx=(0, 16))

        tk.Label(cfg, text="Task", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        TASKS = [
            "simpler_env_google/google_robot_pick_coke_can",
            "simpler_env_google/google_robot_pick_object",
            "simpler_env_google/google_robot_move_near",
            "simpler_env_google/google_robot_open_drawer",
            "simpler_env_google/google_robot_close_drawer",
            "simpler_env_google/google_robot_place_in_closed_drawer",
        ]
        self._task_var = tk.StringVar(value=TASKS[0])
        task_menu = ttk.Combobox(cfg, textvariable=self._task_var, values=TASKS,
                                  font=(FONT, 11), width=52, state="readonly")
        task_menu.grid(row=1, column=1, columnspan=5, sticky="w", pady=(8, 0))

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

        # ---- server control row ----
        srv = tk.Frame(self.root, bg=BG2, pady=6, padx=16)
        srv.pack(fill="x")

        self._srv_status_dot = tk.Label(srv, text="●", bg=BG2, fg=TEXT_DIM,
                                         font=(FONT, 12))
        self._srv_status_dot.pack(side="left", padx=(0, 4))

        self._srv_status_var = tk.StringVar(value="Server: unknown")
        tk.Label(srv, textvariable=self._srv_status_var, bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 10)).pack(side="left", padx=(0, 16))

        self._check_btn = tk.Button(srv, text="⟳  Check Status", bg=BG3,
                                     fg=TEXT, font=(FONT, 10),
                                     relief="flat", padx=10, pady=3,
                                     cursor="hand2",
                                     command=self._on_check_status)
        self._check_btn.pack(side="left", padx=(0, 8))

        self._restart_btn = tk.Button(srv, text="↺  Restart Server", bg=BG3,
                                       fg=TEXT, font=(FONT, 10),
                                       relief="flat", padx=10, pady=3,
                                       cursor="hand2",
                                       command=self._on_restart_server)
        self._restart_btn.pack(side="left")

        sep2 = tk.Frame(self.root, bg=BG3, height=1)
        sep2.pack(fill="x")

        # ---- log panel — packed BEFORE content so it anchors to bottom ----
        sep3 = tk.Frame(self.root, bg=BG3, height=1)
        sep3.pack(side="bottom", fill="x")

        log_outer = tk.Frame(self.root, bg=BG, height=160)
        log_outer.pack(side="bottom", fill="x", padx=16, pady=(4, 6))
        log_outer.pack_propagate(False)

        tk.Label(log_outer, text="LOG", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w")

        log_frame = tk.Frame(log_outer, bg=BG2)
        log_frame.pack(fill="both", expand=True, pady=(2, 0))

        self._log = tk.Text(log_frame, bg=BG2, fg=TEXT, font=("SF Mono", 10),
                             relief="flat", state="disabled", wrap="word")
        self._log.pack(side="left", fill="both", expand=True, padx=8, pady=6)

        sb = ttk.Scrollbar(log_frame, command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.config(yscrollcommand=sb.set)

        self._log.tag_config("ok",  foreground=GREEN)
        self._log.tag_config("err", foreground=RED)
        self._log.tag_config("dim", foreground=TEXT_DIM)
        self._log.tag_config("hi",  foreground=ACCENT)

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
        self._video_lbl.pack(pady=(4, 4))

        self._instr_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self._instr_var, bg=BG, fg=TEXT_DIM,
                 font=(FONT, 10), wraplength=480, justify="left",
                 anchor="w").pack(fill="x", padx=4, pady=(0, 8))

        # episode stats bar
        stats_bar = tk.Frame(left, bg=BG2, pady=8, padx=12)
        stats_bar.pack(fill="x")

        self._ep_lbl   = self._stat(stats_bar, "Episode", "—", 0)
        self._step_lbl = self._stat(stats_bar, "Step",    "—", 1)
        self._sr_lbl   = self._stat(stats_bar, "Success", "—", 2)
        self._inf_lbl  = self._stat(stats_bar, "Infer",   "—", 3)

        # right: actions only
        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", padx=(16, 0))

        tk.Label(right, text="PREDICTED ACTIONS", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w")

        self._action_frame = tk.Frame(right, bg=BG2, pady=8, padx=12)
        self._action_frame.pack(fill="x", pady=(4, 0))

        self._action_bars = {}
        for i, k in enumerate(["x", "y", "z", "roll", "pitch", "yaw", "grip"]):
            self._action_bars[k] = self._action_row(self._action_frame, k, i)

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
        import re, webbrowser
        self._log.config(state="normal")
        url_pattern = re.compile(r'https?://\S+')
        last = 0
        for m in url_pattern.finditer(msg):
            if m.start() > last:
                self._log.insert("end", msg[last:m.start()], tag)
            url = m.group()
            link_tag = f"link_{id(url)}_{self._log.index('end')}"
            self._log.insert("end", url, (tag, link_tag))
            self._log.tag_config(link_tag, foreground=ACCENT,
                                  underline=True)
            self._log.tag_bind(link_tag, "<Button-1>",
                               lambda e, u=url: webbrowser.open(u))
            self._log.tag_bind(link_tag, "<Enter>",
                               lambda e: self._log.config(cursor="hand2"))
            self._log.tag_bind(link_tag, "<Leave>",
                               lambda e: self._log.config(cursor=""))
            last = m.end()
        if last < len(msg):
            self._log.insert("end", msg[last:], tag)
        self._log.insert("end", "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _load_server_url(self):
        url = _fetch_server_url()
        self._q.put((self._url_var.set, (url,), {}))

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

    # ── server status / restart ───────────────────────────────────────────────

    def _on_check_status(self):
        self._check_btn.config(state="disabled", text="⟳  Checking…")
        threading.Thread(target=self._check_status_thread, daemon=True).start()

    def _check_status_thread(self):
        def q(fn, *a, **kw): self._q.put((fn, a, kw))
        url = self._url_var.get().strip().rstrip("/")
        try:
            r = requests.get(f"{url}/control/status", timeout=8)
            r.raise_for_status()
            s = r.json()
            running = s.get("running", False)
            env = (s.get("env") or "").split("/")[-1] or "—"
            steps = s.get("episode_steps") or 0
            uptime = int(s.get("uptime_seconds") or 0)
            mins, secs = divmod(uptime, 60)
            sim_ready = s.get("sim_ready", False)

            if running and sim_ready:
                dot_color = GREEN
                status_text = f"Server: UP  |  env={env}  step={steps}  uptime={mins}m{secs:02d}s"
            elif running:
                dot_color = YELLOW
                status_text = f"Server: starting…  uptime={mins}m{secs:02d}s"
            else:
                dot_color = RED
                status_text = "Server: DOWN"

            q(self._srv_status_dot.config, fg=dot_color)
            q(self._srv_status_var.set, status_text)
            q(self._log_msg, f"[status] {status_text}", "dim")
        except Exception as e:
            q(self._srv_status_dot.config, fg=RED)
            q(self._srv_status_var.set, "Server: unreachable")
            q(self._log_msg, f"[status] {e}", "err")
        q(self._check_btn.config, state="normal", text="⟳  Check Status")

    def _on_restart_server(self):
        self._restart_btn.config(state="disabled", text="↺  Restarting…")
        self._srv_status_dot.config(fg=YELLOW)
        self._srv_status_var.set("Server: restarting…")
        threading.Thread(target=self._restart_server_thread, daemon=True).start()

    def _restart_server_thread(self):
        def q(fn, *a, **kw): self._q.put((fn, a, kw))
        url = self._url_var.get().strip().rstrip("/")
        q(self._log_msg, "[restart] Sending restart to control server…", "dim")
        try:
            r = requests.post(f"{url}/control/restart", timeout=30)
            r.raise_for_status()
            s = r.json()
            running = s.get("running", False)
            dot_color = GREEN if running else YELLOW
            status_text = f"Server: {'restarted ✓' if running else 'starting…'}"
            q(self._srv_status_dot.config, fg=dot_color)
            q(self._srv_status_var.set, status_text)
            q(self._log_msg, f"[restart] {status_text}  pid={s.get('restarted_pid')}", "ok")
        except Exception as e:
            q(self._srv_status_dot.config, fg=RED)
            q(self._srv_status_var.set, "Server: restart failed")
            q(self._log_msg, f"[restart] failed: {e}", "err")
        q(self._restart_btn.config, state="normal", text="↺  Restart Server")

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

        req_file = HERE / "requirements.txt"
        if req_file.exists():
            q(self._log_msg, f"Installing from requirements.txt…", "dim")
            install_cmd = [pip, "install", "-r", str(req_file)]
        else:
            packages = [
                "mlx", "mlx-lm", "transformers>=4.51.0,<5.0",
                "safetensors", "huggingface_hub", "Pillow", "numpy",
                "requests", "msgpack", "tokenizers", "wandb", "weave", "moviepy",
            ]
            q(self._log_msg, f"Installing {len(packages)} packages…", "dim")
            install_cmd = [pip, "install"] + packages

        r = subprocess.run(install_cmd, capture_output=True, text=True)
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
            self._running = False
            self._run_btn.config(text="⏳  Stopping…", bg=BG3, state="disabled")
            # poll until the eval thread fully exits, then re-enable
            self.root.after(200, self._poll_thread_done)
        else:
            self._stop_event.clear()
            self._running = True
            self._run_btn.config(text="■  Stop", bg=RED)
            t = threading.Thread(target=self._eval_thread, daemon=True)
            t.start()
            self._eval_thread_ref = t

    def _poll_thread_done(self):
        t = self._eval_thread_ref
        if t and t.is_alive():
            self.root.after(200, self._poll_thread_done)
        else:
            self._run_btn.config(text="▶  Run Eval", bg=ACCENT, state="normal")

    # ── eval thread ───────────────────────────────────────────────────────────

    def _eval_thread(self):
        def q(fn, *a, **kw): self._q.put((fn, a, kw))

        url    = self._url_var.get().strip()
        n_eps  = int(self._ep_var.get() or 5)
        seed   = int(self._seed_var.get() or 42)
        task   = self._task_var.get()
        n_steps = 8
        n_diff  = int(self._diff_var.get() or 4)

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
            q(self._run_btn.config, text="▶  Run Eval", bg=ACCENT, state="normal")
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

        # ── weave init ────────────────────────────────────────────────────────
        weave_log_ep = None
        try:
            api_key = os.environ.get("WANDB_API_KEY", "")
            if not api_key:
                q(self._log_msg, "WANDB_API_KEY not set — skipping weave logging", "dim")
            else:
                import weave as _weave
                _weave.init("gemma-robot")

                @_weave.op
                def _log_ep(ep, seed, instruction, success, steps, elapsed_s, video_path):
                    try:
                        from moviepy.editor import VideoFileClip
                    except ImportError:
                        from moviepy import VideoFileClip
                    return {
                        "episode": ep, "seed": seed, "instruction": instruction,
                        "success": success, "steps": steps, "elapsed_s": round(elapsed_s, 2),
                        "video": VideoFileClip(video_path),
                    }

                weave_log_ep = _log_ep
                q(self._log_msg, "Weave logging → gemma-robot", "ok")
        except Exception as e:
            q(self._log_msg, f"Weave init failed: {e}", "dim")
            weave_log_ep = None

        successes = []

        for ep in range(n_eps):
            if self._stop_event.is_set():
                break

            ep_seed = seed + ep * 1000
            q(self._log_msg, f"\n── ep {ep+1}/{n_eps}  seed={ep_seed}", "hi")
            q(self._ep_lbl.config, text=f"{ep+1}/{n_eps}")

            try:
                result = client.reset(
                    env_name=task,
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
            ep_frames = []

            while not done and not self._stop_event.is_set():
                image, robot_state, instruction, img_arr = obs_to_inputs(obs)

                ep_frames.append(img_arr.astype(np.uint8))
                q(self._update_video, img_arr)
                q(self._instr_var.set, instruction)

                t0 = time.time()
                raw = self._vla.get_action(
                    image=image,
                    robot_state=robot_state,
                    instruction=instruction,
                    n_diffusion_steps=n_diff,
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

            # ── log video to weave ────────────────────────────────────────────
            if weave_log_ep and ep_frames:
                try:
                    from moviepy.editor import ImageSequenceClip
                except ImportError:
                    try:
                        from moviepy import ImageSequenceClip
                    except Exception:
                        ImageSequenceClip = None
                if weave_log_ep and ImageSequenceClip is not None:
                    try:
                        video_path = f"/tmp/gr00t_ep{ep+1}_seed{ep_seed}.mp4"
                        ImageSequenceClip(list(ep_frames), fps=10).write_videofile(
                            video_path, logger=None
                        )
                        weave_log_ep(
                            ep=ep + 1, seed=ep_seed, instruction=instruction,
                            success=success, steps=step, elapsed_s=0.0,
                            video_path=video_path,
                        )
                        q(self._log_msg, "  → logged to weave", "dim")
                    except Exception as e:
                        q(self._log_msg, f"  weave log failed: {e}", "dim")

        q(self._log_msg, "\nEval complete.", "hi")
        q(self._set_status, "Done", GREEN)
        self._running = False
        # flush all pending Metal GPU work before the thread exits
        try:
            import mlx.core as mx
            mx.synchronize()
        except Exception:
            pass

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
