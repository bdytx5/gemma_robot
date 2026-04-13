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
import logging
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, font as tkfont
import msgpack
import numpy as np
import requests

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ISAAC = HERE.parent / "Isaac-GR00T"

# ── crash log ─────────────────────────────────────────────────────────────────
_LOG_FILE = HERE / "app_crash.log"
logging.basicConfig(
    filename=str(_LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("app")
log.info("=== app started (pid %s) ===", os.getpid())

def _log_exc(label: str, exc: BaseException | None = None):
    """Write a labelled traceback to app_crash.log."""
    tb = traceback.format_exc() if exc is None else "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__))
    log.error("%s\n%s", label, tb)

sys.excepthook = lambda t, v, tb: (
    log.critical("Unhandled exception: %s", "".join(traceback.format_exception(t, v, tb))),
    sys.__excepthook__(t, v, tb),
)

# ── pre-import SDL/pygame on the main thread ──────────────────────────────────
# pygame (pulled in transitively by weave→matplotlib) calls SDL_VideoInit /
# Cocoa_InitKeyboard on first import.  macOS requires that to happen on the
# main thread; if a background thread triggers it first the process gets
# SIGTRAP from dispatch_assert_queue_fail.  Importing here, before any threads
# are spawned, forces the one-time init to land on the main thread safely.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless – no window
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
try:
    import pygame as _pygame
    _pygame.init()
    log.info("pygame pre-init OK (SDL headless)")
except Exception as _e:
    log.warning("pygame pre-init skipped: %s", _e)
EAGLE = HERE.parent / "Eagle" / "Eagle2_5"
for p in [str(HERE), str(ISAAC), str(EAGLE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── remote config ────────────────────────────────────────────────────────────
_CONFIG_DOC = "https://docs.google.com/document/d/1wF4OwIRHZAGgZDBgXPliWfG589QbeTSKM46SozpsxWI/export?format=txt"

def _fetch_server_url() -> str:
    try:
        r = requests.get(_CONFIG_DOC, timeout=5)
        url = r.text.strip().lstrip("\ufeff")
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

# ── dtype config ──────────────────────────────────────────────────────────────
# Controls weight storage and compute precision for all components (vision, LLM, DiT).
# Options: "float16" | "bfloat16" | "float32"
COMPUTE_DTYPE = "bfloat16"

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
NGROK_HEADERS = {"ngrok-skip-browser-warning": "1"}

class SimClient:
    def __init__(self, url, timeout=120):
        self.base = url.rstrip("/")
        self.timeout = timeout

    def ping(self):
        return requests.get(f"{self.base}/info", timeout=8,
                            headers=NGROK_HEADERS).json()

    def reset(self, env_name=None, seed=None, max_episode_steps=None,
              n_action_steps=None):
        body = {}
        if env_name:          body["env_name"] = env_name
        if seed is not None:  body["seed"] = seed
        if max_episode_steps: body["max_episode_steps"] = max_episode_steps
        if n_action_steps:    body["n_action_steps"] = n_action_steps
        r = requests.post(f"{self.base}/reset", data=pack(body),
                          headers={**HEADERS, **NGROK_HEADERS}, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def step(self, action):
        r = requests.post(f"{self.base}/step",
                          data=pack({"action": action}),
                          headers={**HEADERS, **NGROK_HEADERS}, timeout=self.timeout)
        r.raise_for_status()
        return unpack(r.content)

    def step_cuda(self, obs):
        """Ask server to run GPU inference + step in one round-trip."""
        r = requests.post(f"{self.base}/step_cuda",
                          data=pack({"obs": obs}),
                          headers={**HEADERS, **NGROK_HEADERS}, timeout=self.timeout)
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
        self.root.geometry("1380x780")
        self.root.minsize(1100, 660)

        self._q = queue.Queue()
        self._stop_event = threading.Event()
        self._skip_ep_event = threading.Event()
        self._vla = None
        self._running = False
        self._eval_thread_ref = None

        # gallery state
        self._past_episodes = []        # list of {ep, seed, success, steps, frames, instruction, weave_url}
        self._gallery_frames = []       # frames currently loaded in gallery player
        self._gallery_frame_idx = 0
        self._gallery_playing = False
        self._gallery_after_id = None
        self._gallery_fps = 5           # default 5fps
        self._gallery_cycle_var = tk.BooleanVar(value=True)
        self._gallery_current_ep_idx = 0  # which episode is loaded in the player

        # URL edit state
        self._url_editing = False
        self._weave_project_url = None  # set when weave inits

        # dev mode
        self._dev_mode_var = tk.BooleanVar(value=False)

        # gallery filter
        self._show_success_only_var = tk.BooleanVar(value=False)
        self._gallery_ep_indices = []   # display_idx → _past_episodes index

        # episode cache
        self._cache_path = Path.home() / ".cache" / "gemma_robot" / "episodes.pkl"

        # busy flags for label-buttons
        self._setup_busy   = False
        self._restart_busy = False

        self._build_ui()
        self.root.after(16, self._drain_queue)
        threading.Thread(target=self._auto_setup_check, daemon=True).start()

        sys.stdout = _StreamRedirect(self, "")
        sys.stderr = _StreamRedirect(self, "err")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ---- top bar ----
        top = tk.Frame(self.root, bg=BG, pady=10, padx=16)
        top.pack(fill="x")

        tk.Label(top, text="GemmaRobot", bg=BG, fg=ACCENT,
                 font=(FONT, 18, "bold")).pack(side="left")

        self._status_dot = tk.Label(top, text="●", bg=BG, fg=RED,
                                     font=(FONT, 14))
        self._status_dot.pack(side="right", padx=(0, 4))
        self._status_lbl = tk.Label(top, text="Disconnected", bg=BG,
                                     fg=TEXT_DIM, font=(FONT, 11))
        self._status_lbl.pack(side="right")

        tk.Checkbutton(top, text="Dev", variable=self._dev_mode_var,
                       bg=BG, fg=TEXT_DIM, activebackground=BG,
                       activeforeground=TEXT, selectcolor=BG3,
                       font=(FONT, 10), relief="flat", cursor="hand2",
                       command=self._toggle_dev_mode).pack(side="right", padx=(0, 16))

        tk.Frame(self.root, bg=BG3, height=1).pack(fill="x")

        # ---- config row ----
        self._cfg_frame = cfg = tk.Frame(self.root, bg=BG2, pady=10, padx=16)
        cfg.pack(fill="x")

        tk.Label(cfg, text="Server URL", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._url_var = tk.StringVar(value="https://")
        threading.Thread(target=self._load_server_url, daemon=True).start()

        url_frame = tk.Frame(cfg, bg=BG2)
        url_frame.grid(row=0, column=1, padx=(0, 12), sticky="w")
        self._url_mask_lbl = tk.Label(url_frame, text="loading…", bg=BG3,
                                       fg=TEXT_DIM, font=(FONT, 12),
                                       width=38, anchor="w", padx=6, pady=3)
        self._url_mask_lbl.pack(side="left")
        self._url_entry = tk.Entry(url_frame, textvariable=self._url_var, bg=BG3,
                                    fg=TEXT, insertbackground=TEXT, relief="flat",
                                    font=(FONT, 12), width=38)
        self._url_edit_btn = self._lbtn(url_frame, "✎", BG3, ACCENT,
                                        font=(FONT, 10), padx=6, pady=2,
                                        command=self._toggle_url_edit)
        self._url_edit_btn.pack(side="left", padx=(4, 0))
        self._url_var.trace_add("write", lambda *_: self._q.put((self._update_url_mask, (), {})))

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
                 font=(FONT, 12), width=6).grid(row=0, column=5, padx=(0, 12))

        tk.Label(cfg, text="Diff steps", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=6, sticky="w", padx=(0, 6))
        self._diff_var = tk.StringVar(value="4")
        tk.Entry(cfg, textvariable=self._diff_var, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=(FONT, 12), width=3).grid(row=0, column=7, padx=(0, 16))

        self._run_btn = self._lbtn(cfg, "▶  Run Eval", ACCENT, "#000000",
                                    font=(FONT, 12, "bold"), padx=16, pady=4,
                                    command=self._on_run_stop)
        self._run_btn.grid(row=0, column=8, padx=(0, 6))

        self._next_btn = self._lbtn(cfg, "⏭  Next Ep", BG3, "#555555",
                                     font=(FONT, 11), padx=10, pady=4,
                                     command=self._on_next_episode_guarded)
        self._next_btn.grid(row=0, column=9, padx=(0, 6))

        self._setup_btn = self._lbtn(cfg, "⚙  Setup Env", BG3, TEXT_DIM,
                                      font=(FONT, 11), padx=12, pady=4,
                                      command=self._on_setup_env)
        self._setup_btn.grid(row=0, column=10, padx=(0, 6))

        self._restart_btn = self._lbtn(cfg, "⟳  Restart Sim Server", BG3, YELLOW,
                                        font=(FONT, 11), padx=12, pady=4,
                                        command=self._on_restart_sim)
        self._restart_btn.grid(row=0, column=11)

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
        ttk.Combobox(cfg, textvariable=self._task_var, values=TASKS,
                     font=(FONT, 11), width=52,
                     state="readonly").grid(row=1, column=1, columnspan=5,
                                            sticky="w", pady=(8, 0))

        self._dtype_lbl_w = tk.Label(cfg, text="dtype", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11))
        self._dtype_lbl_w.grid(row=1, column=6, sticky="w", padx=(12, 6), pady=(8, 0))
        self._dtype_var = tk.StringVar(value=COMPUTE_DTYPE)
        self._dtype_cb_w = ttk.Combobox(cfg, textvariable=self._dtype_var,
                                 values=["float16", "bfloat16", "float32"],
                                 font=(FONT, 11), width=10, state="readonly")
        self._dtype_cb_w.grid(row=1, column=7, sticky="w", pady=(8, 0))
        self._dtype_var.trace_add("write", self._on_dtype_change)

        tk.Label(cfg, text="Inference", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=1, column=8, sticky="w", padx=(12, 6), pady=(8, 0))
        self._infer_mode_var = tk.StringVar(value="GPU (server)")
        self._infer_mode_cb = ttk.Combobox(cfg, textvariable=self._infer_mode_var,
                                            values=["MLX (local)", "MLX reb0rn", "GPU (server)"],
                                            font=(FONT, 11), width=13, state="readonly")
        self._infer_mode_cb.grid(row=1, column=9, sticky="w", pady=(8, 0))
        self._infer_mode_var.trace_add("write", self._on_infer_mode_change)

        # ---- norm row (dev mode only — hidden by default) ----
        self._norm_row = tk.Frame(self.root, bg=BG2, pady=6, padx=16)
        # NOT packed here; shown via _toggle_dev_mode

        self._state_norm_var = tk.StringVar(value="min/max")
        self._grip_norm_var  = tk.StringVar(value="min/max")
        self._stats_src_var  = tk.StringVar(value="checkpoint")

        tk.Label(self._norm_row, text="State norm", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(self._norm_row, textvariable=self._state_norm_var,
                     values=["min/max", "q01/q99"],
                     font=(FONT, 11), width=9, state="readonly").grid(
                     row=0, column=1, sticky="w")
        self._state_norm_var.trace_add("write", self._on_norm_change)

        tk.Label(self._norm_row, text="Grip denorm", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=2, sticky="w", padx=(20, 8))
        ttk.Combobox(self._norm_row, textvariable=self._grip_norm_var,
                     values=["min/max", "q01/q99"],
                     font=(FONT, 11), width=9, state="readonly").grid(
                     row=0, column=3, sticky="w")
        self._grip_norm_var.trace_add("write", self._on_norm_change)

        tk.Label(self._norm_row, text="Stats", bg=BG2, fg=TEXT_DIM,
                 font=(FONT, 11)).grid(row=0, column=4, sticky="w", padx=(20, 8))
        ttk.Combobox(self._norm_row, textvariable=self._stats_src_var,
                     values=["checkpoint", "root"],
                     font=(FONT, 11), width=10, state="readonly").grid(
                     row=0, column=5, sticky="w")
        self._stats_src_var.trace_add("write", self._on_norm_change)

        # hide dtype widgets by default (dev mode)
        self._dtype_lbl_w.grid_remove()
        self._dtype_cb_w.grid_remove()

        tk.Frame(self.root, bg=BG3, height=1).pack(fill="x")

        # ---- log panel (bottom) ----
        tk.Frame(self.root, bg=BG3, height=1).pack(side="bottom", fill="x")
        log_outer = tk.Frame(self.root, bg=BG, height=140)
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

        # LEFT: live feed
        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both")

        tk.Label(left, text="LIVE FEED", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w")

        self._video_lbl = tk.Label(left, bg="#000000", width=480, height=360)
        self._video_lbl.pack(pady=(4, 4))

        self._instr_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self._instr_var, bg=BG, fg=TEXT_DIM,
                 font=(FONT, 10), wraplength=480, justify="left",
                 anchor="w").pack(fill="x", padx=4, pady=(0, 6))

        stats_bar = tk.Frame(left, bg=BG2, pady=8, padx=12)
        stats_bar.pack(fill="x")
        self._ep_lbl   = self._stat(stats_bar, "Episode", "—", 0)
        self._step_lbl = self._stat(stats_bar, "Step",    "—", 1)
        self._sr_lbl   = self._stat(stats_bar, "Success", "—", 2)
        self._inf_lbl  = self._stat(stats_bar, "Infer",   "—", 3)

        # MIDDLE: action bars
        mid = tk.Frame(content, bg=BG)
        mid.pack(side="left", fill="y", padx=(16, 0))

        tk.Label(mid, text="PREDICTED ACTIONS", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(anchor="w")

        self._action_frame = tk.Frame(mid, bg=BG2, pady=8, padx=12)
        self._action_frame.pack(fill="x", pady=(4, 0))

        self._action_bars = {}
        for i, k in enumerate(["x", "y", "z", "roll", "pitch", "yaw", "grip"]):
            self._action_bars[k] = self._action_row(self._action_frame, k, i)

        # RIGHT: episode gallery
        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        hdr = tk.Frame(right, bg=BG)
        hdr.pack(fill="x")
        tk.Label(hdr, text="EPISODE GALLERY", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(side="left")
        self._gallery_play_btn = self._lbtn(hdr, "▶", BG3, TEXT,
                                             font=(FONT, 10), padx=6, pady=1,
                                             command=self._gallery_toggle_play)
        self._gallery_play_btn.pack(side="right")
        self._lbtn(hdr, "+", BG3, TEXT_DIM, font=(FONT, 10), padx=4, pady=1,
                   command=self._gallery_faster).pack(side="right")
        self._gallery_speed_lbl = tk.Label(hdr, text="5fps", bg=BG,
                                            fg=TEXT_DIM, font=(FONT, 9))
        self._gallery_speed_lbl.pack(side="right", padx=(0, 2))
        self._lbtn(hdr, "−", BG3, TEXT_DIM, font=(FONT, 10), padx=4, pady=1,
                   command=self._gallery_slower).pack(side="right")
        tk.Label(hdr, text="Cycle", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9)).pack(side="right", padx=(0, 2))
        tk.Checkbutton(hdr, variable=self._gallery_cycle_var,
                       bg=BG, activebackground=BG, selectcolor=BG3,
                       fg=ACCENT, activeforeground=ACCENT,
                       relief="flat", cursor="hand2").pack(side="right", padx=(0, 8))
        tk.Label(hdr, text="Success only", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9)).pack(side="right", padx=(0, 2))
        tk.Checkbutton(hdr, variable=self._show_success_only_var,
                       bg=BG, activebackground=BG, selectcolor=BG3,
                       fg=ACCENT, activeforeground=ACCENT,
                       relief="flat", cursor="hand2",
                       command=self._refresh_gallery_list).pack(side="right", padx=(0, 8))
        self._lbtn(hdr, "📂 Load prev", BG3, TEXT_DIM, font=(FONT, 9),
                   padx=6, pady=1,
                   command=self._load_episodes_from_cache).pack(side="right", padx=(0, 8))

        # episode list
        list_frame = tk.Frame(right, bg=BG2)
        list_frame.pack(fill="x", pady=(4, 0))

        self._gallery_list = tk.Listbox(
            list_frame, bg=BG2, fg=TEXT, font=("SF Mono", 11),
            relief="flat", selectbackground=BG3, selectforeground=ACCENT,
            height=6, activestyle="none",
        )
        self._gallery_list.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        list_sb = ttk.Scrollbar(list_frame, command=self._gallery_list.yview)
        list_sb.pack(side="right", fill="y")
        self._gallery_list.config(yscrollcommand=list_sb.set)
        self._gallery_list.bind("<<ListboxSelect>>", self._on_gallery_select)

        # gallery video player
        playback_hdr = tk.Frame(right, bg=BG)
        playback_hdr.pack(fill="x", pady=(10, 0))
        tk.Label(playback_hdr, text="PLAYBACK", bg=BG, fg=TEXT_DIM,
                 font=(FONT, 9, "bold")).pack(side="left")
        self._gallery_weave_lbl = tk.Label(playback_hdr, text="", bg=BG,
                                            fg=ACCENT, font=(FONT, 9),
                                            cursor="hand2")
        self._gallery_weave_lbl.pack(side="right", padx=(0, 4))
        self._gallery_lbl = tk.Label(right, bg="#000000",
                                      width=380, height=285)
        self._gallery_lbl.pack(pady=(4, 0))

        self._gallery_info = tk.Label(right, text="Select an episode to play",
                                       bg=BG, fg=TEXT_DIM, font=(FONT, 10))
        self._gallery_info.pack(anchor="w", padx=4, pady=(4, 0))

    # ── label-button helper (avoids macOS Aqua white-render + focus-click) ──────

    def _lbtn(self, parent, text, bg, fg, font=None, padx=10, pady=4,
              command=None, **kw):
        """A tk.Label that behaves as a button — fires on first click, honours colours."""
        if font is None:
            font = (FONT, 11)
        lbl = tk.Label(parent, text=text, bg=bg, fg=fg, font=font,
                       padx=padx, pady=pady, cursor="hand2", **kw)
        if command:
            lbl.bind("<Button-1>", lambda e: command())
        return lbl

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
            self._log.tag_config(link_tag, foreground=ACCENT, underline=True)
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

    def _update_url_mask(self):
        url = self._url_var.get()
        if len(url) > 16 and not self._url_editing:
            masked = url[:8] + "•••" + url[-6:]
        else:
            masked = url if url else "https://"
        self._url_mask_lbl.config(text=masked)

    def _toggle_url_edit(self):
        if self._url_editing:
            self._url_entry.pack_forget()
            self._url_mask_lbl.pack(side="left")
            self._url_edit_btn.config(text="✎")
            self._url_editing = False
            self._update_url_mask()
        else:
            self._url_mask_lbl.pack_forget()
            self._url_entry.pack(side="left")
            self._url_entry.focus_set()
            self._url_edit_btn.config(text="✓")
            self._url_editing = True

    def _set_status(self, text, color):
        self._status_lbl.config(text=text)
        self._status_dot.config(fg=color)

    def _update_video(self, img_arr):
        from PIL import Image as PILImage, ImageTk
        img = PILImage.fromarray(img_arr.astype(np.uint8))
        img = img.resize((480, 360), PILImage.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._video_lbl.config(image=photo, width=480, height=360)
        self._video_lbl._photo = photo

    def _update_actions(self, actions):
        a = actions[0, 0, :7]
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
        for i, k in enumerate(keys):
            v = float(a[i])
            self._action_bars[k]["val"].config(text=f"{v:+.4f}")
            c = self._action_bars[k]["canvas"]
            c.delete("all")
            mid = 80
            bar_w = min(abs(v) / 0.3 * 75, 75)
            color = ACCENT if v >= 0 else "#ff79c6"
            x0 = mid if v >= 0 else mid - bar_w
            c.create_rectangle(x0, 2, x0 + bar_w, 10, fill=color, outline="")

    # ── gallery ───────────────────────────────────────────────────────────────

    def _add_episode_to_gallery(self, ep_data: dict):
        """Called from eval thread via queue. ep_data has ep/seed/success/steps/frames/instruction."""
        self._past_episodes.append(ep_data)
        self._save_episode_to_cache(ep_data)
        self._refresh_gallery_list()
        # auto-select and play the latest episode (look up display index)
        actual_idx = len(self._past_episodes) - 1
        if actual_idx in self._gallery_ep_indices:
            display_idx = self._gallery_ep_indices.index(actual_idx)
            self._gallery_list.selection_clear(0, "end")
            self._gallery_list.selection_set(display_idx)
            self._gallery_list.see(display_idx)
            self._load_gallery_episode(actual_idx)

    def _on_gallery_select(self, event):
        sel = self._gallery_list.curselection()
        if not sel:
            return
        display_idx = sel[0]
        if display_idx < len(self._gallery_ep_indices):
            actual_idx = self._gallery_ep_indices[display_idx]
            self._load_gallery_episode(actual_idx)

    def _load_gallery_episode(self, idx: int):
        import webbrowser
        if idx < 0 or idx >= len(self._past_episodes):
            return
        # Cancel any running tick loop before starting a new one
        if self._gallery_after_id:
            self.root.after_cancel(self._gallery_after_id)
            self._gallery_after_id = None
        ep = self._past_episodes[idx]
        self._gallery_current_ep_idx = idx
        self._gallery_frames = ep["frames"]
        self._gallery_frame_idx = 0
        if ep.get("stopped"):
            succ = "⏹ stopped"
        elif ep["success"]:
            succ = "✓ success"
        else:
            succ = "✗ failed"
        self._gallery_info.config(
            text=f"Ep {ep['ep']}  {succ}  {ep['steps']} steps  |  {ep['instruction'][:40]}"
        )
        # weave link
        weave_url = ep.get("weave_url") or self._weave_project_url
        if weave_url:
            self._gallery_weave_lbl.config(text="⬡ View in Weave ↗")
            self._gallery_weave_lbl.bind("<Button-1>", lambda e, u=weave_url: webbrowser.open(u))
        else:
            self._gallery_weave_lbl.config(text="")
            self._gallery_weave_lbl.unbind("<Button-1>")
        self._gallery_playing = True
        self._gallery_play_btn.config(text="⏸")
        self._gallery_tick()

    def _gallery_tick(self):
        if not self._gallery_frames or not self._gallery_playing:
            return
        from PIL import Image as PILImage, ImageTk
        idx = self._gallery_frame_idx % len(self._gallery_frames)
        frame = self._gallery_frames[idx]
        img = PILImage.fromarray(frame)
        img = img.resize((380, 285), PILImage.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._gallery_lbl.config(image=photo, width=380, height=285)
        self._gallery_lbl._photo = photo

        next_frame_idx = self._gallery_frame_idx + 1
        if next_frame_idx >= len(self._gallery_frames):
            # Completed one full pass of this episode
            if self._gallery_cycle_var.get() and self._gallery_ep_indices:
                # find current position in display list and advance
                cur = self._gallery_current_ep_idx
                if cur in self._gallery_ep_indices:
                    pos = self._gallery_ep_indices.index(cur)
                else:
                    pos = 0
                next_pos = (pos + 1) % len(self._gallery_ep_indices)
                next_actual = self._gallery_ep_indices[next_pos]
                self._gallery_list.selection_clear(0, "end")
                self._gallery_list.selection_set(next_pos)
                self._gallery_list.see(next_pos)
                self._load_gallery_episode(next_actual)
                return
            else:
                self._gallery_frame_idx = 0
        else:
            self._gallery_frame_idx = next_frame_idx

        delay = max(33, 1000 // self._gallery_fps)
        self._gallery_after_id = self.root.after(delay, self._gallery_tick)

    def _gallery_slower(self):
        fps_steps = [1, 2, 3, 5, 8, 10, 15, 20, 30]
        idx = next((i for i, v in enumerate(fps_steps) if v >= self._gallery_fps), len(fps_steps) - 1)
        if idx > 0:
            self._gallery_fps = fps_steps[idx - 1]
            self._gallery_speed_lbl.config(text=f"{self._gallery_fps}fps")

    def _gallery_faster(self):
        fps_steps = [1, 2, 3, 5, 8, 10, 15, 20, 30]
        idx = next((i for i, v in enumerate(fps_steps) if v >= self._gallery_fps), len(fps_steps) - 1)
        if idx < len(fps_steps) - 1:
            self._gallery_fps = fps_steps[idx + 1]
            self._gallery_speed_lbl.config(text=f"{self._gallery_fps}fps")

    def _gallery_toggle_play(self):
        if not self._gallery_frames:
            return
        self._gallery_playing = not self._gallery_playing
        if self._gallery_playing:
            self._gallery_play_btn.config(text="⏸")
            self._gallery_tick()
        else:
            self._gallery_play_btn.config(text="▶")
            if self._gallery_after_id:
                self.root.after_cancel(self._gallery_after_id)

    # ── inference mode change ─────────────────────────────────────────────────

    def _on_infer_mode_change(self, *_):
        """Invalidate cached model when inference mode changes so it reloads."""
        if self._vla is not None:
            self._vla = None
            mode = self._infer_mode_var.get()
            self._log_msg(f"inference mode → {mode}  (model will reload on next run)", "dim")
        # Norm controls only apply to MLX classic
        self._update_norm_controls_state()

    def _update_norm_controls_state(self):
        """Dim norm controls when reb0rn mode is active (they have no effect)."""
        is_reborn = self._infer_mode_var.get() == "MLX reb0rn"
        fg = TEXT_DIM if is_reborn else TEXT
        # Walk the config frame looking for norm-related widgets — simplest: just
        # log a note so the user knows; full widget disabling would require refs.
        pass  # visual dimming deferred; the log message is enough

    # ── dtype change ─────────────────────────────────────────────────────────

    def _on_dtype_change(self, *_):
        """Invalidate cached model when dtype changes so it reloads on next run."""
        if self._vla is not None:
            self._vla = None
            self._log_msg(f"dtype → {self._dtype_var.get()}  (model will reload on next run)", "dim")

    # ── norm change ───────────────────────────────────────────────────────────

    def _on_norm_change(self, *_):
        """Reapply norm stats to live model when norm strategy changes. No model reload needed."""
        if self._vla is None:
            return
        if self._infer_mode_var.get() == "MLX reb0rn":
            self._log_msg(
                "reb0rn mode: norm is handled by StateActionProcessor — "
                "state/grip dropdowns have no effect", "dim"
            )
            return
        threading.Thread(target=self._reload_norm_stats, daemon=True).start()

    def _reload_norm_stats(self):
        """Reload statistics.json and patch self._vla norm arrays in place."""
        q = lambda fn, *a, **kw: self._q.put((fn, a, kw))
        try:
            import json, numpy as np
            from huggingface_hub import hf_hub_download

            use_ckpt = self._stats_src_var.get() == "checkpoint"
            state_use_minmax = self._state_norm_var.get() == "min/max"
            grip_use_minmax  = self._grip_norm_var.get()  == "min/max"

            stats_name = "checkpoint-2000/statistics.json" if use_ckpt else "statistics.json"
            try:
                stats_path = hf_hub_download(
                    "youngbrett48/gr00t-post-train-fractal-270m", stats_name, token=True)
            except Exception:
                stats_path = hf_hub_download(
                    "youngbrett48/gr00t-post-train-fractal-270m", "statistics.json", token=True)

            with open(stats_path) as f:
                all_stats = json.load(f)
            s = all_stats.get("oxe_google", {}).get("state", {})
            a = all_stats.get("oxe_google", {}).get("action", {})

            state_keys = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
            if state_use_minmax:
                self._vla._state_min = np.array([s[k]["min"][0] for k in state_keys], dtype=np.float32)
                self._vla._state_max = np.array([s[k]["max"][0] for k in state_keys], dtype=np.float32)
            else:
                self._vla._state_min = np.array([s[k]["q01"][0] for k in state_keys], dtype=np.float32)
                self._vla._state_max = np.array([s[k]["q99"][0] for k in state_keys], dtype=np.float32)

            ms_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
            self._vla._action_mean = np.array([a[k]["mean"][0] for k in ms_keys], dtype=np.float32)
            self._vla._action_std  = np.array([a[k]["std"][0]  for k in ms_keys], dtype=np.float32)
            if grip_use_minmax:
                self._vla._action_grip_min = np.float32(a["gripper"]["min"][0])
                self._vla._action_grip_max = np.float32(a["gripper"]["max"][0])
            else:
                self._vla._action_grip_min = np.float32(a["gripper"]["q01"][0])
                self._vla._action_grip_max = np.float32(a["gripper"]["q99"][0])

            q(self._log_msg,
              f"Norm updated: state={'min/max' if state_use_minmax else 'q01/q99'}  "
              f"grip={'min/max' if grip_use_minmax else 'q01/q99'}  "
              f"stats={'ckpt' if use_ckpt else 'root'}", "dim")
        except Exception as e:
            q(self._log_msg, f"Norm reload failed: {e}", "err")

    # ── next episode ──────────────────────────────────────────────────────────

    def _on_next_episode(self):
        self._skip_ep_event.set()

    # ── restart sim ───────────────────────────────────────────────────────────

    def _on_restart_sim(self):
        try:
            log.debug("_on_restart_sim: _restart_busy=%s", self._restart_busy)
            if self._restart_busy:
                return
            self._restart_busy = True
            self._restart_btn.config(text="⟳  Restarting…", fg=TEXT_DIM)
            threading.Thread(target=self._restart_sim_thread, daemon=True).start()
        except Exception as e:
            _log_exc("_on_restart_sim crashed", e)
            raise

    def _restart_sim_thread(self):
        def q(fn, *a, **kw): self._q.put((fn, a, kw))

        def done(ok, msg=""):
            color = GREEN if ok else RED
            label = "⟳  Restart Sim Server"
            self._restart_busy = False
            q(self._restart_btn.config, text=label, fg=YELLOW)
            q(self._set_status, msg or ("Sim ready" if ok else "Restart failed"), color)

        url = self._url_var.get().strip()
        if not url.startswith("http"):
            q(self._log_msg, "No server URL set.", "err")
            done(False, "No URL")
            return

        q(self._log_msg, "Restarting sim (may take up to 60 s)…", "hi")
        q(self._set_status, "Restarting…", YELLOW)
        try:
            r = requests.post(
                f"{url.rstrip('/')}/control/restart",
                json={},
                headers=NGROK_HEADERS,
                timeout=90,
            )
            r.raise_for_status()
            status = r.json()
            sim_ready = status.get("sim_ready", False)
            restarts  = status.get("sim_restart_count", "?")
            q(self._log_msg,
              f"Restart complete — sim_ready={sim_ready}  restart_count={restarts}", "ok" if sim_ready else "err")
            done(sim_ready, "Sim ready" if sim_ready else "Sim not ready")
        except Exception as e:
            q(self._log_msg, f"Restart failed: {e}", "err")
            done(False, "Restart failed")

    # ── setup env ─────────────────────────────────────────────────────────────

    def _on_setup_env(self):
        if self._setup_busy:
            return
        self._setup_busy = True
        self._setup_btn.config(text="⚙  Setting up…", fg=TEXT_DIM)
        t = threading.Thread(target=self._setup_env_thread, daemon=True)
        t.start()

    def _setup_env_thread(self):
        import subprocess
        def q(fn, *a, **kw): self._q.put((fn, a, kw))

        def done(ok):
            self._setup_busy = False
            color = ACCENT if ok else BG3
            q(self._setup_btn.config,
              text="✓  Env Ready" if ok else "⚙  Setup Env",
              bg=color, fg="#000000" if ok else TEXT_DIM)

        q(self._log_msg, "\n── Setting up conda env ──", "hi")
        q(self._set_status, "Setting up…", YELLOW)

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
            q(self._log_msg, "Installing from requirements.txt…", "dim")
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
        q(self._log_msg,
          f"\nEnv '{env_name}' is ready.\nRelaunch the app using:\n  conda activate {env_name} && python app.py",
          "hi")
        q(self._set_status, "Env Ready", GREEN)
        done(True)

    # ── auto setup check ──────────────────────────────────────────────────────

    def _auto_setup_check(self):
        """On startup: if mlx_robot env is missing, auto-trigger setup."""
        import subprocess
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
            r = subprocess.run(["which", "conda"], capture_output=True, text=True)
            if r.returncode == 0:
                conda = r.stdout.strip()
        if not conda:
            return
        r = subprocess.run([conda, "env", "list"], capture_output=True, text=True)
        if "mlx_robot" not in r.stdout:
            self._q.put((self._log_msg, ("mlx_robot env not found — running auto setup…",), {"tag": "hi"}))
            self._q.put((self._on_setup_env, (), {}))

    # ── run/stop ──────────────────────────────────────────────────────────────

    def _on_next_episode_guarded(self):
        try:
            log.debug("_on_next_episode_guarded: _running=%s", self._running)
            if self._running:
                self._skip_ep_event.set()
        except Exception as e:
            _log_exc("_on_next_episode_guarded crashed", e)
            raise

    def _on_run_stop(self):
        try:
            log.debug("_on_run_stop: _running=%s", self._running)
            if self._running:
                self._stop_event.set()
                self._running = False
                self._run_btn.config(text="⏳  Stopping…", bg=BG3, fg=TEXT_DIM)
                self._next_btn.config(fg="#333333")
                self.root.after(200, self._poll_thread_done)
            else:
                self._stop_event.clear()
                self._skip_ep_event.clear()
                self._running = True
                self._run_btn.config(text="■  Stop", bg=RED, fg=TEXT)
                self._next_btn.config(fg=TEXT)
                t = threading.Thread(target=self._eval_thread, daemon=True)
                t.start()
                self._eval_thread_ref = t
        except Exception as e:
            _log_exc("_on_run_stop crashed", e)
            raise

    def _poll_thread_done(self):
        t = self._eval_thread_ref
        if t and t.is_alive():
            self.root.after(200, self._poll_thread_done)
        else:
            self._run_btn.config(text="▶  Run Eval", bg=ACCENT, fg="#000000")
            self._next_btn.config(fg="#555555")

    # ── eval thread ───────────────────────────────────────────────────────────

    def _eval_thread(self):
        try:
            self._eval_thread_inner()
        except Exception as e:
            _log_exc("_eval_thread crashed", e)
            def q(fn, *a, **kw): self._q.put((fn, a, kw))
            q(self._log_msg, f"CRASH: {e} — see app_crash.log", "err")
            self._running = False
            q(self._run_btn.config, text="▶  Run Eval", bg="#4CAF50", fg="#000000")
            q(self._next_btn.config, fg="#555555")

    def _eval_thread_inner(self):
        def q(fn, *a, **kw): self._q.put((fn, a, kw))

        url        = self._url_var.get().strip()
        n_eps      = int(self._ep_var.get() or 5)
        seed       = int(self._seed_var.get() or 42)
        task       = self._task_var.get()
        n_steps    = 8
        n_diff     = int(self._diff_var.get() or 4)
        infer_mode = self._infer_mode_var.get()
        gpu_mode   = infer_mode == "GPU (server)"
        reborn_mode = infer_mode == "MLX reb0rn"

        q(self._log_msg, f"Connecting to {url} …", "dim")
        q(self._set_status, "Connecting…", YELLOW)

        try:
            client = SimClient(url)
            info = client.ping()
            q(self._log_msg, f"Server: {info}", "dim")
            q(self._set_status, "Connected", GREEN)
        except Exception as e:
            q(self._log_msg, f"Connection failed: {e}", "err")
            q(self._set_status, "Disconnected", RED)
            q(self._run_btn.config, text="▶  Run Eval", bg=ACCENT, fg="#000000")
            q(self._next_btn.config, fg="#555555")
            self._running = False
            return

        if gpu_mode:
            q(self._log_msg, "GPU mode — inference runs on server, skipping local model load", "dim")

        if not gpu_mode and self._vla is None:
            label = "GemmaVLA reb0rn" if reborn_mode else "GemmaVLA"
            q(self._log_msg, f"Loading {label}…", "dim")
            q(self._set_status, "Loading model…", YELLOW)
            try:
                _dtype = self._dtype_var.get()
                q(self._log_msg, f"Loading model ({_dtype})…", "dim")

                if reborn_mode:
                    # ── reb0rn: load from mlx_reb0rn with official preprocessing ──
                    reborn_dir = HERE.parent / "mlx_reb0rn"
                    import importlib.util
                    reborn_spec = importlib.util.spec_from_file_location(
                        "gemma_vla_reborn", reborn_dir / "gemma_vla.py"
                    )
                    reborn_mod = importlib.util.module_from_spec(reborn_spec)
                    reborn_spec.loader.exec_module(reborn_mod)
                    GemmaVLA = reborn_mod.GemmaVLA

                    weights_dir = reborn_dir / "gr00t_weights_mlx"
                    llm_dir     = reborn_dir / "gr00t_llm_mlx"
                    if not (weights_dir / "meta.json").exists():
                        raise FileNotFoundError(f"reb0rn weights not found at {weights_dir}")
                    if not llm_dir.exists() or not any(llm_dir.iterdir()):
                        raise FileNotFoundError(f"reb0rn LLM not found at {llm_dir}")

                    q(self._log_msg, "Using reb0rn weights + official preprocessing", "dim")
                    self._vla = GemmaVLA.from_exported(
                        weights_dir=str(weights_dir),
                        mlx_llm_path=str(llm_dir),
                        n_diffusion_steps=4,
                        dtype=_dtype,
                    )
                else:
                    # ── classic: load from this directory (mlx_gr00t) ────────────
                    from gemma_vla import GemmaVLA

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
                        q(self._log_msg, "Using bundled weights", "dim")
                        self._vla = GemmaVLA.from_exported(
                            weights_dir=str(weights_dir),
                            mlx_llm_path=str(llm_dir),
                            n_diffusion_steps=4,
                            dtype=_dtype,
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
                            dtype=_dtype,
                        )

                q(self._log_msg, f"{label} ready ✓", "ok")
                q(self._set_status, "Ready", GREEN)
            except Exception as e:
                q(self._log_msg, f"Model load failed: {e}", "err")
                q(self._set_status, "Error", RED)
                self._running = False
                return

        # weave init
        weave_log_ep = None
        try:
            api_key = os.environ.get("WANDB_API_KEY", "")
            if not api_key:
                q(self._log_msg, "WANDB_API_KEY not set — skipping weave logging", "dim")
            else:
                import weave as _weave
                wc = _weave.init("gemma-robot")
                try:
                    self._weave_project_url = f"https://wandb.ai/{wc.entity}/{wc.project}/weave"
                except Exception:
                    self._weave_project_url = None

                @_weave.op
                def _log_ep(ep, seed, instruction, success, steps, elapsed_s, video_path):
                    try:
                        from moviepy.editor import VideoFileClip
                    except ImportError:
                        from moviepy import VideoFileClip
                    return {
                        "episode": ep, "seed": seed, "instruction": instruction,
                        "success": success, "steps": steps,
                        "elapsed_s": round(elapsed_s, 2),
                        "video": VideoFileClip(video_path),
                    }

                weave_log_ep = _log_ep
                link = self._weave_project_url or "wandb.ai"
                q(self._log_msg, f"Weave logging → {link}", "ok")
        except Exception as e:
            q(self._log_msg, f"Weave init failed: {e}", "dim")

        successes = []

        for ep in range(n_eps):
            if self._stop_event.is_set():
                break

            self._skip_ep_event.clear()
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
                continue

            obs = result["obs"]
            done = False
            success = False
            step = 0
            ep_frames = []
            instruction = "pick up the object"
            t_ep_start = time.time()

            while not done and not self._stop_event.is_set() and not self._skip_ep_event.is_set():
                image, robot_state, instruction, img_arr = obs_to_inputs(obs)

                ep_frames.append(img_arr.astype(np.uint8))
                q(self._update_video, img_arr)
                q(self._instr_var.set, instruction)

                t0 = time.time()
                if gpu_mode:
                    # Single round-trip: server runs GPU inference + steps env
                    try:
                        result = client.step_cuda(obs)
                    except Exception as e:
                        q(self._log_msg, f"step_cuda failed: {e}", "err")
                        break
                    dt = time.time() - t0
                    q(self._inf_lbl.config, text=f"{dt*1000:.0f}ms  (GPU)")
                else:
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

            skipped = self._skip_ep_event.is_set()
            stopped = self._stop_event.is_set()
            was_cut_short = skipped or stopped
            elapsed = time.time() - t_ep_start

            if not was_cut_short:
                successes.append(success)
            sr = sum(successes) / len(successes) if successes else 0.0
            color = "ok" if success else ("dim" if was_cut_short else "err")
            cut_note = "  [skipped]" if skipped else ("  [stopped]" if stopped else "")
            q(self._log_msg,
              f"  done  steps={step}  success={success}  sr={sr:.0%}{cut_note}", color)
            q(self._sr_lbl.config, text=f"{sr:.0%}")

            # add to gallery (non-blocking — runs in main thread via queue)
            if ep_frames:
                ep_data = {
                    "ep": ep + 1,
                    "seed": ep_seed,
                    "success": success,
                    "stopped": was_cut_short,
                    "steps": step,
                    "frames": ep_frames,
                    "instruction": instruction,
                    "weave_url": self._weave_project_url,
                }
                q(self._add_episode_to_gallery, ep_data)

            # weave logging
            if weave_log_ep and ep_frames:
                try:
                    from moviepy.editor import ImageSequenceClip
                except ImportError:
                    try:
                        from moviepy import ImageSequenceClip
                    except Exception:
                        ImageSequenceClip = None
                if ImageSequenceClip is not None:
                    try:
                        video_path = f"/tmp/gr00t_ep{ep+1}_seed{ep_seed}.mp4"
                        ImageSequenceClip(list(ep_frames), fps=10).write_videofile(
                            video_path, logger=None
                        )
                        weave_log_ep(
                            ep=ep + 1, seed=ep_seed, instruction=instruction,
                            success=success, steps=step, elapsed_s=elapsed,
                            video_path=video_path,
                        )
                        q(self._log_msg, "  → logged to weave", "dim")
                    except Exception as e:
                        q(self._log_msg, f"  weave log failed: {e}", "dim")

        q(self._log_msg, "\nEval complete.", "hi")
        q(self._set_status, "Done", GREEN)
        q(self._next_btn.config, fg="#555555")
        self._running = False
        try:
            import mlx.core as mx
            mx.synchronize()
        except Exception:
            pass

    # ── dev mode ──────────────────────────────────────────────────────────────

    def _toggle_dev_mode(self):
        on = self._dev_mode_var.get()
        if on:
            self._norm_row.pack(fill="x", after=self._cfg_frame)
            self._dtype_lbl_w.grid()
            self._dtype_cb_w.grid()
        else:
            self._norm_row.pack_forget()
            self._dtype_lbl_w.grid_remove()
            self._dtype_cb_w.grid_remove()

    # ── gallery list refresh (respects success-only filter) ───────────────────

    def _refresh_gallery_list(self):
        self._gallery_list.delete(0, "end")
        self._gallery_ep_indices = []
        success_only = self._show_success_only_var.get()
        for i, ep_data in enumerate(self._past_episodes):
            stopped = ep_data.get("stopped", False)
            if success_only and (not ep_data["success"] or stopped):
                continue
            self._gallery_ep_indices.append(i)
            if stopped:
                icon  = "⏹"
                color = YELLOW
            elif ep_data["success"]:
                icon  = "✓"
                color = GREEN
            else:
                icon  = "✗"
                color = RED
            label = f"  Ep {ep_data['ep']:2d}  {icon}  {ep_data['steps']:3d} steps  —  {ep_data['instruction'][:28]}"
            self._gallery_list.insert("end", label)
            self._gallery_list.itemconfig("end", fg=color)

        # If the currently-displayed episode was filtered out, load the first
        # visible one (or clear the player if nothing passes the filter).
        if self._gallery_current_ep_idx not in self._gallery_ep_indices:
            if self._gallery_ep_indices:
                self._gallery_list.selection_set(0)
                self._load_gallery_episode(self._gallery_ep_indices[0])
            else:
                # nothing to show — stop playback and clear the info label
                if self._gallery_after_id:
                    self.root.after_cancel(self._gallery_after_id)
                    self._gallery_after_id = None
                self._gallery_playing = False
                self._gallery_info.config(text="No episodes match the filter")

    # ── episode cache ─────────────────────────────────────────────────────────

    def _save_episode_to_cache(self, ep_data: dict):
        """Append one episode (with downsampled frames) to disk cache."""
        try:
            import pickle
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Subsample frames to keep cache lean (max 60 frames per episode)
            frames = ep_data.get("frames", [])
            if len(frames) > 60:
                step = len(frames) / 60
                frames = [frames[int(i * step)] for i in range(60)]
            entry = {k: v for k, v in ep_data.items() if k != "frames"}
            entry["frames"] = frames
            # Load existing cache, append, keep last 200 episodes
            existing = []
            if self._cache_path.exists():
                try:
                    with open(self._cache_path, "rb") as f:
                        existing = pickle.load(f)
                except Exception:
                    existing = []
            existing.append(entry)
            existing = existing[-200:]
            with open(self._cache_path, "wb") as f:
                pickle.dump(existing, f)
        except Exception as e:
            self._log_msg(f"Cache save failed: {e}", "dim")

    def _load_episodes_from_cache(self):
        """Load previously cached episodes into the gallery."""
        try:
            import pickle
            if not self._cache_path.exists():
                self._log_msg("No episode cache found.", "dim")
                return
            with open(self._cache_path, "rb") as f:
                cached = pickle.load(f)
            added = 0
            existing_keys = {(e["ep"], e["seed"]) for e in self._past_episodes}
            for ep_data in cached:
                key = (ep_data["ep"], ep_data.get("seed", 0))
                if key in existing_keys:
                    continue
                self._past_episodes.append(ep_data)
                existing_keys.add(key)
                added += 1
            if added:
                self._refresh_gallery_list()
                self._log_msg(f"Loaded {added} episode(s) from cache.", "ok")
            else:
                self._log_msg("No new episodes in cache.", "dim")
        except Exception as e:
            self._log_msg(f"Cache load failed: {e}", "err")

    # ── queue drain ───────────────────────────────────────────────────────────

    def _drain_queue(self):
        try:
            while True:
                fn, args, kwargs = self._q.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
        self.root.after(16, self._drain_queue)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 2.0)
    except Exception:
        pass
    app = GemmaRobotApp(root)
    root.mainloop()
