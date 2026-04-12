"""
control_server.py — runs on H100, port 8090

Wraps sim_server.py as a managed subprocess. Tunneled via ngrok.
All /reset /step /step_cuda /predict /info calls are proxied to sim_server.
Management endpoints are under /control/*.

Usage:
    cd Isaac-GR00T
    PYTHONPATH=. .venv/bin/python remote_eval/control_server.py \
        --sim_cmd ".venv/bin/python remote_eval/sim_server.py --env_name simpler_env_google/google_robot_close_drawer --port 8080 --no_policy" \
        --sim_port 8080 \
        --port 8090

Then expose via ngrok:
    ngrok http 8090
"""

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# ── global state ──────────────────────────────────────────────────────────────
app = FastAPI()

_sim_proc: subprocess.Popen | None = None
_sim_start_time: float = 0.0
_sim_cmd: list[str] = []
_sim_port: int = 8080
_sim_log_path: str = "/tmp/gr00t_sim_server.log"
_sim_env: dict = {}

PROXY_PATHS = {"/reset", "/step", "/step_cuda", "/predict", "/info"}


# ── sim process management ────────────────────────────────────────────────────

def _start_sim():
    global _sim_proc, _sim_start_time
    if _sim_proc and _sim_proc.poll() is None:
        _sim_proc.terminate()
        try:
            _sim_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _sim_proc.kill()

    log_fh = open(_sim_log_path, "w")
    print(f"[control] Starting sim_server: {' '.join(_sim_cmd)}")
    _sim_proc = subprocess.Popen(
        _sim_cmd,
        stdout=log_fh,
        stderr=log_fh,
        env=_sim_env,
        start_new_session=True,
    )
    _sim_start_time = time.time()
    print(f"[control] sim_server PID={_sim_proc.pid}")
    return _sim_proc.pid


def _sim_status() -> dict:
    running = _sim_proc is not None and _sim_proc.poll() is None
    uptime = round(time.time() - _sim_start_time, 1) if running else 0

    # Try to fetch /info from sim_server for live env/step info
    sim_info = {}
    if running:
        try:
            r = httpx.get(f"http://localhost:{_sim_port}/info", timeout=2.0)
            if r.status_code == 200:
                sim_info = r.json()
        except Exception:
            pass

    return {
        "running": running,
        "pid": _sim_proc.pid if _sim_proc else None,
        "uptime_seconds": uptime,
        "env": sim_info.get("env"),
        "episode_steps": sim_info.get("step"),
        "sim_ready": bool(sim_info),
    }


# ── proxy ─────────────────────────────────────────────────────────────────────

async def _proxy(request: Request, path: str) -> Response:
    url = f"http://localhost:{_sim_port}/{path.lstrip('/')}"
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                content=body,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host", "content-length")},
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )
    except httpx.ConnectError:
        return JSONResponse({"error": "sim_server not reachable — try /control/restart"}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── sim passthrough endpoints ─────────────────────────────────────────────────

@app.get("/info")
async def info(request: Request):
    return await _proxy(request, "/info")

@app.post("/reset")
async def reset(request: Request):
    return await _proxy(request, "/reset")

@app.post("/step")
async def step(request: Request):
    return await _proxy(request, "/step")

@app.post("/step_cuda")
async def step_cuda(request: Request):
    return await _proxy(request, "/step_cuda")

@app.post("/predict")
async def predict(request: Request):
    return await _proxy(request, "/predict")


# ── control endpoints ─────────────────────────────────────────────────────────

@app.get("/control/status")
async def control_status():
    """Return sim_server health: running, pid, uptime, env, episode_steps."""
    return JSONResponse(_sim_status())


@app.post("/control/restart")
async def control_restart():
    """Kill and restart sim_server subprocess. Returns new status after 3s."""
    pid = _start_sim()
    # Give it a moment to start listening
    await asyncio.sleep(3)
    status = _sim_status()
    status["restarted_pid"] = pid
    return JSONResponse(status)


@app.get("/control/logs")
async def control_logs(n: int = 100):
    """Return last N lines of sim_server log."""
    try:
        lines = Path(_sim_log_path).read_text(errors="replace").splitlines()
        return JSONResponse({"lines": lines[-n:], "total": len(lines)})
    except FileNotFoundError:
        return JSONResponse({"lines": [], "total": 0})


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_cmd", required=True,
                        help="Shell command to launch sim_server (quoted string)")
    parser.add_argument("--sim_port", type=int, default=8080,
                        help="Port sim_server listens on")
    parser.add_argument("--port", type=int, default=8090,
                        help="Port this control server listens on")
    parser.add_argument("--sim_log", default="/tmp/gr00t_sim_server.log",
                        help="Path for sim_server stdout/stderr log")
    parser.add_argument("--no_autostart", action="store_true",
                        help="Don't auto-start sim_server on launch")
    args = parser.parse_args()

    _sim_port = args.sim_port
    _sim_log_path = args.sim_log
    _sim_cmd = shlex.split(args.sim_cmd)
    _sim_env = {
        **os.environ,
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "PYTHONPATH": f"{Path(__file__).resolve().parent.parent}:{os.environ.get('PYTHONPATH', '')}",
    }

    if not args.no_autostart:
        _start_sim()
        print(f"[control] Waiting for sim_server on port {_sim_port}...")
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                httpx.get(f"http://localhost:{_sim_port}/info", timeout=2.0)
                print(f"[control] sim_server ready.")
                break
            except Exception:
                time.sleep(2)

    print(f"[control] Control server on port {args.port}")
    print(f"[control] Endpoints: /control/status  /control/restart  /control/logs")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
