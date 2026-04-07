"""
GR00T Demo API — FastAPI backend
Endpoints: /submit, /next_job, /jobs/{id}, /jobs/{id}/complete, /health
"""
import sqlite3
import uuid
import time
import os
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from emailer import send_result_email

DB_PATH = os.environ.get("DB_PATH", "jobs.db")

app = FastAPI(title="GR00T Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_TASKS = {
    "open_drawer",
    "close_drawer",
    "place_in_closed_drawer",
    "pick_coke_can",
    "pick_object",
    "move_near",
}


# ── DB ────────────────────────────────────────────────────────────────────────

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                task        TEXT NOT NULL,
                email       TEXT NOT NULL,
                status      TEXT DEFAULT 'queued',
                created_at  REAL,
                result_url  TEXT,
                success     INTEGER,
                notified    INTEGER DEFAULT 0
            )
        """)
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


init_db()


# ── Models ────────────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    task: str
    email: str


class CompleteRequest(BaseModel):
    result_url: str
    success: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/submit")
def submit(req: SubmitRequest):
    if req.task not in VALID_TASKS:
        raise HTTPException(400, f"Unknown task '{req.task}'. Valid: {sorted(VALID_TASKS)}")
    job_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, task, email, created_at) VALUES (?, ?, ?, ?)",
            (job_id, req.task, req.email, time.time()),
        )
        position = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued'"
        ).fetchone()[0]
    return {"job_id": job_id, "position": position}


@app.get("/status/{job_id}")
def status(job_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return dict(row)


@app.get("/next_job")
def next_job():
    """Claim the next queued job atomically. Called by the worker."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return {"job": None}
        conn.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
    return {"job": dict(row)}


@app.patch("/jobs/{job_id}/complete")
def complete(job_id: str, req: CompleteRequest):
    """Mark a job done and send the result email."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        conn.execute(
            "UPDATE jobs SET status='done', result_url=?, success=?, notified=1 WHERE id=?",
            (req.result_url, int(req.success), job_id),
        )
    # Send email (non-blocking — failures are logged, not fatal)
    try:
        send_result_email(
            to=row["email"],
            task=row["task"],
            success=req.success,
            video_url=req.result_url,
        )
    except Exception as e:
        print(f"[email] Failed to send to {row['email']}: {e}")
    return {"status": "done"}


@app.patch("/jobs/{job_id}/fail")
def fail_job(job_id: str):
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
    return {"status": "failed"}


@app.get("/jobs")
def list_jobs(limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
