# GR00T Robot Demo — Interactive QR Code System

## Overview

Users scan a QR code (one per task), enter their email, and receive a link to the robot video + result once the eval completes.

```
QR Code (encodes task URL)
    ↓
GitHub Pages frontend  (static HTML/JS)
    ↓  POST {task, email}
Backend API            (FastAPI, publicly accessible)
    ↓  enqueue job
Job Queue              (SQLite DB on backend host)
    ↓  worker picks up
Robot Machine          (polls queue, runs SimplerEnv eval)
    ↓  uploads video + result
Cloud Storage          (R2/S3 or direct URL from robot)
    ↓  send email
Email Service          (SendGrid / Resend free tier)
    ↓
User inbox             (video link + success/fail badge)
```

---

## Component Breakdown

### 1. QR Codes (pre-generated, physical)

- One QR code per task (6 tasks = 6 QR codes)
- Each encodes a URL: `https://byyoung3.github.io/gr00t-demo/?task=open_drawer`
- Print and stick next to the physical robot / display

### 2. Frontend — GitHub Pages (`gh-pages` branch or `docs/` folder)

Single `index.html` with vanilla JS (no build step needed):

```
/docs
  index.html        # main page: shows task, email form
  tasks.json        # task metadata: display name, description, thumbnail
  qr/               # pre-generated QR code PNGs (one per task)
```

**Flow:**
1. Page loads, reads `?task=open_drawer` from URL
2. Looks up task metadata (pretty name, description, GIF preview)
3. Shows: "You're about to watch the robot **open a drawer**. Enter your email to receive the video."
4. User submits email → `POST https://api.yourhost.com/submit` with `{task, email}`
5. Shows confirmation: "We'll email you when it's done (usually ~5 min)"

### 3. Backend API — FastAPI

Hosted on a publicly reachable server. Options:
- **Railway / Render free tier** — easiest, free HTTPS, always on
- **The robot machine itself** with Cloudflare Tunnel (`cloudflared tunnel`) — zero port-forwarding
- **Fly.io** — free small instance

**Endpoints:**
```
POST /submit          {task: str, email: str}  → {job_id: str, position: int}
GET  /status/{job_id}                          → {status, result_url, success}
GET  /health                                   → 200 OK
```

**Database:** SQLite file `jobs.db`
```sql
CREATE TABLE jobs (
    id         TEXT PRIMARY KEY,
    task       TEXT NOT NULL,
    email      TEXT NOT NULL,
    status     TEXT DEFAULT 'queued',   -- queued | running | done | failed
    created_at REAL,
    result_url TEXT,
    success    INTEGER,
    notified   INTEGER DEFAULT 0
);
```

### 4. Robot Worker — Polling Loop

Runs on the robot machine (same machine that runs SimplerEnv):

```python
# scripts/demo_worker.py
# Polls backend API every 10s, picks up queued jobs, runs eval, uploads result
```

**Worker loop:**
1. `GET /next_job` — claim one queued job (atomic UPDATE … WHERE status='queued' LIMIT 1)
2. Run: `N_EPISODES=1 N_ENVS=1 TASK=<task> bash scripts/eval_single.sh`
3. Upload video to a pre-signed R2/S3 URL (or serve from robot via Cloudflare Tunnel)
4. `PATCH /jobs/{id}` with `{status: done, result_url, success}`
5. Backend sends email (or worker sends it directly)

**Alternative:** worker can be on the same host as the API (single-machine deploy), using Python threading or a simple task queue (e.g. `arq`, `rq`, or just a background thread).

### 5. Email — Resend or SendGrid

Both have free tiers (100 emails/day free on Resend).

Email template:
```
Subject: Your robot demo is ready! 🤖

Hi there,

The GR00T robot attempted to: **Open a Drawer**

Result: ✅ Success  (or ❌ Failed)

Watch the video: https://...

Powered by NVIDIA GR00T N1.6
```

---

## Deployment Plan

### Phase 1 — Local / Tunnel (fastest to demo)

```
Robot machine
├── FastAPI backend     (uvicorn, port 8000)
├── SQLite jobs.db
├── demo_worker.py      (background thread or separate process)
└── cloudflared tunnel  → https://gr00t-demo.yourdomain.workers.dev

GitHub Pages
└── frontend hits ^^ tunnel URL
```

Setup steps:
1. `pip install fastapi uvicorn resend`
2. `cloudflared tunnel --url http://localhost:8000` (free, no account needed for quick test)
3. `git subtree push --prefix docs origin gh-pages` to publish frontend
4. Generate QR codes pointing to `https://byyoung3.github.io/gr00t-demo/?task=<task>`

### Phase 2 — Cloud API (more reliable for demos)

Deploy API to Railway:
1. `railway init` in `demo_api/` folder
2. `railway up` — gets a persistent HTTPS URL
3. Robot machine polls the Railway API for new jobs
4. Worker runs locally, results emailed directly

---

## File Structure (new files to create)

```
Isaac-GR00T/
├── demo_api/
│   ├── main.py           # FastAPI app + job queue endpoints
│   ├── emailer.py        # Resend/SendGrid wrapper
│   ├── requirements.txt
│   └── jobs.db           # auto-created
├── scripts/
│   ├── demo_worker.py    # polls API, runs eval, reports back
│   └── eval_single.sh    # stripped-down single-task eval (reuses rollout_policy.py)
└── docs/                 # GitHub Pages frontend
    ├── index.html
    ├── tasks.json
    └── style.css
```

---

## Task List to Implement

1. **`demo_api/main.py`** — FastAPI with `/submit`, `/next_job`, `/jobs/{id}`, email trigger
2. **`demo_api/emailer.py`** — Resend integration with HTML template
3. **`scripts/demo_worker.py`** — polling loop that invokes the existing eval pipeline
4. **`scripts/eval_single.sh`** — thin wrapper for a single-task, single-episode eval (returns video path + result JSON)
5. **`docs/index.html`** — static frontend: reads `?task=`, email form, polls `/status/{id}`
6. **`docs/tasks.json`** — task metadata (name, description, preview image URL)
7. **QR generation script** — `python scripts/gen_qr_codes.py` outputs one PNG per task
8. **Deploy** — Cloudflare Tunnel or Railway, update `API_URL` constant in `index.html`

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Frontend hosting | GitHub Pages | Free, no infra, instant |
| API hosting | Cloudflare Tunnel (demo) → Railway (prod) | Tunnel = zero setup; Railway = more reliable |
| Queue | SQLite on API host | Simple, no Redis needed at this scale |
| Email | Resend (free 100/day) | Simple REST API, good deliverability |
| Video storage | Serve from robot via tunnel URL | Avoids S3 setup for demo |
| Auth/rate-limiting | None for demo, email as natural gate | Keep it simple |

---

## QR Code → Email Flow (end-to-end)

```
[QR: ?task=open_drawer]
         ↓ scan
[Page loads, shows "Open Drawer" task]
         ↓ user types email, clicks Submit
[POST /submit {task: open_drawer, email: user@x.com}]
         ↓ returns {job_id: abc123, position: 2}
[Page shows "You're #2 in queue, ~4 min"]
         ↓ (optional: page polls GET /status/abc123 every 15s)
[Worker picks up job, runs eval_single.sh]
         ↓ 1 episode, ~2 min
[Worker PATCH /jobs/abc123 {status: done, result_url, success: true}]
         ↓
[API triggers Resend email to user@x.com]
         ↓
[User gets email with video link + ✅/❌ result]
```
