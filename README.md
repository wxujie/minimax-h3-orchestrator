# MiniMax H3 Orchestrator

Build **MiniMax-H3** image-to-video clips across a pool of free-tier **Kaggle
GPU notebooks**, all behind one plain HTTP controller. A notebook runs two
ComfyUI workers (one per GPU); each worker is reached by the controller through
a private Cloudflare tunnel. As a client you only ever talk to the controller —
Kaggle accounts, GPU notebooks, and tunnels are fully abstracted away.

```
 ┌────────────┐   HTTP api/v1    ┌────────────────────────────────────┐
 │  Client    │ ───────────────▶ │  Controller (FastAPI)              │
 │ (SDK/curl) │ ◀─────────────── │  • queue video jobs                │
 └────────────┘   video bytes    │  • lazily provision Kaggle notebooks│
                                 │  • dispatch to READY workers       │
                                 └───────┬──────────────┬─────────────┘
                                         │ tunnel URL   │ tunnel URL
                                         ▼              ▼
                             ┌────────────────┐   ┌────────────────┐
                             │ Kaggle nb A    │   │ Kaggle nb B    │
                             │  worker-0 agent│   │  worker-0 agent│
                             │  worker-1 agent│   │  worker-1 agent│
                             │   · 2× ComfyUI │   │   · 2× ComfyUI │
                             └────────────────┘   └────────────────┘
```

---

## Table of contents

1. [What it is](#what-it-is)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Install](#install)
5. [Configure](#configure)
6. [Run the controller](#run-the-controller)
7. [Submit a job — Python SDK](#submit-a-job--python-sdk)
8. [Submit a job — raw REST / curl](#submit-a-job--raw-rest--curl)
9. [Live dashboard](#live-dashboard)
10. [How provisioning works](#how-provisioning-works)
11. [Running the worker agent](#running-the-worker-agent)
12. [Testing](#testing)
13. [Documentation](#documentation)
14. [Security notes](#security-notes)
15. [Limits & troubleshooting](#limits--troubleshooting)

---

## What is it

- **Controller** — a FastAPI app that queues video jobs, lazily spins up Kaggle
  notebooks when no worker is ready, dispatches each job to a READY worker, and
  serves the finished `.mp4` back.
- **Worker agent** — a small authenticated HTTP server that runs inside each
  Kaggle notebook next to ComfyUI, drives ComfyUI's `/prompt` API, and is reached
  by the controller over a Cloudflare tunnel.
- **Python SDK & live dashboard** — the two friendly ways to drive the controller.

Design facts worth knowing up front:

- **Two GPUs per notebook.** Each notebook runs `GPU_PER_NOTEBOOK` (2) ComfyUI
  workers, so one notebook renders two jobs concurrently.
- **Lazy provisioning.** A notebook is started only when it's actually needed —
  when a job arrives and no READY worker exists. Nothing is pre-armed.
- **Automatic failover.** If a worker dies mid-render its job is requeued onto
  another ready worker; retries are bounded by `MAX_JOB_RETRIES`.
- **No secrets in the repo.** Kaggle keys and the worker auth secret live only
  in your `.env` (git-ignored), and only in memory at runtime.

---

## Repository layout

| Path | What it is |
|------|------------|
| `controller/` | Scheduler, job/worker/account registries, storage, workflow adapter, Kaggle + FastAPI layer |
| `worker/` | Per-GPU agent that runs inside a notebook (ComfyUI client, Cloudflare tunnel) |
| `client/sdk.py` | Python SDK for creating / polling / downloading video jobs |
| `web/index.html` | Live dashboard, served at `/dashboard/` |
| `workflows/workflow.json` | The `MiniMaxH3ImageToVideo` ComfyUI subgraph (editable JSON) |
| `notebooks/minimax-h3-comfyui.ipynb` | The notebook pushed to Kaggle |
| `tests/` | 36 in-process tests: scheduler, worker agent, workflow, storage, API, SDK |

---

## Requirements

- Python **3.11+**
- One or more free **Kaggle accounts** with API keys ([settings](https://www.kaggle.com/settings))
- **Outbound internet** on the controller host (Kaggle API + Cloudflare)

> More accounts → the pool finishes jobs faster. The tool only uses your own
> accounts; it never works around Kaggle rate limits.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configure

```bash
cp .env.example .env
```

Edit `.env`. The two things you *must* set before anything actually renders:

| Variable | What to put |
|----------|-------------|
| `KAGGLE_ACCOUNT_1_USERNAME` | Your Kaggle username |
| `KAGGLE_ACCOUNT_1_KEY` | Your Kaggle API key |
| `WORKER_AUTH_SECRET` | A long random string shared between controller and agents; protects the agent→controller `/agents/register` handshake |

Add `KAGGLE_ACCOUNT_2_*`, `KAGGLE_ACCOUNT_3_*`, … blocks for more accounts — the
count is auto-discovered. Every other setting has a sensible default (host,
ports, poll intervals, retries, retention); see `.env.example` and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

---

## Run the controller

```bash
source .venv/bin/activate
uvicorn controller.main:app --reload --port 8000
```

- Dashboard: **http://localhost:8000/dashboard/**
- API root: **http://localhost:8000/api/v1/**

The scheduler runs in a background thread automatically — you don't start it
separately.

> The controller happily serves a queue with **zero** notebooks running. The
> moment a job arrives, it lazily provisions a notebook on your account.

---

## Submit a job — Python SDK

```python
from client.sdk import VideoClient

# The client-facing REST API is not token-gated; run the controller on a
# private network or behind an auth reverse proxy (see Security notes).
c = VideoClient("http://localhost:8000")

job = c.create_video(
    prompt="a raccoon plays the drums on a rooftop",
    duration=2.0,          # seconds → snapped to the nearest 24 fps frame count
    # image="first.png",     # optional first frame
    # last_image="last.png", # optional last frame
    # width=1024, height=576,
    # seed=123, priority=0,
)
print("queued:", job.job_id)

done = c.wait_for_result(job.job_id, timeout=1800)   # blocks until COMPLETED
path = c.download(job.job_id, "out.mp4")
print("video saved:", path)
```

Also on `VideoClient`:

- `get_job(id)` — poll status (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`,
  `CANCELLED`) plus `progress` / `error` / `download_url`.
- `cancel(id)` — cancel a queued/running job.
- `system_status()`, `workers()`, `accounts()` — pool introspection.

---

## Submit a job — raw REST / curl

The API is JSON first; include an optional first/last frame via **multipart**.
Everything lives under `/api/v1/`.

**Create (JSON, no frames):**

```bash
curl -sS -X POST http://localhost:8000/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a neon cat runs through a city","duration":4.0}'
# {"job_id":"job_ab12cd34ef56","position":0,"status":"QUEUED"}
```

**Create (multipart, with frames):**

```bash
curl -sS -X POST http://localhost:8000/api/v1/jobs/multipart \
  -F 'prompt=a neon cat runs through a city' \
  -F 'duration=4.0' \
  -F 'first_frame=@first.png'
```

**Poll, download, cancel:**

```bash
curl -sS http://localhost:8000/api/v1/jobs/job_ab12cd34ef56             # status/progress
curl -sS -o out.mp4 http://localhost:8000/api/v1/jobs/job_ab12cd34ef56/result
curl -sS -X POST http://localhost:8000/api/v1/jobs/job_ab12cd34ef56/cancel
```

**System + pool:**

```bash
curl -sS http://localhost:8000/api/v1/system/status   # ready_workers / queued_jobs / …
curl -sS http://localhost:8000/api/v1/workers
curl -sS http://localhost:8000/api/v1/accounts
```

The client-facing endpoints are **not** token-gated today: `WORKER_AUTH_SECRET`
shields the agent→controller registration handshake only. Protect the controller
with a private network or an auth reverse proxy before exposing it (see
[Security notes](#security-notes)). The finished `.mp4` is streamed from
`GET /jobs/{id}/result` only once the job is `COMPLETED` (a `409` otherwise).

---

## Live dashboard

Open http://localhost:8000/dashboard/ in a browser. It polls the same endpoints
and shows at a glance how many workers are **READY / BUSY / OFFLINE**, how many
jobs are **QUEUED / RUNNING / COMPLETED / FAILED**, and each Kaggle account's
state — without ever printing a credential.

---

## How provisioning works

The controlling idea is **lazy, conservative** provisioning:

1. A job arrives → the scheduler looks for a **READY** worker.
2. Worker free? → dispatch immediately.
3. No worker free → start a notebook on the next enabled, non-quota account,
   unless the pool is already at `MAX_CONCURRENT_NOTEBOOKS`.
4. When the notebook boots, its worker agent opens a Cloudflare tunnel and tells
   the controller `POST /api/v1/agents/register` (Bearer-authed).
5. The controller marks the two placeholder workers **READY** and drains the queue.

Scale **up** by adding more `KAGGLE_ACCOUNT_N_*` blocks; cap spend by lowering
`MAX_CONCURRENT_NOTEBOOKS`. The scheduler stays conservative because the Kaggle
API does not expose real-time GPU quota.

---

## Running the worker agent

The worker side isn't a separate install — it's the notebook pushed to Kaggle
(`notebooks/minimax-h3-comfyui.ipynb`). Inside, the agent reads the env the
controller set, points at its local ComfyUI, starts its own small FastAPI server,
opens a `quick` (auto `trycloudflare.com`) or `named` tunnel, and registers with
the controller.

`TUNNEL_MODE=quick` is zero-config for a private experiment — the controller
reconciles the random URL on each `/health`. For production, use `TUNNEL_MODE=named`
with `TUNNEL_DOMAIN` and `CLOUDFLARE_TOKEN` so the tunnel URL is deterministic and
survives notebook restarts. Tunnels expose only the agent — ComfyUI stays local
on the GPU. See [`docs/OPERATIONS.md`](docs/OPERATIONS.md#tunnels).

---

## Testing

The whole suite runs **in-process — no network, no Kaggle, no GPU**:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

It exercises the scheduler (dispatch / failover / lazy provisioning), the worker
agent (fake ComfyUI + fake tunnel), the workflow adapter against the real
`workflows/workflow.json`, storage path-safety + retention, and the API + SDK.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, data model, job &
  worker state machines, the scheduler loop, retry policy.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — every `.env` key, tunnels (quick vs
  named), Kaggle capacity nuances, security hardening, maintenance.

---

## Security notes

- **Secrets never touch the repo.** `.env`, `.env.local`, `storage/` and `*.db`
  are git-ignored (see `.gitignore`). Only the `.env.example` template is committed.
- **In memory only.** Kaggle keys and `WORKER_AUTH_SECRET` live only in
  `config.Settings()` and are scrubbed from every log line by the redacting formatter.
- **Agent → controller handshake.** `POST /api/v1/agents/register` is
  Bearer-protected with `WORKER_AUTH_SECRET`; the agent's `GET /health` is
  deliberately unauthenticated (liveness).
- **Client-facing REST API is not token-gated today.** The controller's `/jobs`,
  `/workers`, `/accounts`, and `/system` endpoints accept requests without
  authentication regardless of `WORKER_AUTH_SECRET`. Treat the controller as
  private: run it on a trusted network or behind an auth reverse proxy.

---

## Limits & troubleshooting

| Symptom | Usual cause → fix |
|---------|-------------------|
| Jobs stay `QUEUED`, no workers appear | Kaggle quota exhausted / provisioning slow → check `system/status`, add accounts, raise `MAX_CONCURRENT_NOTEBOOKS`. |
| Job `RUNNING` then requeues | A worker died mid-render — that is the failover boundary; TRANSIENT errors retry automatically. |
| Job `FAILED` with a validation message | PERMANENT error (bad prompt/duration/workflow) — fix the input and resend. |
| Editing `workflow.json` breaks jobs | The adapter validates exact node ids/types against the `MiniMaxH3ImageToVideo` subgraph — revert + see `docs/ARCHITECTURE.md`. |
| `GET /result` returns `409` | The job isn't `COMPLETED` yet — poll first. |

---

> Use this with your own Kaggle accounts and within your own quota/terms. It does
> not — and will not — bypass Kaggle's rate limits; it simply spreads jobs across
> the accounts you own and fails cleanly when capacity is exhausted.