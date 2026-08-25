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
| `controller/` | Scheduler, job/worker/account registries, storage, workflow adapters, Kaggle + FastAPI layer |
| `worker/` | Per-GPU agent that runs inside a notebook (ComfyUI client, Cloudflare tunnel) |
| `client/sdk.py` | Python SDK for creating / polling / downloading video jobs |
| `web/index.html` | Live dashboard, served at `/dashboard/` |
| `workflows/workflow.json` | The `MiniMaxH3ImageToVideo` ComfyUI subgraph (editable JSON) |
| `workflows/workflow_r2v.json` | The `MiniMaxH3ReferenceToVideo` ComfyUI template (read by the R2V adapter) |
| `notebooks/minimax-h3-comfyui.ipynb` | The notebook pushed to Kaggle |
| `tests/` | In-process tests: scheduler, worker agent, workflow (FL2VA + R2V), storage, API, SDK |

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

Edit `.env`. The things you *must* set before anything actually renders:

| Variable | What to put |
|----------|-------------|
| `KAGGLE_ACCOUNT_1_USERNAME` | Your Kaggle username |
| `KAGGLE_ACCOUNT_1_KEY` | Your Kaggle API key |
| `WORKER_AUTH_SECRET` | A long random string shared between controller and agents; protects the agent→controller `/agents/register` handshake |
| `CONTROLLER_PUBLIC_URL` | Publicly reachable root of **this** controller, e.g. `https://<host>:8000`. The pushed Kaggle notebook clones the repo and POSTs `/api/v1/agents/register` here when it boots, so it must be reachable **from Kaggle** (a deployed host or a controller-side tunnel). Without it, the scheduler won't start notebooks — a notebook it can't reach could never round-trip. |

`ORCHESTRATOR_REPO_URL` (default `https://github.com/msaadakram/minimax-h3-orchestrator`)
is what the notebook clones to load the worker package + workflow — only override it
if you fork the repo. Add `KAGGLE_ACCOUNT_2_*`, `KAGGLE_ACCOUNT_3_*`, … blocks for
more accounts — the count is auto-discovered. Every other setting has a sensible
default (host, ports, poll intervals, retries, retention); see `.env.example` and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

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

## Reference-to-Video (R2V) with Turbo LoRA

In addition to the default first/last-frame image-to-video mode (FL2VA), the
orchestrator ships a **Reference-to-Video** workflow backed by the official
`MiniMaxH3ReferenceToVideo` node. R2V locks a character / style / motion from
up to 3 reference images, and supports an optional **Turbo LoRA** that cuts
sampling from 20 steps to 4 steps.

### Workflow selection

Set `"workflow": "minimax-h3-r2v"` in the job JSON. The controller, worker,
and the `R2VWorkflowAdapter` (`controller/workflow_r2v.py`) then build the
flat ComfyUI graph for the R2V node instead of the FL2VA one. The two modes
use **different diffusion models**:

| Mode | Core node | Unet |
|------|-----------|------|
| FL2VA (default) | `MiniMaxH3ImageToVideo` | `minimax_h3_fl2va_pruned_int8_convrot` |
| R2V | `MiniMaxH3ReferenceToVideo` | `minimax_h3_ref2va_pruned_int8_convrot` |

The R2V notebook cells download both the ref2va unet and the Turbo LoRA into
`models/diffusion_models` and `models/loras` respectively.

### Creating an R2V job (JSON)

```bash
curl -sS -X POST http://localhost:8000/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "workflow": "minimax-h3-r2v",
    "prompt": "anime cel-shaded style, the hero in the reference image leaps across neon rooftops, dynamic action, no text",
    "duration": 5.0,
    "turbo": true,
    "ref_images": ["anime_ref.png"]
  }'
```

`ref_images` holds filenames already staged under `storage/uploads/` (upload
them with the `/jobs/multipart` endpoint, or drop them into the uploads dir).
Up to 3 are wired into the R2V node as `ref_images.ref_image_0..2`. The prompt
may reference them with `<Picture 1>`, `<Picture 2>`, … tags.

### Turbo mode

`turbo` routes the unet through `LoraLoaderModelOnly` with the official
`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` LoRA and a
`ComfySwitchNode`, dropping sampling from 20 steps to 4 (`turbo_steps`,
default 4). Optional knobs:

- `turbo_steps` — step count in turbo mode (default 4)
- `turbo_lora_strength` — LoRA strength (default 1.0)
- `ref_image_size` — `match` (faster, default) or `max` (best identity fidelity, several× slower)

Turbo trades a little audio/motion quality for roughly 2× speed.

### T4-safe default resolution

The ref2va unet (≈20 GB int8) plus the qwen3vl CLIP, dual VAEs, and audio
decoder exceed a **Tesla T4's 15 GB** VRAM at the native 1344×768 canvas,
causing OOM / `no video output produced`. The R2V adapter therefore defaults
to **832×480** when no `width`/`height` is passed. Explicit `width` + `height`
still allow full resolution on larger GPUs (T4×2, A100, …). The same T4 safe
size applies to FL2VA for consistency.

> **Deploy note.** The Kaggle worker clones this repo once at notebook boot;
> it does **not** hot-reload pushed code changes. Any change to the adapter or
> worker logic requires restarting the Kaggle kernel (which re-downloads the
> models). Controller-side fixes, however, take effect on controller restart.

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
4. At push time the controller builds the notebook (the ComfyUI bootstrap +
   [`worker/runner.py`](worker/runner.py) as the last cell). When it boots, the
   runner cell clones the orchestrator repo, injects this notebook's identity
   (`NOTEBOOK_ID` / `CONTROLLER_PUBLIC_URL` / `WORKER_AUTH_SECRET`), serves a
   small authenticated agent per GPU, opens a Cloudflare tunnel, and calls
   `POST /api/v1/agents/register` (Bearer-authed).
5. The controller verifies each agent's `/health` through the tunnel, marks the
   placeholder workers **READY**, and drains the queue — now handing the job to
   the agent, which drives its local ComfyUI to render the video.

Scale **up** by adding more `KAGGLE_ACCOUNT_N_*` blocks; cap spend by lowering
`MAX_CONCURRENT_NOTEBOOKS`. The scheduler stays conservative because the Kaggle
API does not expose real-time GPU quota.

---

## Running the worker agent

The worker side isn't a separate install — it's the notebook pushed to Kaggle
(`notebooks/minimax-h3-comfyui.ipynb`). At push time the controller appends
[`worker/runner.py`](worker/runner.py) to that notebook. When the notebook
boots, the runner reads the identity the controller embedded (`NOTEBOOK_ID`,
`CONTROLLER_PUBLIC_URL`, `WORKER_AUTH_SECRET`), clones the orchestrator repo for
the worker package + workflow adapter, serves a small authenticated FastAPI
agent per GPU, opens a `quick` (auto `trycloudflare.com`) or `named` tunnel, and blocks to keep
the session alive so the controller can keep dispatching jobs through it. It
registers each worker with the controller, then re-registers on a recycled quick
tunnel so the worker rejoins automatically.

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