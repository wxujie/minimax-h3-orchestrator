# Architecture

This is the design of the MiniMax-H3 orchestrator: how video jobs flow through
a pool of lazily-provisioned Kaggle notebooks running ComfyUI, and back.

## Components

| Component | File | Owns |
|-----------|------|------|
| Scheduler | `controller/scheduler.py` | The loop that dispatches, monitors, and completes jobs |
| Job registry | `controller/jobs.py` | Job object + state machine, attempts, retry policy |
| Worker registry | `controller/workers.py` | Worker rows, heartbeat, READY/BUSY/OFFLINE transitions |
| Account registry | `controller/accounts.py` | Configured Kaggle credentials; never exposed |
| Kaggle manager | `controller/kaggle_manager.py` | Thin `kaggle` CLI wrapper: capacity, push/status/output |
| Storage | `controller/storage.py` | Uploads + per-job artifacts, retention sweep |
| Workflow adapter | `controller/workflow.py` | `workflow.json` → flat ComfyUI `/prompt` graph |
| Worker client | `controller/worker_client.py` | HTTP client for one worker agent's tunnel |
| Worker agent | `worker/agent.py` | In-notebook authenticated HTTP server per ComfyUI |
| ComfyUI client | `worker/comfy_client.py` | ComfyUI upload/submit/poll/history/download |
| Cloudflare tunnel | `worker/cloudflare.py` | Quick or named cloudflared tunnel to the agent |

## Data model

- **Account** — one Kaggle login. GPU quota is `UNKNOWN` by policy (the Kaggle
  API does not expose it); scheduling stays conservative (see *Provisioning*).
- **Notebook** — a Kaggle kernel hosting `GPU_PER_NOTEBOOK` (2) GPUs.
- **Worker** — one GPU/ComfyUI instance on a notebook, id `<notebook>-gpu<index>`.
- **Job** — a video-render request plus its attempts.
- **JobAttempt** — one dispatch of a job to one worker (records the remote ComfyUI
  `prompt_id`).
- **Artifact** — a stored input/workflow/output file.

## Job state machine

```
 QUEUED --start_attempt--> STARTING --set RUNNING--> RUNNING --poll DONE--> COMPLETED
   ▲                                                  │
   │ requeue                                          │ poll FAILED (attempt <= retries)
   │                                                  ▼
   └------------ RETRYING <---- fail TRANSIENT     FAILED <---- fail PERMANENT
                                                          or retries exhausted

 CANCELLED <---- cancel (from QUEUED/STARTING/RUNNING)
```

Transitions are driven by the scheduler; the registry (`jobs.py`) only persists
them. `start_attempt()` increments `attempt_count` and leaves the assigned
worker intact so the monitor can poll it.

## Worker state machine

```
UNREGISTERED ──register──▶ WORKER_STARTING ──health ok──▶ WORKER_READY
     ▲                                                      │  dispatch
     │                                                     ▼
     └────────── mark_offline ◀── heartbeat lost ────── WORKER_BUSY ─mark_idle──▶ READY
```

A READY worker with a live `/health` through its tunnel is a dispatch candidate.
A hard health failure marks the worker `OFFLINE`; any job it was running is
requeued as TRANSIENT.

## The scheduler loop

Each `Scheduler.tick()`:

1. **heartbeat_all()** — probe each worker's health; retire dead ones.
2. **monitor_running()** — poll each `RUNNING` job; on `DONE` download the video
   and complete it; on `FAILED` record+retry.
3. **dispatch_job()** — run the next queued job on a READY worker, or else
   provision a notebook lazily.

### Dispatch (`_run_on`)

- `assign()` worker + `start_attempt()`
- `_submit()`: stage the first/last-frame files to the worker, then `client.submit()`
- on success: mark job `RUNNING`, worker `BUSY`, remember the ComfyUI `prompt_id`
- on `WorkerClientError` (TRANSIENT): mark the worker errored and `_maybe_retry`

### Provisioning (no READY worker)

- `_can_provision()`: allow a new notebook only while
  `starting/running notebooks + in_flight < max_concurrent_notebooks`.
- `_account_to_start()`: first **enabled**, non-quota accounts with no notebook
  currently starting/running.
- `_lazy_start()`: insert a `NOTEBOOK_STARTING` row, call
  `provider.start_notebook(...)`, and on success create the 2 placeholder workers
  via `provision()`. On failure the notebook becomes `QUOTA_EXHAUSTED`.

The real provider is `controller/main.py::KaggleProvider` around
`kaggle_manager` (capacity check → `ensure_notebook` push).

## Retry / failure policy

- **TRANSIENT** — network, tunnel loss, worker crash, 5xx: retry on another
  worker. `JobManager.retryable()` returns true while the error class is
  TRANSIENT/NONE and `attempt_count <= max_retries`.
- **PERMANENT** — deterministic validation/workflow errors: fail immediately.

## Workflow → ComfyUI

`WorkflowAdapter` reads the editable `workflow.json`, validates the exact node
ids and types against the `MiniMaxH3ImageToVideo` subgraph, and emits the flat
API prompt: loaders (UNET/CLIP/VAE), `LoadImage` frames, the core
`MiniMaxH3ImageToVideo`, `RandomNoise`, the sampler chain, `CreateVideo`,
`SaveVideo`. Duration is grid-snapped at 24 fps (`duration_to_frames`). The
worker uploads the frames, then `POST /prompt` with this graph.

## Storage

`storage/uploads/` holds client frames; `storage/artifacts/<job_id>/{input,output}`
holds per-job files. Filenames are sanitized and every write is contained inside
the job directory (no path traversal). `sweep()` deletes completed jobs older than
`JOB_OUTPUT_RETENTION_HOURS`.

## Security

Secrets live only in memory; `config.Settings.log_secrets()` enumerates every
username/key/token and the `RedactingFormatter` scrubs them from every log line.
Controller↔worker traffic is Bearer-authed with `WORKER_AUTH_SECRET`. The worker
agent's `/health` is deliberately unauthenticated (it exposes only liveness).