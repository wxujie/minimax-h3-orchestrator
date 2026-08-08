# Operations

Run and operate the MiniMax-H3 orchestrator safely on a pool of Kaggle
notebooks + Cloudflare tunnels.

## Environment reference

Set these in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONTROLLER_HOST` / `PORT` | `0.0.0.0` / `8000` | Controller HTTP bind |
| `DATABASE_URL` | `sqlite:///./storage/controller.db` | SQLAlchemy URL (Postgres-ready) |
| `STORAGE_DIR` | `./storage` | Uploads + artifacts root |
| `WORKFLOW_PATH` | `./workflows/workflow.json` | Workflow adapter source |
| `NOTEBOOK_PATH` | `./notebooks/….ipynb` | Notebook pushed to Kaggle |
| `MAX_JOB_RETRIES` | `2` | Attempts before a job is marked FAILED |
| `JOB_OUTPUT_RETENTION_HOURS` | `24` | Artifact sweep age |
| `MAX_CONCURRENT_NOTEBOOKS` | `2` | Pool-wide provisioning cap |
| `SCHEDULER_POLL_S` / `JOB_POLL_S` | `5.0` / `3.0` | Scheduler + job poll cadence |
| `WORKER_HEALTH_TIMEOUT_S` / `HEARTBEAT` | `30` / `60` | Health probing |
| `NOTEBOOK_START_WAIT_S` | `30` | Time to wait for a notebook to come up |
| `WORKER_AUTH_SECRET` | — | Shared controller↔worker Bearer secret |
| `WORKER_AUTH_REQUIRED` | `true` | Gate every worker-agent endpoint |
| `CLOUDFLARE_TOKEN` | — | Only for `named` tunnel mode |
| `KAGGLE_ACCOUNT_<N>_USERNAME/KEY/ENABLED` | — | One block per Kaggle account |

## Running the controller

```bash
source .venv/bin/activate
uvicorn controller.main:app --host 0.0.0.0 --port 8000
```

The scheduler runs in a background thread inside the app (`Scheduler.start()`).
Dashboard: **`/dashboard/`**. Health/API under `/api/v1/*`.

## Worker agent inside a notebook

The notebook loads `worker/agent.py`, builds a `WorkerSpec` from its own env,
starts the agent's FastAPI `app`, opens a Cloudflare tunnel, and registers with
the controller:

```
POST /api/v1/agents/register
{ worker_id, notebook_id, gpu, comfy_port, comfy_url, tunnel_url }   # Bearer secret
```

Config that the agent reads:
- `WORKER_AUTH_SECRET` — match the controller
- `TUNNEL_MODE=quick|named`, `TUNNEL_DOMAIN` (named mode), `CLOUDFLARED_BIN`

## Tunnels

**Quick (default, zero-config).** `cloudflared tunnel --url http://127.0.0.1:<agent_port>`
yields a random `.trycloudflare.com` URL parsed from the log. Best for ephemeral
notebooks; URLs change on every run and the controller reconciles via `/health`.

**Named (production).** Requires `CLOUDFLARE_TOKEN` + `TUNNEL_DOMAIN`. The public
URL is deterministic (`https://<worker-id>.<domain>`) so the controller can derive
it and reconnect the tunnel after notebook restarts.

In both modes only the **agent** is exposed — ComfyUI stays local on the GPU.

## Kaggle capacity reality

The Kaggle API has **no** official endpoints for GPU-quota, session budget, or
kernel *terminate*. The manager therefore:
- reports capacity as `UNKNOWN` and stays conservative (never exceed one notebook
  per account, and a pool-wide `max_concurrent_notebooks`);
- treats a failed `ensure_notebook` push as `QUOTA_EXHAUSTED`, falling to the
  next account;
- cannot stop an idle notebook programmatically — it waits for Kaggle's own
  runtime limits.

## Reliability & failover

- **Worker loss** mid-job → job requeued as TRANSIENT onto another READY worker.
- **Submit failure** → worker marked ERROR, requeue.
- **PERMANENT errors** (workflow validation, bad duration) → job FAILED, no waste.
- Retries are bounded by `attempt_count <= max_retries`.

Tune `MAX_CONCURRENT_NOTEBOOKS` to cap spend, `MAX_JOB_RETRIES` to bound wasted
GPU minutes, and `JOB_OUTPUT_RETENTION_HOURS` to bound disk.

## Security hardening

1. Use a long, random `WORKER_AUTH_SECRET`; never commit `.env`.
2. `WORKER_AUTH_REQUIRED=true` and keep `/agent register` behind the token.
3. Run the controller on a private host/network or behind an auth proxy if it
   will be internet-facing.
4. Secrets are scrubbed from logs by `RedactingFormatter`; treat account
   username/key as sensitive and don't log them manually.

## Maintenance

- `python -m pytest tests/ -q` for the full in-process suite (no cloud).
- `storage/artifacts` is swept automatically; schedule manual cleanup for
  completed runs you want to back up.
- Watch `/api/v1/system/status` or the dashboard for `ready_workers`/`queued_jobs`
  drift.

> Run within your own quota/terms, this tool does not work around Kaggle rate
> limits; it simply spreads jobs across the accounts you own and fails cleanly
> when capacity is exhausted.