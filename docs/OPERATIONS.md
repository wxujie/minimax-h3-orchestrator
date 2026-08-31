# Operations

Run and operate the MiniMax-H3 orchestrator safely on a pool of Kaggle
notebooks (2×T4) + Colab sessions (1×T4) behind Cloudflare tunnels.

## Environment reference

Set these in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONTROLLER_HOST` / `PORT` | `0.0.0.0` / `8001` | Controller HTTP bind（8000 被无关 node 进程占用） |
| `DATABASE_URL` | `sqlite:///./storage/controller.db` | SQLAlchemy URL (Postgres-ready) |
| `STORAGE_DIR` | `./storage` | Uploads + artifacts root |
| `WORKFLOW_PATH` | `./workflows/workflow.json` | Workflow adapter source |
| `NOTEBOOK_PATH` | `./notebooks/….ipynb` | Notebook pushed to Kaggle |
| `MAX_JOB_RETRIES` | `2` | Attempts before a job is marked FAILED |
| `JOB_OUTPUT_RETENTION_HOURS` | `24` | Artifact sweep age |
| `MAX_CONCURRENT_NOTEBOOKS` | `2` | Pool-wide provisioning cap |
| `SCHEDULER_POLL_S` / `JOB_POLL_S` | `5.0` / `3.0` | Scheduler + job poll cadence |
| `JOB_TIMEOUT_S` | `7200` | 单个任务最大渲染时长（秒）；任务 API 里的 `timeout_s` 可覆盖它 |
| `WORKER_HEALTH_TIMEOUT_S` / `HEARTBEAT` | `30` / `60` | Health probing |
| `NOTEBOOK_START_WAIT_S` | `30` | Time to wait for a notebook to come up |
| `WORKER_AUTH_SECRET` | — | Shared controller↔worker Bearer secret |
| `WORKER_AUTH_REQUIRED` | `true` | Gate every worker-agent endpoint |
| `TUNNEL_DOMAIN` | `jayapp.cn` | named 模式的固定域名（**必须一级子域**，见下） |
| `TUNNEL_MODE` | `quick` | `quick`（随机 trycloudflare URL）或 `named`（固定域名） |
| `COLAB_ACCOUNT_<N>_ID/ENABLED` | — | One block per Colab account（OAuth 登录态在隔离 HOME 下，无 env 密钥） |
| `KAGGLE_ACCOUNT_<N>_USERNAME/KEY/ENABLED` | — | One block per Kaggle account |

## Running the controller

```bash
source .venv/bin/activate
uvicorn controller.main:app --host 0.0.0.0 --port 8001
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
- `CLOUDFLARE_TUNNEL_CONFIG` / `CLOUDFLARE_TUNNEL_CREDENTIALS`（named 模式，由 controller 注入）

## Tunnels

**Quick (default, zero-config).** `cloudflared tunnel --url http://127.0.0.1:<agent_port>`
yields a random `.trycloudflare.com` URL parsed from the log. Best for ephemeral
notebooks; URLs change on every run and the controller reconciles via `/health`.

**Named (production, locally-managed).** 用 `cloudflared tunnel create` 建的
locally-managed 隧道（credentials + config.yml，**不是** remotely-managed `--token`）。
固定 URL 是 `https://<worker-id>.<TUNNEL_DOMAIN>`，worker 重启后能自动重连同一个域名。

三条硬性要求（都踩过坑）：

1. **TUNNEL_DOMAIN 必须是一级子域**（如 `jayapp.cn`），worker hostname 形如
   `<worker-id>.jayapp.cn`。Cloudflare 通用证书只覆盖 `*.jayapp.cn`，**二级子域**
   （`*.tunnel.jayapp.cn`）不在证书里，会导致 TLS handshake failure（alert 40）。
2. **DNS 记录必须 proxied=false（灰云）**。`cloudflared tunnel route dns` 默认建
   `proxied=true`（橙云），橙云会让边缘尝试 SSL 代理，与 tunnel 的 QUIC 通道冲突，
   同样 TLS handshake failure。`scripts/create-worker-tunnel.sh` 已自动改 false。
3. **credentials-file 路径**：凭证写到 `/tmp/cloudflared-tunnel/credentials.json`
   （不能用 `/tmp/cloudflared`，那是二进制文件，会 FileExistsError）。

建隧道用 `scripts/create-worker-tunnel.sh`（幂等，`--force` 强制重建，自动配
一级子域 + proxied=false + 写回 .env）。Cloudflare 控制台里的隧道若被删，.env
里的 credentials 会失效（服务端报 `Tunnel not found`），需 `--force` 重建。

In both modes only the **agent** is exposed — ComfyUI stays local on the GPU。
worker agent 还暴露 `/debug` 端点（Bearer 认证），返回 threads / comfy_queue /
comfy_log_tail 三字段，用于卡死排查。

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