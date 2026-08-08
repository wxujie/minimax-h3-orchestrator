"""FastAPI application factory for the controller.

Wires the SQLite store, manages singletons (accounts, workers, jobs,
scheduler), exposes REST routers, and serves the static dashboard. Also
provides the one inbound endpoint the notebook's worker agents call to register
their Cloudflare URL (POST /api/v1/agents/register).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles

from . import db
from .accounts import AccountManager
from .config import settings, redact
from .constants import GPU_PER_NOTEBOOK, WorkerStatus
from .jobs import JobManager
from .kaggle_manager import KaggleManager
from .logging_conf import get_logger, setup_logging
from .scheduler import Scheduler
from .workers import WorkerManager

log = get_logger("app")

# Set in `create_app` after store is built.
_store: Optional[db.Store] = None
_accounts: Optional[AccountManager] = None
_workers: Optional[WorkerManager] = None
_jobs: Optional[JobManager] = None
_scheduler: Optional[Scheduler] = None
_workflow_adapter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    _init()   # build the store + managers and start the scheduler on server startup
    log.info("controller_starting")
    yield
    if _scheduler:
        _scheduler.stop()
    log.info("controller_stopping")


def create_app() -> FastAPI:
    from .api import accounts as api_accounts
    from .api import jobs as api_jobs
    from .api import system as api_system
    from .api import workers as api_workers

    app = FastAPI(title="MiniMax H3 Orchestrator", version="1.0.0", lifespan=lifespan)

    app.include_router(api_jobs.router, prefix="/api/v1")
    app.include_router(api_workers.router, prefix="/api/v1")
    app.include_router(api_accounts.router, prefix="/api/v1")
    app.include_router(api_system.router, prefix="/api/v1")

    # Inbound registration from a notebook's worker agent.
    app.add_api_route("/api/v1/agents/register", agent_register, methods=["POST"])

    # Static dashboard.
    dash_dir = Path(__file__).resolve().parent.parent / "web"
    if dash_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dash_dir), html=True),
                  name="dashboard")
    return app


class KaggleProvider:
    """Adapter so the scheduler can talk to whichever Kaggle account manager
    exists. Real running uses KaggleManager per account; the singleton keeps a
    registry of the last-used manager."""

    def __init__(self) -> None:
        self.managers: dict[str, KaggleManager] = {}

    def start_notebook(self, account_id: str, notebook_name: str) -> bool:
        ac = (_accounts.credential(account_id) if _accounts else None)
        if not ac:
            return False
        mgr = self.managers.get(account_id) or KaggleManager(ac)
        self.managers[account_id] = mgr
        try:
            cap = mgr.capacity()
            if not cap.usable:
                return False
            slug = f"{ac.username}/{notebook_name}"
            return mgr.ensure_notebook(slug, DEFAULT_NOTEBOOK_BODY())
        except Exception as exc:  # noqa: BLE001
            log.error("kaggle_start_notebook_failed account=%s err=%s",
                      account_id, exc)
            return False


def DEFAULT_NOTEBOOK_BODY() -> dict:
    """Sketch of the notebook cell-document (filled by build at run time)."""
    return {"cells": []}


def _init() -> None:
    global _store, _accounts, _workers, _jobs, _scheduler
    _store = db.Store()
    _accounts = AccountManager(_store)
    _accounts.sync_from_config()
    _workers = WorkerManager(_store)
    _jobs = JobManager(_store)
    _scheduler = Scheduler(_store, _jobs, _workers, _accounts, KaggleProvider())
    _scheduler.start()


async def agent_register(request, authorization: str = Header(default="")):
    """Worker agent calls this once its Cloudflare tunnel is up."""
    _require_token(authorization)
    payload = await request.json()
    worker_id = _clean(payload.get("worker_id", ""))
    notebook_id = _clean(payload.get("notebook_id", ""))
    tunnel_url = _clean(payload.get("tunnel_url", ""))
    gpu = int(payload.get("gpu", 0))
    if not worker_id or not notebook_id or not tunnel_url:
        raise HTTPException(status_code=400, detail="worker_id/notebook_id/tunnel_url required")
    _workers.register(worker_id=worker_id, notebook_id=notebook_id, gpu_index=gpu,
                      comfy_port=8188 + gpu, comfy_url=f"http://127.0.0.1:{8188+gpu}",
                      tunnel_url=tunnel_url)
    return {"ok": True, "worker_id": worker_id}


def _require_token(authorization: str) -> None:
    secret = settings.worker_auth_secret if settings else ""
    if not secret:
        return  # auth disabled in dev
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid token")
    token = authorization.split(" ", 1)[1].strip()
    import hmac
    if not hmac.compare_digest(token, secret):
        raise HTTPException(status_code=401, detail="invalid token")


def _payload(d):  # noqa
    return d


def _clean(s):
    import re
    return re.sub(r"[^\w\-.:/]", "", s or "")


# Re-exports for tests
def get_store():
    return _store


def get_scheduler() -> Scheduler:
    return _scheduler


# Module-level app so the ASGI server can load `controller.main:app`.
# Runtime init (store/scheduler) happens in `lifespan` on server startup, so
# importing this module has no side effects.
app = create_app()