"""System status + dashboard data endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from .. import main as pmain
from ..constants import NotebookStatus, WorkerStatus
from ..jobs import JobManager
from ..workers import WorkerManager

router = APIRouter()


def _jobs() -> JobManager:
    store = pmain.get_store()
    return JobManager(store)


def _workers() -> WorkerManager:
    store = pmain.get_store()
    return WorkerManager(store)


@router.get("/system/status")
def system_status():
    from .. import db as _db
    store = pmain.get_store()
    counts = _jobs().counts()
    workers = _workers().list()
    ready = sum(1 for w in workers if w["status"] == WorkerStatus.WORKER_READY.value)
    busy = sum(1 for w in workers if w["status"] == WorkerStatus.WORKER_BUSY.value)
    with store.session() as s:
        nb_running = (s.query(_db.Notebook)
                      .filter(_db.Notebook.status == NotebookStatus.NOTEBOOK_RUNNING.value)
                      .count())
    queued = counts.get("QUEUED", 0)
    return {
        "ready_workers": ready,
        "busy_workers": busy,
        "queued_jobs": queued,
        "running_notebooks": nb_running,
        "job_counts": counts,
    }