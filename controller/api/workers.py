"""Worker status endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from .. import main as pmain
from ..workers import WorkerManager

router = APIRouter()


def _workers() -> WorkerManager:
    store = pmain.get_store()
    return WorkerManager(store)


@router.get("/workers")
def list_workers():
    return _workers().list()


@router.get("/workers/{worker_id}")
def get_worker(worker_id: str):
    w = _workers().get(worker_id)
    if not w:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="worker not found")
    return w