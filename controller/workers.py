"""Worker registry: discovery, heartbeat, state transitions.

Workers are created when a notebook registers them (POST /agents/register from
the notebook's worker agent) or when the scheduler provisions a notebook and
expects GPU_PER_NOTEBOOK workers. State is stored in the DB; the scheduler
refreshes ``last_heartbeat_at`` by polling /health through the Cloudflare URL.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

from . import db
from .config import settings
from .constants import WorkerStatus
from .logging_conf import get_logger
from .worker_client import WorkerClient, WorkerClientError

log = get_logger("workers")


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class WorkerManager:
    def __init__(self, store: db.Store) -> None:
        self.store = store
        self._client_cache: dict[str, WorkerClient] = {}

    # ------------------------------------------------------------ lifecycle ---
    def register(self, *, worker_id: str, notebook_id: str, gpu_index: int,
                 comfy_port: int, comfy_url: str, tunnel_url: str,
                 token: Optional[str] = None) -> dict:
        """Upsert a worker's row; used by the notebook's registration call."""
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w is None:
                w = db.Worker(id=worker_id, notebook_id=notebook_id,
                              gpu_index=gpu_index, comfy_port=comfy_port,
                              comfy_url=comfy_url, tunnel_url=tunnel_url,
                              status=WorkerStatus.WORKER_STARTING.value,
                              last_heartbeat_at=_now())
                s.add(w)
            else:
                w.notebook_id = notebook_id
                w.gpu_index = gpu_index
                w.comfy_port = comfy_port
                w.comfy_url = comfy_url
                w.tunnel_url = tunnel_url
                w.last_heartbeat_at = _now()
            if token:
                w.token_id = token  # reference id only, never the secret
            s.flush()
            worker_id = w.id
        log.info("worker_registered worker=%s gpu=%s url=%s",
                 worker_id, gpu_index, tunnel_url)
        return {"worker_id": worker_id, "status": WorkerStatus.WORKER_STARTING.value}

    def provisioned(self, notebook_id: str, gpu_count: int) -> list[str]:
        """Create placeholder rows for a notebook's expected GPU workers."""
        created = []
        with self.store.session() as s:
            nb = s.query(db.Notebook).get(notebook_id)
            if not nb:
                return created
            for g in range(gpu_count):
                wid = f"{notebook_id}-gpu{g}"
                exists = s.query(db.Worker).get(wid)
                if exists is None:
                    s.add(db.Worker(
                        id=wid, notebook_id=notebook_id, gpu_index=g,
                        comfy_port=8188 + g,
                        status=WorkerStatus.UNREGISTERED.value,
                    ))
                    created.append(wid)
        return created

    def mark_ready(self, worker_id: str) -> None:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w:
                w.status = WorkerStatus.WORKER_READY.value
                w.last_heartbeat_at = _now()
        log.info("worker_ready worker=%s", worker_id)

    def mark_busy(self, worker_id: str, job_id: Optional[str] = None) -> None:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w:
                w.status = WorkerStatus.WORKER_BUSY.value
                w.current_job_id = job_id
        log.info("worker_busy worker=%s job=%s", worker_id, job_id)

    def mark_idle(self, worker_id: str) -> None:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w:
                w.status = WorkerStatus.WORKER_READY.value
                w.current_job_id = None
                w.last_heartbeat_at = _now()

    def mark_error(self, worker_id: str, error: str) -> None:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w:
                w.status = WorkerStatus.WORKER_ERROR.value
                w.last_error = error
                w.current_job_id = None

    def mark_offline(self, worker_id: str) -> None:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w:
                w.status = WorkerStatus.WORKER_OFFLINE.value
                w.current_job_id = None

    # --------------------------------------------------------------- queries ---
    def get(self, worker_id: str) -> Optional[dict]:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            return self._public(w) if w else None

    def list(self) -> list[dict]:
        with self.store.session() as s:
            rows = s.query(db.Worker).order_by(db.Worker.id).all()
            return [self._public(w) for w in rows]

    def ready_workers(self) -> list[dict]:
        with self.store.session() as s:
            rows = (s.query(db.Worker)
                     .filter(db.Worker.status == WorkerStatus.WORKER_READY.value)
                     .order_by(db.Worker.id).all())
            return [self._public(w) for w in rows]

    def by_notebook(self, notebook_id: str) -> list[dict]:
        with self.store.session() as s:
            rows = (s.query(db.Worker)
                     .filter(db.Worker.notebook_id == notebook_id)
                     .order_by(db.Worker.gpu_index).all())
            return [self._public(w) for w in rows]

    def _public(self, w: db.Worker) -> dict:
        return {
            "id": w.id,
            "notebook_id": w.notebook_id,
            "gpu_index": w.gpu_index,
            "comfy_port": w.comfy_port,
            "comfy_url": w.comfy_url,
            "tunnel_url": w.tunnel_url,
            "status": w.status,
            "current_job_id": w.current_job_id,
            "last_heartbeat_at": w.last_heartbeat_at.isoformat() if w.last_heartbeat_at else None,
            "last_error": w.last_error,
        }

    # ------------------------------------------------------------- heartbeat ---
    def poll_health(self, worker: dict) -> Optional[dict]:
        """Call GET /health through the tunnel; update DB state accordingly.

        Returns the health payload on success, None on hard failure.
        """
        url = worker.get("tunnel_url")
        if not url:
            return None
        client = self._client_cache.get(worker["id"])
        if client is None or client.public_url != url.rstrip("/"):
            client = WorkerClient(url, self._token_for(worker["id"]))
            self._client_cache[worker["id"]] = client
        try:
            h = client.health()
        except WorkerClientError as e:
            log.warning("worker_health_failed worker=%s err=%s", worker["id"], e.message)
            self.mark_offline(worker["id"])
            return None
        # health says ready -> mark ready (if it was offline/starting)
        if h.get("status") == "ready" and worker["status"] != WorkerStatus.WORKER_BUSY.value:
            self.mark_ready(worker["id"])
        else:
            self._touch_heartbeat(worker["id"])
        return h

    def _token_for(self, worker_id: str) -> str:
        # In production a per-worker token derived from a pool secret. We use
        # the pool secret for all workers; the agent compares with the same.
        return settings.worker_auth_secret if settings else ""

    def _touch_heartbeat(self, worker_id: str) -> None:
        with self.store.session() as s:
            w = s.query(db.Worker).get(worker_id)
            if w:
                w.last_heartbeat_at = _now()