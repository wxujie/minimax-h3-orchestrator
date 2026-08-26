"""Scheduler: dispatches queued jobs to healthy workers, lazily provisions
Kaggle notebooks, monitors running jobs, downloads finished video, and
recovers from transient worker/notebook failures.

Design
------
- READY workers always beat provisioning a new notebook.
- With no READY worker, choose an account and start its notebook lazily.
- Jobs are handed to a worker via its WorkerClient; the worker agent does the
  ComfyUI upload/submit/poll inside the notebook. The controller polls status,
  downloads the finished bytes, signs an artifact, and completes the job.
- Transient failures are requeued (retry), permanent ones are failed.

Deterministic + DB-driven; runs in-process during integration tests with a fake
provider and fake workers (no real Kaggle or network needed).
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from . import db
from .accounts import AccountManager
from .config import settings
from .constants import (
    AccountStatus, ErrorClass, GPU_PER_NOTEBOOK, JobStatus,
    NotebookStatus, WorkerStatus,
)
from .jobs import JobManager
from .logging_conf import get_logger
from .storage import Storage
from .worker_client import WorkerClient, WorkerClientError
from .workers import WorkerManager

log = get_logger("scheduler")


class Scheduler:
    def __init__(
        self,
        store: db.Store,
        jobs: JobManager,
        workers: WorkerManager,
        accounts: AccountManager,
        provider,                        # start_notebook(account_id, name) -> bool
        storage: Optional[Storage] = None,
        client_factory: Optional[callable] = None,
    ) -> None:
        self.store = store
        self.jobs = jobs
        self.workers = workers
        self.accounts = accounts
        self.provider = provider
        self.storage = storage or Storage()
        self.client_factory = client_factory
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._in_flight: set[str] = set()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("scheduler_started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        poll = max(0.5, (settings.schedule_poll_s if settings else 5.0))
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                log.error("scheduler_error err=%s", exc)
            self._stop.wait(poll)

    # ---------------------------------------------------------------------- tick
    def tick(self) -> None:
        self.heartbeat_all()
        self.monitor_running()
        job = self.jobs.next_in_queue()
        if job:
            self.dispatch_job(job["job_id"])

    # ------------------------------------------------------------- heartbeat ---
    def heartbeat_all(self) -> int:
        alive = 0
        for wk in self.workers.list():
            if not wk.get("tunnel_url"):
                continue
            h = self.workers.poll_health(wk)
            if h and h.get("status") == "ready":
                if wk.get("status") == WorkerStatus.WORKER_READY.value:
                    alive += 1
            else:
                if wk.get("status") in (WorkerStatus.WORKER_BUSY.value,
                                        WorkerStatus.WORKER_READY.value):
                    self._recover_worker(wk)
                    self.workers.mark_offline(wk["id"])
                else:
                    self.workers.mark_offline(wk["id"])
        return alive

    def _client_for(self, worker: dict) -> WorkerClient:
        if self.client_factory:
            return self.client_factory(worker)
        token = settings.worker_auth_secret if settings else ""
        return WorkerClient(worker["tunnel_url"], token)

    # --------------------------------------------------------------- dispatch --
    def dispatch_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job["status"] != JobStatus.QUEUED.value:
            return False
        w = self.pick_ready()
        if w:
            return self._run_on(job, w)
        if self._can_provision():
            pick = self._account_to_start()
            if pick:
                self._lazy_start(pick["account_id"], pick["notebook_name"])
        return False

    def dispatch(self, job_id: str) -> bool:
        return self.dispatch_job(job_id)

    def pick_ready(self) -> Optional[dict]:
        for wk in self.workers.ready_workers():
            h = self.workers.poll_health(wk)
            if h and h.get("status") == "ready":
                return wk
        return None

    def _run_on(self, job: dict, worker: dict) -> bool:
        w_id = worker["id"]
        self.jobs.assign(job["job_id"], w_id, worker.get("notebook_id") or "")
        self.jobs.start_attempt(job["job_id"])
        client = self._client_for(worker)
        try:
            remote_id = self._submit(job, client, worker)
        except WorkerClientError as e:
            self.jobs.fail(job["job_id"], e.message, error_class=ErrorClass.TRANSIENT)
            self.workers.mark_error(w_id, e.message)
            self._maybe_retry(job["job_id"], w_id)
            return False
        self.jobs.set_status(job["job_id"], JobStatus.RUNNING)
        self.workers.mark_busy(w_id, job["job_id"])
        # remember the remote worker job id for later status/download
        self._note_remote(job["job_id"], remote_id)
        log.info("scheduler_worker_selected job=%s worker=%s remote=%s",
                 job["job_id"], w_id, remote_id)
        return True

    def _submit(self, job: dict, client: WorkerClient, worker: dict) -> str:
        """Stage frame/ref files to the worker, then submit; returns remote job id."""
        inp = job.get("input") or {}
        workflow = job.get("workflow", "minimax-h3")
        staged = {}
        for key in ("first_frame", "last_frame"):
            raw = inp.get(key)
            if not raw:
                continue
            fname = _safe(raw)
            local = self.storage.upload_path(fname)
            if self.storage.exists(local):
                staged[key] = self._upload_to_worker(client, local)

        payload = {
            "prompt_text": inp.get("prompt", ""),
            "duration": inp.get("duration", 2.0),
            "width": inp.get("width"), "height": inp.get("height"),
            "seed": inp.get("seed"),
            "first_frame": staged.get("first_frame"),
            "last_frame": staged.get("last_frame") or staged.get("first_frame"),
            "model_overrides": inp.get("model_overrides") or {},
            "workflow": workflow,
        }

        # Multishot mode: controller builds the full chained-shot graph.
        if workflow == "minimax-h3-multishot":
            payload["script"] = inp.get("script", "")
            payload["frames_per_shot"] = inp.get("frames_per_shot")
            payload["shot_count"] = inp.get("shot_count", 0)
            try:
                from .workflow_multishot import MultishotWorkflowAdapter
                payload["graph"] = MultishotWorkflowAdapter().build_prompt(
                    script=payload["script"],
                    width=payload["width"] or 768,
                    height=payload["height"] or 768,
                    frames_per_shot=payload["frames_per_shot"] or 243,
                    seed=payload["seed"],
                    shot_count=payload["shot_count"],
                )
            except Exception:  # noqa: BLE001 - fall back to worker adapter
                payload.pop("graph", None)

        # R2V mode: stage + upload reference images, toggle turbo.
        elif workflow == "minimax-h3-r2v":
            ref_names = []
            for raw in (inp.get("ref_images") or []):
                fname = _safe(str(raw))
                local = self.storage.upload_path(fname)
                if self.storage.exists(local):
                    ref_names.append(self._upload_to_worker(client, local))
            payload["ref_images"] = ref_names
            payload["ref_image_size"] = inp.get("ref_image_size") or "match"
            payload["turbo"] = bool(inp.get("turbo", False))
            payload["turbo_steps"] = inp.get("turbo_steps") or 4
            payload["turbo_lora_strength"] = inp.get("turbo_lora_strength") or 1.0
            # Build the full graph controller-side for the no-reference-image
            # case so older Kaggle workers (with a stale adapter) can still
            # execute R2V. The worker falls back to its own adapter when refs
            # are provided (which requires uploading images it can reference).
            if not ref_names:
                try:
                    from .workflow_r2v import R2VWorkflowAdapter
                    payload["graph"] = R2VWorkflowAdapter().build_prompt(
                        prompt_text=payload["prompt_text"],
                        duration=payload["duration"],
                        width=payload["width"], height=payload["height"],
                        seed=payload["seed"],
                        ref_images=[],
                        ref_image_size=payload["ref_image_size"],
                        turbo=payload["turbo"],
                        turbo_steps=payload["turbo_steps"],
                        turbo_lora_strength=payload["turbo_lora_strength"],
                        model_overrides=payload["model_overrides"],
                    )
                except Exception:  # noqa: BLE001 - fall back to worker adapter
                    payload.pop("graph", None)
        elif not staged.get("first_frame"):
            raise WorkerClientError(
                "first_frame not provided or missing on disk", transient=False)
        return client.submit(payload)["job_id"]

    def _upload_to_worker(self, client: WorkerClient, path) -> str:
        fname = str(path).rsplit("/", 1)[-1]
        with open(path, "rb") as f:
            r = client._http.post(f"{client.public_url}/jobs/input",
                                  headers=client._headers,
                                  files={"file": (fname, f, "image/png")})
        if r.status_code >= 400:
            raise WorkerClientError(f"input upload failed: {r.text[:200]}",
                                    transient=r.status_code >= 500)
        return r.json().get("filename")

    def _note_remote(self, job_id: str, remote: str) -> None:
        with self.store.session() as s:
            att = (s.query(db.JobAttempt)
                   .filter(db.JobAttempt.job_id == job_id)
                   .order_by(db.JobAttempt.attempt.desc()).first())
            if att:
                att.prompt_id = remote  # reuse field to hold worker job ref
                s.flush()

    def _remote(self, job_id: str) -> str:
        with self.store.session() as s:
            att = (s.query(db.JobAttempt)
                   .filter(db.JobAttempt.job_id == job_id)
                   .order_by(db.JobAttempt.attempt.desc()).first())
            return att.prompt_id if att and att.prompt_id else job_id

    # --------------------------------------------------------------- monitor ---
    def monitor_running(self) -> None:
        for job in self.jobs.list(status=JobStatus.RUNNING.value):
            self._poll(job)

    def _poll(self, job: dict) -> None:
        job_id = job["job_id"]
        w_id = job.get("worker_id")
        if not w_id:
            return
        wk = self.workers.get(w_id)
        if not wk or not wk.get("tunnel_url"):
            self.jobs.fail(job_id, "worker lost while running",
                           error_class=ErrorClass.TRANSIENT)
            self._maybe_retry(job_id, w_id)
            return
        client = self._client_for(wk)
        try:
            st = client.job_status(self._remote(job_id))
        except WorkerClientError as e:
            self.jobs.fail(job_id, "status poll: " + e.message,
                           error_class=ErrorClass.TRANSIENT)
            return
        if not st:
            return
        if st.get("status") == "DONE":
            self._complete(job_id, w_id, client, self._remote(job_id))
        elif st.get("status") in ("FAILED", "CANCELLED"):
            self.jobs.fail(job_id, st.get("error") or "worker reported failure",
                           error_class=ErrorClass.TRANSIENT)
            self._maybe_retry(job_id, w_id)

    def _complete(self, job_id: str, w_id: str, client: WorkerClient, remote: str) -> None:
        try:
            data = client.download(remote)
        except WorkerClientError as e:
            self.jobs.fail(job_id, "download: " + e.message,
                           error_class=ErrorClass.TRANSIENT)
            return
        if not data:
            self.jobs.fail(job_id, "empty download",
                           error_class=ErrorClass.TRANSIENT)
            return
        fname = f"{job_id}.mp4"
        local = self.storage.write_output(job_id, fname, data)
        aid = self._artifact(job_id, fname, str(local), len(data))
        self.jobs.complete(job_id, aid)
        self.workers.mark_idle(w_id)
        log.info("job_completed job_id=%s file=%s size=%s", job_id, fname, len(data))

    def _artifact(self, job_id: str, fname: str, path: str, size: int) -> str:
        aid = f"art_{uuid.uuid4().hex[:10]}"
        with self.store.session() as s:
            s.add(db.Artifact(id=aid, job_id=job_id, kind="output",
                              filename=fname, path=path, size=size))
        return aid

    # ------------------------------------------------------------- retry/recov --
    def _maybe_retry(self, job_id: str, worker_id: str = "") -> None:
        if worker_id:
            self.workers.mark_idle(worker_id)
        if self.jobs.retryable(job_id):
            self.jobs.set_status(job_id, JobStatus.QUEUED)
            log.info("job_retry_scheduled job_id=%s", job_id)
        else:
            self.jobs.set_status(job_id, JobStatus.FAILED)
            log.warning("job_failed_permanent job_id=%s", job_id)

    def _recover_worker(self, worker: dict) -> None:
        jid = worker.get("current_job_id")
        if not jid:
            return
        with self.store.session() as s:
            j = s.query(db.Job).get(jid)
            if j and j.status == JobStatus.RUNNING.value:
                j.status = JobStatus.QUEUED.value
                j.error_class = ErrorClass.TRANSIENT
                log.warning("job_requeued_after_worker_drop job_id=%s", jid)

    # ----------------------------------------------------------- provisioning --
    def _can_provision(self) -> bool:
        with self.store.session() as s:
            active = (s.query(db.Notebook)
                      .filter(db.Notebook.status.in_([
                          NotebookStatus.NOTEBOOK_STARTING.value,
                          NotebookStatus.NOTEBOOK_RUNNING.value]))
                      .count())
        cap = settings.max_concurrent_notebooks if settings else 2
        return active + len(self._in_flight) < cap

    def _account_to_start(self) -> Optional[dict]:
        with self.store.session() as s:
            q = (s.query(db.Account)
                 .filter(db.Account.enabled == 1)
                 .order_by(db.Account.id).all())
            for acc in q:
                if acc.status in (AccountStatus.QUOTA_EXHAUSTED.value,
                                  AccountStatus.ACCOUNT_DISABLED.value):
                    continue
                if acc.id in self._in_flight:
                    continue
                running = (s.query(db.Notebook)
                           .filter(db.Notebook.account_id == acc.id,
                                   db.Notebook.status.in_([
                                       NotebookStatus.NOTEBOOK_STARTING.value,
                                       NotebookStatus.NOTEBOOK_RUNNING.value]))
                           .count())
                if running == 0:
                    return {"account_id": acc.id,
                            "notebook_name": f"nb-{acc.id}"}
        return None

    def _lazy_start(self, account_id: str, notebook_name: str) -> None:
        with self.store.session() as s:
            s.merge(db.Notebook(id=notebook_name, account_id=account_id,
                                gpu_count=GPU_PER_NOTEBOOK,
                                status=NotebookStatus.NOTEBOOK_STARTING.value))
        self._in_flight.add(account_id)
        try:
            ok = self.provider.start_notebook(account_id, notebook_name)
        except Exception as exc:  # noqa: BLE001
            log.error("scheduler_notebook_start_failed notebook=%s err=%s",
                      notebook_name, exc)
            ok = False
        finally:
            self._in_flight.discard(account_id)
        if not ok:
            with self.store.session() as s:
                nb = s.query(db.Notebook).get(notebook_name)
                if nb:
                    nb.status = NotebookStatus.QUOTA_EXHAUSTED.value
                    nb.last_error = "provider.start_notebook failed"
            return
        self.workers.provisioned(notebook_name, GPU_PER_NOTEBOOK)
        log.info("scheduler_notebook_started notebook=%s account=%s",
                 notebook_name, account_id)


def _safe(raw: str) -> str:
    import os.path as _p
    return _p.basename(raw)[:120]