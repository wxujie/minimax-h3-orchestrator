"""Job registry + state machine.

A job transitions: QUEUED -> STARTING -> RUNNING -> DOWNLOADING -> COMPLETED
(also FAILED / CANCELLED / RETRYING). Attempts are recorded in job_attempts.
Retry policy distinguishes TRANSIENT (retry elsewhere) from PERMANENT (do not
retry) workflow errors. DB-only; the scheduler drives transitions.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import uuid
from typing import Optional

from . import db
from .constants import ErrorClass, JobStatus
from .logging_conf import get_logger

log = get_logger("jobs")

_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class JobError(Exception):
    def __init__(self, message: str, error_class: str = ErrorClass.PERMANENT) -> None:
        super().__init__(message)
        self.message = message
        self.error_class = error_class


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:12]


class JobManager:
    def __init__(self, store: db.Store) -> None:
        self.store = store

    # ----------------------------------------------------------------- create --
    def create(self, *, workflow: str = "minimax-h3",
               input_data: dict, priority: int = 0,
               max_retries: Optional[int] = None,
               ref_images: Optional[list] = None) -> dict:
        job_id = new_job_id()
        with self.store.session() as s:
            if ref_images:
                input_data = dict(input_data)
                input_data["ref_images"] = list(ref_images)
            job = db.Job(
                id=job_id,
                status=JobStatus.QUEUED.value,
                priority=priority,
                max_retries=max_retries,
                workflow=workflow,
                input=_json.dumps(input_data, ensure_ascii=False),
            )
            s.add(job)
            s.flush()
            position = self._queue_position(s, job_id)
        return {"job_id": job_id, "status": JobStatus.QUEUED.value, "position": position}

    def _queue_position(self, s, job_id: str) -> int:
        rows = (s.query(db.Job)
                 .filter(db.Job.status == JobStatus.QUEUED.value)
                 .order_by(db.Job.priority.desc(), db.Job.created_at.asc())
                 .all())
        for i, j in enumerate(rows):
            if j.id == job_id:
                return i
        return len(rows)

    # ----------------------------------------------------------------- reads --
    def get(self, job_id: str) -> Optional[dict]:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            return self._public(j) if j else None

    def list(self, status: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self.store.session() as s:
            q = s.query(db.Job).order_by(db.Job.created_at.desc())
            if status:
                q = q.filter(db.Job.status == status)
            return [self._public(j) for j in q.limit(limit).all()]

    def counts(self) -> dict:
        with self.store.session() as s:
            rows = s.query(db.Job.status).all()
        counts = {st.value: 0 for st in JobStatus}
        for (st,) in rows:
            counts[st] = counts.get(st, 0) + 1
        return counts

    def next_in_queue(self) -> Optional[dict]:
        with self.store.session() as s:
            j = (s.query(db.Job)
                 .filter(db.Job.status == JobStatus.QUEUED.value)
                 .order_by(db.Job.priority.desc(), db.Job.created_at.asc())
                 .first())
            return self._public(j) if j else None

    def _public(self, j: db.Job) -> dict:
        return {
            "job_id": j.id,
            "status": j.status,
            "priority": j.priority,
            "created_at": db.utc_iso(j.created_at),
            "worker_id": j.worker_id,
            "account_id": j.account_id,
            "error_class": j.error_class,
            "last_error": j.last_error,
            "result_artifact_id": j.result_artifact_id,
            "input": _json.loads(j.input) if j.input else {},
            "workflow": j.workflow,
        }

    # ---------------------------------------------------------------- updates --
    def set_status(self, job_id: str, status: JobStatus, **kw) -> None:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if not j:
                return
            j.status = status.value
            for k, v in kw.items():
                if hasattr(j, k):
                    setattr(j, k, v)
            log.info("job_status_change job_id=%s status=%s", job_id, status.value)

    def assign(self, job_id: str, worker_id: str, account_id: str) -> None:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if j:
                j.worker_id = worker_id
                j.account_id = account_id

    def fail(self, job_id: str, error: str,
             error_class: str = ErrorClass.TRANSIENT) -> None:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if not j:
                return
            j.last_error = error
            j.error_class = error_class
            if error_class == ErrorClass.PERMANENT:
                j.status = JobStatus.FAILED.value
            else:
                j.status = JobStatus.RETRYING.value
            s.flush()
            j.attempt_count = j.attempt_count or 0

    def record_attempt(self, job_id: str, attempt: int, status: str,
                       error_class: str = ErrorClass.NONE, error: str = "",
                       prompt_id: Optional[str] = None) -> None:
        with self.store.session() as s:
            s.add(db.JobAttempt(
                job_id=job_id, attempt=attempt, status=status,
                error_class=error_class, error=error, prompt_id=prompt_id,
                started_at=_dt.datetime.now(_dt.timezone.utc),
            ))

    def start_attempt(self, job_id: str) -> int:
        """Increment attempt_count and mark STARTING; returns the new count."""
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if not j:
                return 0
            j.attempt_count = (j.attempt_count or 0) + 1
            j.status = JobStatus.STARTING.value
            # Keep the assigned worker_id: the scheduler calls assign() just
            # before this, and monitor_running() needs it to poll the worker.
            j.started_at = _dt.datetime.now(_dt.timezone.utc)
            s.add(db.JobAttempt(job_id=job_id, attempt=j.attempt_count,
                                status="STARTING",
                                started_at=_dt.datetime.now(_dt.timezone.utc)))
            return j.attempt_count

    def finish_attempt(self, job_id: str, success: bool, error: str = "",
                       error_class: str = ErrorClass.NONE,
                       prompt_id: Optional[str] = None) -> None:
        with self.store.session() as s:
            att = (s.query(db.JobAttempt)
                    .filter(db.JobAttempt.job_id == job_id)
                    .order_by(db.JobAttempt.attempt.desc())
                    .first())
            if att:
                att.status = JobStatus.COMPLETED.value if success else JobStatus.FAILED.value
                att.finished_at = _dt.datetime.now(_dt.timezone.utc)
                att.error = error
                att.error_class = error_class
                if prompt_id:
                    att.prompt_id = prompt_id
                s.flush()

    def complete(self, job_id: str, artifact_id: Optional[str] = None) -> None:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if not j:
                return
            j.status = JobStatus.COMPLETED.value
            j.result_artifact_id = artifact_id
            j.completed_at = _dt.datetime.now(_dt.timezone.utc)

    def cancel(self, job_id: str) -> bool:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if not j or j.status in {st.value for st in _TERMINAL}:
                return False
            j.status = JobStatus.CANCELLED.value
            j.cancelled_at = _dt.datetime.now(_dt.timezone.utc)
            return True

    def retryable(self, job_id: str) -> bool:
        with self.store.session() as s:
            j = s.query(db.Job).get(job_id)
            if not j:
                return False
            attempts = j.attempt_count or 0
            return (j.error_class in {ErrorClass.TRANSIENT, ErrorClass.NONE} and
                    attempts <= (j.max_retries or 0))