"""SQLAlchemy data layer.

Uses SQLAlchemy 2.x with a URL that can point at either SQLite or PostgreSQL;
the ORM models and queries are identical, so switching DATABASE_URL later does
not require rewriting the scheduler. A thin UnitOfWork wrapper keeps service
code DB-agnostic (it only ever calls ``store`` methods, never raw sessions).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import settings
from .constants import (
    AccountStatus, JobStatus, NotebookStatus, WorkerStatus,
)

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime | None = None) -> str | None:
    """Render a timestamp as an explicit-UTC ISO 8601 string (ends in ``Z``).

    SQLite drops tzinfo when a ``DateTime`` round-trips through it, so values
    read back are naive but represent UTC anyway. Emitting an explicit offset
    keeps browsers (``new Date(...)``) from reinterpreting them as local time.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class Account(Base):
    __tablename__ = "accounts"
    id = Column(String, primary_key=True)
    username = Column(String, nullable=False)
    enabled = Column(Integer, default=1)
    status = Column(String, default=AccountStatus.ACCOUNT_AVAILABLE.value)
    # Quota is UNKNOWN because the Kaggle API does not expose it. We keep the
    # field so tooling/DB can later populate it if a source appears.
    quota_status = Column(String, default="UNKNOWN")
    quota_gpu_available = Column(Integer, nullable=True)
    quota_code_usage_hours = Column(Float, nullable=True)
    last_checked_at = Column(DateTime, default=_utcnow)

    notebooks = relationship("Notebook", back_populates="account")


class Notebook(Base):
    __tablename__ = "notebooks"
    id = Column(String, primary_key=True)  # notebook name (kernel ref)
    account_id = Column(String, ForeignKey("accounts.id"), nullable=False)
    gpu_count = Column(Integer, default=2)
    status = Column(String, default=NotebookStatus.NOT_CREATED.value)
    kaggle_kernel_slug = Column(String, nullable=True)  # owner/kernel-title
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)

    account = relationship("Account", back_populates="notebooks")
    workers = relationship("Worker", back_populates="notebook")


class Worker(Base):
    __tablename__ = "workers"
    id = Column(String, primary_key=True)  # worker-0, worker-1, ...
    notebook_id = Column(String, ForeignKey("notebooks.id"), nullable=False)
    gpu_index = Column(Integer, nullable=False)
    comfy_port = Column(Integer, nullable=False)
    comfy_url = Column(String, nullable=True)   # http://127.0.0.1:<port>
    tunnel_url = Column(String, nullable=True)  # cloudflare public URL
    status = Column(String, default=WorkerStatus.UNREGISTERED.value)
    current_job_id = Column(String, nullable=True)
    last_heartbeat_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    token_id = Column(String, nullable=True)     # not the token itself!

    notebook = relationship("Notebook", back_populates="workers")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    status = Column(String, default=JobStatus.QUEUED.value, index=True)
    priority = Column(Integer, default=0)
    attempt_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=2)
    workflow = Column(Text, nullable=True)
    input = Column(Text, nullable=True)     # serialized JSON, redacted on read
    worker_id = Column(String, nullable=True)
    account_id = Column(String, nullable=True)
    error_class = Column(String, default="NONE")
    last_error = Column(Text, nullable=True)
    # 0-100 真实渲染进度（由 controller 轮询 worker 时同步）。
    # 0=排队/未开始，100=完成；RUNNING 时是 worker 上报的最新采样进度。
    progress = Column(Integer, default=0)
    result_artifact_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)


class JobAttempt(Base):
    __tablename__ = "job_attempts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    attempt = Column(Integer, nullable=False)
    worker_id = Column(String, nullable=True)
    status = Column(String, nullable=False)
    error_class = Column(String, default="NONE")
    error = Column(Text, nullable=True)
    prompt_id = Column(String, nullable=True)  # ComfyUI prompt id
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(String, primary_key=True)
    job_id = Column(String, nullable=False, index=True)
    kind = Column(String, nullable=False)  # input|output|workflow
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)  # storage-relative
    size = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, index=True)
    source = Column(String, default="")
    level = Column(String, default="INFO")
    job_id = Column(String, nullable=True)
    message = Column(Text, nullable=False)


def _default_db_url() -> str:
    return settings.database_url if settings else "sqlite:///./storage/controller.db"


class Store:
    """Owns the engine + session. Schema-relative paths are resolved to abs."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        url = database_url or _default_db_url()
        self.url = url
        connect_args: dict = {}
        if url.startswith("sqlite"):
            # Auto-create parent dir for sqlite files.
            # e.g. sqlite:///./storage/controller.db -> ./storage
            if url.startswith("sqlite:///./"):
                p = url.split("sqlite:///", 1)[1]
                Path(p).parent.mkdir(parents=True, exist_ok=True)
            connect_args["check_same_thread"] = False
            # Enable WAL for safer multi-writer sqlite access (helps sqlite conn).
            if url.startswith("sqlite"):
                connect_args["timeout"] = 30
        self.engine = create_engine(url, connect_args=connect_args, future=True)
        maker = sessionmaker(bind=self.engine, future=True)
        self.Session = maker
        Base.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """Lightweight column migrations for existing SQLite databases.

        ``create_all`` only creates missing tables; it never adds columns to
        an already-existing table. Add additive migrations here.
        """
        if not self.url.startswith("sqlite"):
            return
        import sqlite3 as _sq
        db_path = self.url.split("sqlite:///", 1)[1]
        if db_path == ":memory:":
            return
        conn = _sq.connect(db_path)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
            if "progress" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN progress INTEGER DEFAULT 0")
                conn.commit()
        finally:
            conn.close()

    @contextmanager
    def session(self) -> Iterator[object]:
        s = self.Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()