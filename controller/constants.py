"""Shared state constants and enums for the orchestration system.

Single source of truth for every state a Job / Worker / Notebook / Account
can be in. Kept import-light so both the controller and the worker agent can
import it without pulling in heavy dependencies.
"""
from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DOWNLOADING = "DOWNLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


class WorkerStatus(str, Enum):
    UNREGISTERED = "UNREGISTERED"
    WORKER_STARTING = "WORKER_STARTING"
    WORKER_READY = "WORKER_READY"
    WORKER_BUSY = "WORKER_BUSY"
    WORKER_ERROR = "WORKER_ERROR"
    WORKER_OFFLINE = "WORKER_OFFLINE"


class NotebookStatus(str, Enum):
    NOT_CREATED = "NOT_CREATED"
    NOTEBOOK_STARTING = "NOTEBOOK_STARTING"
    NOTEBOOK_RUNNING = "NOTEBOOK_RUNNING"
    NOTEBOOK_STOPPING = "NOTEBOOK_STOPPING"
    NOTEBOOK_STOPPED = "NOTEBOOK_STOPPED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"


class AccountStatus(str, Enum):
    ACCOUNT_DISABLED = "ACCOUNT_DISABLED"
    ACCOUNT_AVAILABLE = "ACCOUNT_AVAILABLE"
    ACCOUNT_ERROR = "ACCOUNT_ERROR"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"


class ErrorClass(str, Enum):
    """Differentiation driving retry policy.

    TRANSIENT  -> safe to retry on another worker (network, tunnel, crash).
    PERMANENT  -> deterministic workflow/validation error, do not retry.
    """
    NONE = "NONE"
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"


# How many independent GPUs (and therefore ComfyUI instances / workers) a
# single Kaggle notebook is designed to host.
GPU_PER_NOTEBOOK = 2


class ConfigNode(str, Enum):
    """The workflow.json nodes the adapter can override."""

    IMAGE_TO_VIDEO = "105"
    RESOLUTION_SELECTOR = "115"
    SAVE_VIDEO = "92"


def worker_id(notebook_name: str, gpu_index: int) -> str:
    """Stable worker identifier derived from a notebook + gpu slot."""
    return f"{notebook_name}-gpu{gpu_index}"