"""Deterministic fakes so the scheduler is testable without Kaggle/GPU/tunnels.

FakeProvider records start_notebook calls and can be made to succeed or fail.
FakeWorkerClient stands in for the controller's WorkerClient: it has the same
surface (public_url, _http for frame upload, submit / job_status / download) but
returns scripted responses so no real network is touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class FakeProvider:
    """Provider with |start_notebook(account_id, notebook_name) -> bool|."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.fail_next = False
        self.limit_reached = False
        self.started: list[tuple[str, str]] = []

    def start_notebook(self, account_id: str, notebook_name: str) -> bool:
        if self.fail_next or self.limit_reached or not self.ok:
            return False
        self.started.append((account_id, notebook_name))
        return True


class WorkerClientError(Exception):
    """Mirrors controller.worker_client.WorkerClientError."""

    def __init__(self, message: str, transient: bool = True, status_code: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.transient = transient
        self.status_code = status_code


class _FakeResp:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.text = ""

    def json(self) -> dict:
        return self._body


class _FakeHttp:
    def post(self, *args, **kwargs):
        files = kwargs.get("files") or {}
        filename = "frame.png"
        for key in ("first_frame", "last_frame", "file"):
            if key in files:
                _n = files[key]
                if isinstance(_n, tuple) and len(_n) >= 2:
                    filename = _n[0]
        return _FakeResp(200, {"filename": filename})


@dataclass
class FakeWorkerScript:
    """Scripted behaviour for one FakeWorkerClient (keyed by worker id)."""

    status: str = "ready"        # ready|offline, then job_test -> DONE/FAILED
    job_status: str = "DONE"     # terminal status job_status settles on
    pending_steps: int = 0       # how many RUNNING polls before settling
    error: str = ""
    download: bytes = b"\x00\x00\x00\x18ftypmp42 dummy-mp4"


class FakeWorkerClient:
    """Drops-in for controller.worker_client.WorkerClient."""

    def __init__(self, worker: dict, script: FakeWorkerScript | None = None) -> None:
        self.worker = worker
        self.job_id = f"wj-{worker['id']}"
        self.script = script or FakeWorkerScript()
        self._calls = 0

    @property
    def public_url(self) -> str:
        return "http://fake.local"

    @property
    def _headers(self) -> dict:
        return {"Authorization": "Bearer test"}

    @property
    def _http(self) -> _FakeHttp:
        return _FakeHttp()

    def submit(self, payload: dict) -> dict:
        return {"job_id": self.job_id}

    def job_status(self, remote: str) -> dict:
        if self.script.status in ("offline", "starting"):
            raise WorkerClientError("worker offline", transient=True)
        self._calls += 1
        if self.script.pending_steps > 0:
            self.script.pending_steps -= 1
            return {"job_id": remote, "status": "RUNNING", "progress": 20}
        return {"job_id": remote, "status": self.script.job_status,
                "error": self.script.error or ""}

    def download(self, remote: str) -> bytes:
        if not self.script.download:
            raise WorkerClientError("download: empty result", transient=True)
        return self.script.download

    def health(self) -> dict:
        return {"status": "ready" if self.script.status != "offline" else "busy"}


def make_client_factory(scripts: dict[str, FakeWorkerScript] | None = None):
    """Return a scheduler client_factory that yields a FakeWorkerClient per id."""
    scripts = scripts or {}

    def factory(worker: dict) -> FakeWorkerClient:
        return FakeWorkerClient(worker, scripts.get(worker["id"]))

    return factory