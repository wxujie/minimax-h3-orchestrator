"""Controller-side HTTP client for a single worker agent.

Talks to a worker's public Cloudflare URL with Bearer auth (WORKER_AUTH_SECRET).
All the heavy lifting (upload, submit, poll, download) happens on the agent; the
controller issues only these remote operations:

    health() / status()
    submit(prompt_text, duration, ..., filenames)
    job_status(job_id)
    download(job_id) -> bytes
    cancel(job_id)

Every read carries an optimistic per-worker auth token. Errors are raised as
WorkerClientError with a ``transient`` flag.
"""
from __future__ import annotations

from typing import Optional

import httpx

from .logging_conf import get_logger

log = get_logger("worker_client")


class WorkerClientError(Exception):
    def __init__(self, message: str, transient: bool = True, status_code: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.transient = transient
        self.status_code = status_code


class WorkerClient:
    def __init__(self, public_url: str, token: str, timeout: float = 120.0) -> None:
        self.public_url = public_url.rstrip("/")
        self.token = token
        self._headers = {"Authorization": f"Bearer {token}"}
        self._http = httpx.Client(timeout=timeout)

    # --------------------------------------------------------------- plumbing --
    def _get(self, path: str, **kw):
        try:
            return self._http.get(f"{self.public_url}{path}",
                                  headers=self._headers, **kw)
        except httpx.TransportError as e:
            raise WorkerClientError(f"GET {path}: {e}", transient=True) from e

    def _post_json(self, path: str, body: dict):
        try:
            return self._http.post(f"{self.public_url}{path}", json=body,
                                   headers=self._headers)
        except httpx.TransportError as e:
            raise WorkerClientError(f"POST {path}: {e}", transient=True) from e

    @staticmethod
    def _check(r: httpx.Response):
        if r.status_code >= 500:
            raise WorkerClientError(
                f"worker HTTP {r.status_code}: {r.text[:300]}",
                transient=True, status_code=r.status_code)
        if r.status_code >= 400:
            raise WorkerClientError(
                f"worker HTTP {r.status_code}: {r.text[:300]}",
                transient=False, status_code=r.status_code)

    # ---------------------------------------------------------------- health ---
    def health(self) -> dict:
        r = self._get("/health")
        if r.status_code >= 400:
            raise WorkerClientError(f"health returned {r.status_code}",
                                    transient=r.status_code >= 500)
        return r.json()

    def status(self) -> dict:
        r = self._get("/status")
        self._check(r)
        return r.json()

    # ---------------------------------------------------------------- submit ---
    def submit(self, payload: dict) -> dict:
        r = self._post_json("/jobs", payload)
        self._check(r)
        return r.json()

    def job_status(self, job_id: str) -> dict:
        r = self._get(f"/jobs/{job_id}")
        self._check(r)
        return r.json()

    def download(self, job_id: str) -> bytes:
        r = self._get(f"/jobs/{job_id}/result")
        self._check(r)
        return r.content

    def cancel(self, job_id: str) -> bool:
        try:
            r = self._post_json(f"/jobs/{job_id}/cancel", {})
            return r.status_code < 400
        except WorkerClientError:
            return False