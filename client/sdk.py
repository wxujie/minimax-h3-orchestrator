"""MiniMax H3 Orchestrator Python client.

Independent of the Kaggle implementation: the caller never sees accounts,
notebooks, Cloudflare URLs, or GPU assignments — only the controller API.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class VideoJob:
    client: "VideoClient"
    job_id: str
    status: str = "QUEUED"
    progress: int = 0
    error: Optional[str] = None
    download_url: Optional[str] = None


class VideoClient:
    def __init__(self, base_url: str, token: Optional[str] = None,
                 timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout, verify=True)
        self._headers = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------ jobs --
    def create_video(self, *, prompt: str, duration: float = 2.0,
                     image: Optional[str] = None,
                     last_image: Optional[str] = None,
                     width: Optional[int] = None, height: Optional[int] = None,
                     seed: Optional[int] = None, priority: int = 0) -> VideoJob:
        """Create a video job. ``image``/``last_image`` are local paths to
        frame images uploaded to the controller."""
        files = {}
        data = {
            "prompt": prompt, "duration": duration, "priority": priority,
            "width": width, "height": height, "seed": seed,
        }
        if image:
            files["first_frame"] = ("first.png", open(image, "rb"), "image/png")
        if last_image:
            files["last_frame"] = ("last.png", open(last_image, "rb"), "image/png")
        if files:
            r = self._http.post(f"{self.base_url}/api/v1/jobs/multipart",
                                data=_clean(data), files=files,
                                headers=self._headers, timeout=120.0)
        else:
            if not last_image:
                data["first_frame"] = None
            r = self._http.post(f"{self.base_url}/api/v1/jobs",
                               json=data, headers=self._headers, timeout=120.0)
        _check(r)
        j = r.json()
        return VideoJob(self, j["job_id"], status=j.get("status", "QUEUED"))

    def get_job(self, job_id: str) -> VideoJob:
        r = self._http.get(f"{self.base_url}/api/v1/jobs/{job_id}",
                           headers=self._headers)
        _check(r)
        j = r.json()
        return VideoJob(self, j["job_id"], status=j["status"],
                        progress=j.get("progress", 0), error=j.get("error"),
                        download_url=j.get("download_url"))

    def cancel(self, job_id: str) -> None:
        r = self._http.post(f"{self.base_url}/api/v1/jobs/{job_id}/cancel",
                            headers=self._headers)
        _check(r)

    def download(self, job_id: str, dest: str | Path) -> Path:
        """Download the finished video to ``dest`` (file path)."""
        r = self._http.get(f"{self.base_url}/api/v1/jobs/{job_id}/result",
                           headers=self._headers, timeout=300.0)
        _check(r)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def wait_for_result(self, job_id: str, timeout: float = 1800.0,
                        poll: float = 5.0) -> VideoJob:
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.get_job(job_id)
            if j.status == "COMPLETED":
                return j
            if j.status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"job {job_id} {j.status}: {j.error}")
            time.sleep(poll)
        raise TimeoutError(f"job {job_id} did not finish in {timeout}s")

    # ------------------------------------------------------------- system ----
    def workers(self) -> list[dict]:
        r = self._http.get(f"{self.base_url}/api/v1/workers", headers=self._headers)
        _check(r)
        return r.json()

    def accounts(self) -> list[dict]:
        r = self._http.get(f"{self.base_url}/api/v1/accounts", headers=self._headers)
        _check(r)
        return r.json()

    def system_status(self) -> dict:
        r = self._http.get(f"{self.base_url}/api/v1/system/status", headers=self._headers)
        _check(r)
        return r.json()


def _clean(data: dict) -> dict:
    return {k: v for k, v in data.items() if v is not None}


def _check(r: httpx.Response) -> None:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"controller HTTP {r.status_code}: {detail}")