"""Minimal ComfyUI client used by the worker agent.

Talks to ONE local ComfyUI instance (its own GPU). Exposes the subset of the
ComfyUI API needed by the orchestrator:

    GET  /system_stats
    GET  /object_info            (node class schemas)
    POST /upload/image
    POST /prompt                 (submit flat API graph)
    GET  /prompt/{prompt_id}     (queue/execution info)
    GET  /history/{prompt_id}    (results, incl. output file entries)
    GET  /view                   (download a file)
    POST /queue                  (cancel via {"delete": [prompt_id]})

Timeouts are generous: H3 video generation runs many minutes. Every call is
bounded; failures raise ``ComfyError`` which the agent maps to a TRANSIENT or
PERMANENT job error.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx


class ComfyError(Exception):
    """ComfyUI communication failure. ``transient`` controls retry policy."""

    def __init__(self, message: str, transient: bool = True, status_code: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.transient = transient
        self.status_code = status_code


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188",
                 timeout: float = 60.0, client: Optional[httpx.Client] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = client or httpx.Client(timeout=timeout)
        self.timeout = timeout

    # ------------------------------------------------------------- plumbing ----
    def _get(self, path: str, **kw) -> httpx.Response:
        try:
            return self._http.get(f"{self.base_url}{path}", **kw)
        except httpx.TransportError as e:
            raise ComfyError(f"GET {path}: {e}", transient=True) from e

    def _post(self, path: str, **kw) -> httpx.Response:
        try:
            return self._http.post(f"{self.base_url}{path}", **kw)
        except httpx.TransportError as e:
            raise ComfyError(f"POST {path}: {e}", transient=True) from e

    @staticmethod
    def _check(r: httpx.Response, ok=(200,)) -> None:
        if r.status_code not in ok:
            raise ComfyError(
                f"HTTP {r.status_code} on {r.request.method} {r.request.url.path}: "
                f"{r.text[:400]}",
                transient=r.status_code >= 500,
                status_code=r.status_code,
            )

    # ------------------------------------------------------------- liveness ----
    def health(self) -> dict:
        r = self._get("/system_stats")
        self._check(r)
        return r.json()

    def object_info(self, class_type: str) -> Optional[dict]:
        """Return the node schema for ``class_type`` (None if unknown)."""
        r = self._get("/object_info/" + class_type)
        if r.status_code == 404:
            return None
        self._check(r)
        data = r.json()
        return data.get(class_type)

    # ------------------------------------------------------------- upload -----
    def upload_image(self, filepath: str, subfolder: str = "input") -> str:
        """Upload an image to ComfyUI; returns the filename (with subfolder)."""
        with open(filepath, "rb") as f:
            files = {"image": (filepath.rsplit("/", 1)[-1], f, "image/png")}
            data = {"type": "input", "overwrite": "true", "subfolder": subfolder}
            try:
                r = self._http.post(f"{self.base_url}/upload/image",
                                    files=files, data=data)
            except httpx.TransportError as e:
                raise ComfyError(f"upload image: {e}", transient=True) from e
        self._check(r)
        j = r.json()
        name = j.get("name", "")
        sub = j.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    # ------------------------------------------------------------- prompt -----
    def submit_prompt(self, graph: dict, client_id: str = "orchestrator") -> str:
        """POST a flat API graph; returns ComfyUI's prompt_id."""
        body = {"prompt": graph, "client_id": client_id}
        r = self._post("/prompt", json=body)
        if r.status_code == 422:
            # Graph/validation error -> permanent; include ComfyUI's detail.
            raise ComfyError(f"ComfyUI rejected prompt: {r.text[:1000]}",
                             transient=False, status_code=422)
        self._check(r)
        return r.json()["prompt_id"]

    def prompt_status(self, prompt_id: str) -> dict:
        """GET /prompt/{id}: {status_info, prompt_metadata} or {} if unknown."""
        r = self._get(f"/prompt/{prompt_id}")
        if r.status_code == 404:
            return {}
        self._check(r)
        return r.json()

    def history(self, prompt_id: str) -> Optional[dict]:
        """GET /history/{id}; None if not done, {} if missing."""
        r = self._get(f"/history/{prompt_id}")
        if r.status_code == 404:
            return None
        self._check(r)
        data = r.json()
        return data.get(prompt_id)

    def wait_for_history(self, prompt_id: str, timeout_s: float = 3600,
                         poll_s: float = 5.0,
                         on_progress=None) -> dict:
        """Poll until history exists (i.e. finished) or timeout.

        Returns the history entry dict. Raises ComfyError(transient=False) on
        timeout and transient=True on transport errors.
        """
        deadline = time.time() + timeout_s
        last = {}
        while time.time() < deadline:
            h = self.history(prompt_id)
            if h is not None and h.get("status"):
                return h
            if h is not None:
                return h  # status may be absent but entry exists => done
            # still running: read queue status for progress
            try:
                ps = self.prompt_status(prompt_id)
                last = ps
                if on_progress and ps.get("status_info"):
                    on_progress(ps["status_info"])
            except ComfyError:
                pass
            time.sleep(poll_s)
        raise ComfyError(f"ComfyUI prompt {prompt_id} timed out after {timeout_s}s",
                         transient=True)

    def cancel(self, prompt_id: str) -> None:
        """Best-effort cancel via /queue delete."""
        try:
            r = self._post("/queue", json={"delete": [prompt_id]})
            self._check(r)
        except ComfyError:
            pass

    # ------------------------------------------------------------- download ---
    def download_output(self, history_entry: dict, dest_dir: str,
                        preferred_suffix: str = ".mp4") -> Optional[str]:
        """Locate the generated video in a history entry and download it.

        History entries carry ``outputs`` as ``{node_id: {images/audio/videos:
        [..]}}``. The H3 graph writes a VIDEO via CreateVideo/SaveVideo; find
        any entry with a video file (or any file with a media suffix) and fetch
        it with /view. Returns the local file path or None.
        """
        outputs = history_entry.get("outputs") or {}
        candidates: list[dict] = []
        for node_id, out in outputs.items():
            for key in ("videos", "audio", "images"):
                for item in out.get(key) or []:
                    candidates.append(item)
        if not candidates:
            return None
        # Prefer the first .mp4/.mkv/.webm.
        chosen = next((c for c in candidates if str(c.get("filename", "")).endswith(preferred_suffix)),
                      candidates[0])
        filename = chosen.get("filename", "")
        subfolder = chosen.get("subfolder", "")
        out_type = chosen.get("type", "output")
        # /view requires filename (+subfolder/type) and returns bytes.
        params = {"filename": filename, "type": out_type}
        if subfolder:
            params["subfolder"] = subfolder
        r = self._http.get(f"{self.base_url}/view", params=params)
        self._check(r)
        import os
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, os.path.basename(filename))
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path