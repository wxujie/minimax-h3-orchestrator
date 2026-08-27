"""Colab notebook lifecycle manager (Colab CLI backend).

Wraps the official ``colab`` CLI (google-colab-cli) to run the same
ComfyUI bootstrap + worker runner notebook that Kaggle runs. The worker side
is unchanged: it reads NOTEBOOK_ID / CONTROLLER_PUBLIC_URL /
WORKER_AUTH_SECRET / GPU_COUNT from the environment and registers back to the
controller the same way regardless of which cloud host runs it.

Differences from KaggleManager:

  * Auth is the Colab CLI's local OAuth login state, not a username/key pair.
    No secret ever enters this process or the DB.
  * ``colab new --gpu T4 ...`` starts a runtime explicitly (Kaggle has no
    equivalent) and ``colab stop`` terminates it (Kaggle cannot).
  * The notebook is uploaded as a jupytext ``percent`` script and executed
    with ``colab exec`` (or ``colab run``), rather than pushed as a kernel.

The manager is deliberately lazy about importing jupytext / colab so that
importing this module works even when the Colab CLI is not installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import AccountConfig
from .logging_conf import get_logger

log = get_logger("colab_manager")


@dataclass
class ColabCapacity:
    usable: bool
    gpu_available: Optional[int] = None
    quota_unknown: bool = True
    reason: str = ""


class ColabManager:
    """Thin wrapper over the ``colab`` CLI (official google-colab-cli)."""

    GPU_T4 = "T4"

    def __init__(self, account: AccountConfig,
                 gpu: str = GPU_T4) -> None:
        self.account = account
        self.gpu = gpu
        self._bin = shutil.which("colab")

    # ---------------- env ----------------
    def _cmd(self, args: list[str]) -> list[str]:
        if self._bin:
            return [self._bin] + args
        return [sys.executable, "-m", "colab"] + args

    def _run(self, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
        """Run a colab CLI command, inheriting the user's local OAuth login.

        No credentials are injected: the CLI reads its own session state from
        its default config path. Output is captured for parsing.
        """
        cmd = self._cmd(args)
        log.debug("colab cmd=%s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    # ---------------- capacity ----------------
    def capacity(self) -> ColabCapacity:
        """Best effort: verify the CLI is present and has a login state."""
        if not self._bin:
            return ColabCapacity(
                usable=False,
                quota_unknown=True,
                reason="colab CLI not installed (pip install google-colab-cli)",
            )
        try:
            r = self._run(["sessions"], timeout=60)
            # `colab sessions` returns 0 once logged in; first run prompts
            # for OAuth, which shows up as non-zero or an auth message.
            out = (r.stdout + r.stderr).lower()
            if "visit" in out or "authorize" in out or "oauth" in out:
                return ColabCapacity(
                    usable=False, gpu_available=None, quota_unknown=True,
                    reason="colab CLI not authorized (OAuth login required)",
                )
            ok = r.returncode == 0
            return ColabCapacity(
                usable=ok,
                gpu_available=None,   # UNKNOWN - quota is account/tier based
                quota_unknown=True,
                reason="" if ok else (r.stderr or r.stdout or "colab CLI error"),
            )
        except Exception as e:  # noqa: BLE001
            return ColabCapacity(
                usable=False,
                gpu_available=None,
                quota_unknown=True,
                reason=f"colab CLI error: {e}",
            )

    # ---------------- notebook lifecycle ----------------
    def ensure_notebook(self, slug: str, notebook_json: dict) -> bool:
        """Create-or-attach a Colab session and start the notebook.

        ``slug`` is the session name (the notebook_name from the scheduler).
        ``notebook_json`` is the ipynb document built by notebook_builder
        (the ComfyUI bootstrap + runner cell). We convert it to a percent
        script and execute it on the session.

        Colab does not have a persistent "notebook" artifact the way Kaggle
        kernels do; a session IS the running notebook. So ensure_notebook
        means: ensure a session named ``slug`` exists (start one if not) and
        the bootstrap has been executed.
        """
        if self._session_exists(slug):
            return True
        if not self._start_session(slug):
            return False
        return self._run_notebook(slug, notebook_json)

    def _session_exists(self, slug: str) -> bool:
        try:
            r = self._run(["sessions"], timeout=60)
            return r.returncode == 0 and slug in (r.stdout or "")
        except Exception:
            return False

    def _start_session(self, slug: str) -> bool:
        """``colab new --session <slug> --gpu T4`` and wait until running."""
        try:
            r = self._run(["new", "--session", slug, "--gpu", self.gpu],
                          timeout=180)
            if r.returncode != 0:
                log.warning("colab_new_failed session=%s stderr=%s",
                            slug, r.stderr[:200])
                return False
        except subprocess.TimeoutExpired:
            log.warning("colab_new_timeout session=%s", slug)
            return False
        return self.poll_until_running(slug, timeout_s=300, interval_s=15)

    def _run_notebook(self, slug: str, notebook_json: dict) -> bool:
        """Upload + execute the built notebook as a jupytext script."""
        try:
            import jupytext  # local import; optional dep
            script = jupytext.writes(notebook_json, fmt="py:percent")
        except Exception as e:  # noqa: BLE001
            log.error("jupytext_convert_failed err=%s", e)
            return False

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            py_path = td / "bootstrap.py"
            py_path.write_text(script, encoding="utf-8")
            remote = "/content/bootstrap.py"
            if not self._upload(slug, str(py_path), remote):
                return False
            return self._exec(slug, remote, timeout=300)

    def _upload(self, slug: str, local: str, remote: str) -> bool:
        try:
            r = self._run(["upload", "-s", slug, local, remote], timeout=120)
            return r.returncode == 0
        except Exception:
            return False

    def _exec(self, slug: str, remote: str, timeout: int = 300) -> bool:
        try:
            r = self._run(["exec", "-s", slug, "--timeout", str(float(timeout)),
                           "--file", remote], timeout=timeout + 60)
            return r.returncode == 0
        except Exception:
            return False

    def status(self, slug: str) -> str:
        """Return the session status string via `colab status`."""
        r = self._run(["status", "-s", slug], timeout=60)
        if r.returncode != 0:
            return "unknown"
        low = (r.stdout or r.stderr or "").lower()
        for s in ("running", "idle", "stopped", "error", "starting"):
            if s in low:
                return s
        return "unknown"

    def poll_until_running(self, slug: str, timeout_s: int = 600,
                           interval_s: int = 20) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            st = self.status(slug)
            log.info("colab status slug=%s status=%s", slug, st)
            if st == "running":
                return True
            if st in ("stopped", "error"):
                return False
            time.sleep(interval_s)
        return False

    def stop(self, slug: str) -> bool:
        """``colab stop -s <slug>`` — terminate the session (Kaggle cannot)."""
        try:
            r = self._run(["stop", "-s", slug], timeout=120)
            return r.returncode == 0
        except Exception:
            return False

    # ---------------- limitation shim ----------------
    @staticmethod
    def api_limitations() -> list[str]:
        return [
            "Colab CLI auth is the local OAuth login; no key in env/DB.",
            "GPU availability varies by Colab subscription tier (free T4).",
            "Sessions are transient; the bootstrap re-runs on each start.",
        ]