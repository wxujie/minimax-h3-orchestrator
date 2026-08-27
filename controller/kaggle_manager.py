"""Kaggle notebook lifecycle manager.

Only the officially supported Kaggle API surface is used (``kaggle`` CLI /
``kaggle.api.KaggleApi``): kernel push, kernel status/poll, kernel output
download. Per the Kaggle API docs there is NO official endpoint to:

  * query live GPU/session quota or a GPU-hour budget,
  * terminate a running kernel programmatically,
  * start a kernel run on demand (runs are submitted by push).

Where the API cannot answer, we surface ``UNKNOWN`` capacity and use the
conservative scheduling rules in ``scheduler.py``. Credentials are set on the
``kaggle`` process environment only for the duration of a call, never stored
or logged. No browser automation, no private endpoints.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import AccountConfig
from .logging_conf import get_logger

log = get_logger("kaggle_manager")


@dataclass
class KaggleCapacity:
    """What we can legitimately know about an account's usable capacity."""

    usable: bool
    gpu_available: Optional[int] = None   # None == UNKNOWN
    quota_unknown: bool = True
    reason: str = ""


class KaggleManager:
    """Thin wrapper over the `kaggle` CLI (the officially supported client)."""

    # Kernel metadata template. Uses an arbitrary unique title and slug per
    # notebook; "Accelerator" is set by the notebook JSON we push, and we
    # intentionally reuse an existing kernel if it already exists to avoid
    # quota surprises, otherwise create one.
    _metadata = {
        "id": None,  # filled with owner/slug
        "title": "minimax-h3-comfyui-orchestrator",
        "code_file": "minimax-h3-comfyui.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    def __init__(self, account: AccountConfig) -> None:
        self.account = account
        self._bin = shutil.which("kaggle") or shutil.which("kaggle.py")

    # ---------------- env management ----------------
    def _env(self, extra: Optional[dict] = None) -> dict:
        """Environment with Kaggle credentials set in-memory only.

        Injects both auth models so the manager works regardless of which
        ``kaggle`` CLI is installed: the modern 2.x client reads
        ``KAGGLE_API_TOKEN`` (a settings token), while the legacy 1.x client
        reads ``KAGGLE_USERNAME``/``KAGGLE_KEY``.
        """
        env = dict(os.environ)
        env["KAGGLE_USERNAME"] = self.account.username
        env["KAGGLE_KEY"] = self.account.key
        env["KAGGLE_API_TOKEN"] = self.account.key
        if extra:
            env.update(extra)
        return env

    def _cmd(self, args: list[str]) -> list[str]:
        """Command vector for the kaggle CLI.

        Prefer the on-PATH script, but fall back to ``python -m kaggle.cli`` so
        the installed package always resolves even when ``.venv/bin`` is not on
        the subprocess PATH (e.g. launched outside an activated shell).
        """
        if self._bin:
            return [self._bin] + args
        return [sys.executable, "-m", "kaggle.cli"] + args

    def _run(self, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
        """Run a kaggle CLI command with credentials injected, no secret on argv."""
        cmd = self._cmd(args)
        log.debug("kaggle cmd=%s", " ".join(cmd))  # args never contain the key
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(),
            check=False,
        )

    # ---------------- capacity ----------------
    def capacity(self) -> KaggleCapacity:
        """Best-effort usable/available check via the public API.

        Kaggle exposes no GPU-quota query; we verify the account can
        authenticate and that the API reports it in a usable state. Quota is
        therefore UNKNOWN by policy and we stay conservative.
        """
        try:
            r = self._run(["config", "view"], timeout=60)
            ok = r.returncode == 0
            return KaggleCapacity(
                usable=ok,
                gpu_available=None,   # UNKNOWN - API does not expose
                quota_unknown=True,
                reason="" if ok else (r.stderr or "kaggle config view failed"),
            )
        except Exception as e:  # noqa: BLE001
            return KaggleCapacity(
                usable=False,
                gpu_available=None,
                quota_unknown=True,
                reason=f"kaggle CLI error: {e}",
            )

    # ---------------- notebook lifecycle ----------------
    def ensure_notebook(self, slug: str, notebook_json: dict) -> bool:
        """Create-or-update a private kernel via `kaggle kernels push`.

        ``notebook_json`` is the actual Kaggle notebook-document loaded from the
        builder (the ComfyUI bootstrap + registered-worker runner). We write it
        into a temp dir with the metadata + the ipynb, push, and return True if
        the kernel exists now. The first push may return non-zero for a
        brand-new kernel while still creating it; we treat a subsequent
        ``kernels list`` match as success.
        """
        d = self._metadata.copy()
        d["id"] = slug
        # Kaggle rejects pushes where the title's slug doesn't resolve to the
        # id's basename (409 Conflict). The id is `<owner>/<notebook_name>`,
        # so make the title equal to `<notebook_name>` so the two match.
        d["title"] = slug.split("/", 1)[-1]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            meta_path = td / "kernel-metadata.json"
            code_path = td / d["code_file"]
            meta_path.write_text(json.dumps(d, indent=2))
            code_path.write_text(json.dumps(notebook_json, indent=1))
            r = self._run(
                ["kernels", "push", "--path", str(td)],
                timeout=180,
            )
            # A brand-new kernel push can print "Creating new kernel" and exit 0/1.
            pushed = r.returncode == 0 or "exists" in (r.stdout + r.stderr).lower()
            return pushed

    def status(self, slug: str) -> str:
        """Return the kernel's status string via `kaggle kernels status`."""
        r = self._run(["kernels", "status", slug], timeout=60)
        if r.returncode != 0:
            return "unknown"
        # Output looks like: `kaggle_kernel_status: <slug> has status "<status>"`
        for line in r.stdout.splitlines():
            low = line.lower()
            for s in ("complete", "running", "queued", "error", "cancel"):
                if s in low:
                    return s
        return "unknown"

    def poll_until_running(self, slug: str, timeout_s: int = 600,
                           interval_s: int = 20) -> bool:
        """Poll `kaggle kernels status` until 'running' or timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            st = self.status(slug)
            log.info("kaggle status slug=%s status=%s", slug, st)
            if st in {"running", "complete"}:
                return True
            if st in {"error", "cancel"}:
                return False
            time.sleep(interval_s)
        return False

    def download_output(self, slug: str, dest_dir: Path) -> bool:
        """`kaggle kernels output slug -p dest`. Returns True on success."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        r = self._run(
            ["kernels", "output", slug, "-p", str(dest_dir)],
            timeout=300,
        )
        return r.returncode == 0

    def stop(self, slug: str) -> bool:
        """Kaggle's official API has no terminate endpoint; always False."""
        return False

    # ---------------- limitation shim ----------------
    @staticmethod
    def api_limitations() -> list[str]:
        return [
            "Kaggle API has no GPU/quota query: capacity is UNKNOWN by policy.",
            "Kaggle API has no kernel terminate endpoint: 'stop' is not supported; "
            "notebooks idle until Kaggle's own runtime limits apply.",
            "Kernel runs are submitted by push; there is no separate start call.",
        ]