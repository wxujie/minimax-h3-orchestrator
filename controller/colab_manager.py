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
import re
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
        # Colab CLI stores its OAuth token at a FIXED path under $HOME
        # (~/.config/colab-cli/token.json) and has no per-account flag. To
        # support multiple Colab accounts we give each account its own HOME
        # so tokens are isolated per account instead of overwriting each other.
        self._home = self._account_home(account)

    @staticmethod
    def _account_home(account: AccountConfig) -> str:
        base = os.environ.get(
            "COLAB_ACCOUNTS_HOME",
            os.path.expanduser("~/.colab-accounts"),
        )
        # account id is safe (config-controlled), but sanitize anyway
        safe_id = re.sub(r"[^\w\-.]", "_", account.id)
        return os.path.join(base, safe_id)

    def _ensure_home(self) -> str:
        """Create the per-account home dir and return it."""
        os.makedirs(self._home, exist_ok=True)
        return self._home

    # ---------------- env ----------------
    def _cmd(self, args: list[str]) -> list[str]:
        if self._bin:
            return [self._bin] + args
        return [sys.executable, "-m", "colab"] + args

    def _run(self, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
        """Run a colab CLI command with the account's isolated HOME.

        No credentials are injected: the CLI reads its own OAuth token from
        ``<account_home>/.config/colab-cli/token.json``. The per-account HOME
        keeps multiple Colab accounts from clobbering each other's login.
        """
        cmd = self._cmd(args)
        log.debug("colab cmd=%s home=%s", " ".join(cmd), self._home)
        env = dict(os.environ)
        env["HOME"] = self._ensure_home()
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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

        The bootstrap (model download + ComfyUI + worker runner) takes
        20-40 minutes, so it is launched **in the background** — exec returns
        immediately after firing it. Completion is signalled by a marker file
        (``/tmp/.bootstrap_done``) written right before the runner blocks. This
        avoids the historical failure where ``colab exec`` timed out after 600s
        and the session was wrongly marked QUOTA_EXHAUSTED even though the
        bootstrap was still downloading models.
        """
        # A session that hasn't finished bootstrapping must be re-drive, not
        # short-circuited: the VM may have died mid-download, or the previous
        # bootstrap may have failed before writing the marker. The bootstrap is
        # idempotent (wget -c resumes) so re-running is cheap relative to a
        # dead worker that never registers.
        if self._session_exists(slug) and self._bootstrap_done(slug):
            return True
        if not self._session_exists(slug):
            if not self._start_session(slug):
                return False
        return self._run_notebook(slug, notebook_json)

    def _session_exists(self, slug: str) -> bool:
        try:
            r = self._run(["sessions"], timeout=60)
            return r.returncode == 0 and slug in (r.stdout or "")
        except Exception:
            return False

    def _bootstrap_done(self, slug: str) -> bool:
        """Check the remote marker file without going through the kernel.

        ``colab ls`` talks to the VM filesystem directly (no Jupyter kernel
        round-trip), so it works even while the long-running bootstrap is still
        occupying the kernel.
        """
        try:
            r = self._run(["ls", "-s", slug, "/tmp/.bootstrap_done"],
                          timeout=60)
            return r.returncode == 0 and ".bootstrap_done" in (r.stdout or "")
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
        """Execute the built notebook as a plain Python script.

        ``colab exec -f`` runs a local ``.py`` file. Jupyter magics / ``!shell``
        lines do NOT survive that conversion (jupytext would comment them out),
        so we translate each code cell into runnable Python instead: ``!cmd``
        becomes ``subprocess.run(["bash", "-lc", cmd], check=True)`` and ``%cd
        dir`` becomes ``os.chdir(dir)``. The remaining code runs as-is.
        """
        import nbformat  # local import; optional dep
        # The Kaggle builder adds a top-level ``kaggle`` key (accelerator hints)
        # that Kaggle's CLI understands but the nbformat v4 schema rejects as an
        # unexpected property. Colab executes the notebook as a plain script, so
        # those hints are meaningless here — strip the key before validation.
        colab_nb = json.loads(json.dumps(notebook_json))
        colab_nb.pop("kaggle", None)
        try:
            doc = nbformat.reads(json.dumps(colab_nb), as_version=4)
        except Exception as e:  # noqa: BLE001
            log.error("notebook_parse_failed err=%s", e)
            return False

        lines: list[str] = [
            "import os, subprocess, sys",
            "",
        ]
        for cell in doc.cells:
            if cell.cell_type != "code":
                continue
            src = cell.source
            if not src.strip():
                continue
            lines.append("")
            raw_lines = src.splitlines()
            i = 0
            while i < len(raw_lines):
                s = raw_lines[i].rstrip()
                if s.startswith("!"):
                    # Join shell line continuations (trailing backslash) into
                    # one command so multi-line wget/apt blocks stay intact.
                    cmd = s[1:].strip()
                    while cmd.endswith("\\") and i + 1 < len(raw_lines):
                        i += 1
                        cmd = cmd[:-1].rstrip() + " " + raw_lines[i].strip()
                    lines.append(
                        f'subprocess.run(["bash", "-lc", {cmd!r}], check=True)')
                elif s.startswith("%cd"):
                    d = s[3:].strip()
                    lines.append(f"os.chdir({d!r})")
                elif s.startswith("%"):
                    # other magics cannot run outside a kernel; drop them
                    pass
                else:
                    lines.append(raw_lines[i])
                i += 1
        script = "\n".join(lines) + "\n"

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            py_path = td / "bootstrap.py"
            py_path.write_text(script, encoding="utf-8")
            # 上传 bootstrap 到远端，再用一个极小 wrapper 后台启动它。
            # 直接 colab exec bootstrap 会同步等 20-40 分钟（模型下载），
            # 期间 controller 的 scheduler 线程整个阻塞，还会在 600s 超时后
            # 把正常冷启动中的 notebook 误标 QUOTA_EXHAUSTED。
            remote_py = "/tmp/bootstrap.py"
            if not self._upload(slug, str(py_path), remote_py):
                return False
            wrapper = td / "launch.py"
            wrapper.write_text(
                "import subprocess, sys\n"
                "subprocess.Popen(\n"
                "    [sys.executable, '/tmp/bootstrap.py'],\n"
                "    start_new_session=True,\n"
                "    stdout=open('/tmp/bootstrap.log', 'w'),\n"
                "    stderr=subprocess.STDOUT,\n"
                ")\n"
                "print('BOOTSTRAP_LAUNCHED')\n",
                encoding="utf-8",
            )
            return self._exec(slug, str(wrapper), timeout=300)

    def _upload(self, slug: str, local: str, remote: str) -> bool:
        try:
            r = self._run(["upload", "-s", slug, local, remote], timeout=180)
            return r.returncode == 0
        except Exception:
            return False

    def _exec(self, slug: str, local_file: str, timeout: int = 300) -> bool:
        """``colab exec -s <slug> -f <local_file>`` — CLI uploads and runs.

        ``-f`` refers to a LOCAL path; the CLI itself handles transferring the
        script to the remote runtime.
        """
        try:
            r = self._run(["exec", "-s", slug, "--timeout", str(float(timeout)),
                           "--file", local_file], timeout=timeout + 60)
            return r.returncode == 0
        except Exception:
            return False

    def status(self, slug: str) -> str:
        """Return the session status string via `colab status`."""
        r = self._run(["status", "-s", slug], timeout=60)
        if r.returncode != 0:
            return "unknown"
        low = (r.stdout or r.stderr or "").lower()
        for s in ("idle", "running", "stopped", "error", "starting", "ready"):
            if s in low:
                return s
        return "unknown"

    def poll_until_running(self, slug: str, timeout_s: int = 600,
                           interval_s: int = 20) -> bool:
        """Poll until the session is usable for code execution.

        Colab reports a healthy idle session as ``IDLE`` (not ``running``),
        and that state accepts ``exec``, so both are treated as ready.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            st = self.status(slug)
            log.info("colab status slug=%s status=%s", slug, st)
            if st in ("idle", "running", "ready"):
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