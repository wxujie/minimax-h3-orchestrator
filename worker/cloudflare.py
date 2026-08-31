"""Cloudflare tunnel management, worker-side.

Two supported modes (selected via ``TUNNEL_MODE`` env on the worker):

  * ``quick``   (default) - ``cloudflared tunnel --url http://127.0.0.1:<port>``
    launches a random ``*.trycloudflare.com`` public URL. The URLs are
    discovered by parsing the cloudflared log (rapid timer, no token needed).
    This is the zero-config mode used by the reference notebook.

  * ``named``   — locally-managed fixed tunnel. Requires the worker env
    CLOUDFLARE_TUNNEL_CONFIG (config.yml) + CLOUDFLARE_TUNNEL_CREDENTIALS
    (credentials.json) + TUNNEL_DOMAIN, all injected per account by the
    controller. The worker runs ``cloudflared tunnel run --config`` with no
    browser login; the public URL is deterministic
    (``https://<worker-id>.<domain>``) so the controller can compute it
    without reading logs, and it survives worker restarts.

Whichever mode, the tunnel fronts ONLY the worker agent (never ComfyUI directly).
The agent's /health + /status mirror the public URL so the controller can verify
the mapping it expects, and auto-re-establish a dead tunnel.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger("worker.cloudflare")

_QUICK_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def ensure_cloudflared(bin_path: str = "/tmp/cloudflared") -> str:
    """Download cloudflared if absent; returns path."""
    if os.path.exists(bin_path):
        return bin_path
    log.info("downloading cloudflared -> %s", bin_path)
    subprocess.run(
        ["wget", "-q",
         "https://github.com/cloudflare/cloudflared/releases/latest/download/"
         "cloudflared-linux-amd64", "-O", bin_path],
        check=True)
    os.chmod(bin_path, 0o755)
    return bin_path


def parse_quick_url(log_path: str) -> Optional[str]:
    """Extract the most recent .trycloudflare.com URL from a quick-tunnel log."""
    try:
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    urls = _QUICK_URL_RE.findall(text)
    return urls[-1] if urls else None


class QuickTunnel:
    """Run/restart a cloudflared quick tunnel and track its public URL."""

    def __init__(self, local_port: int, log_path: str,
                 bin_path: str = "/tmp/cloudflared") -> None:
        self.local_port = local_port
        self.log_path = log_path
        self.bin = ensure_cloudflared(bin_path)
        self.proc: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None

    def start(self) -> None:
        self.stop()
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # Truncate ('w') so parse_quick_url only ever sees URLs from THIS tunnel
        # instance. With append ('a') a prior run's URL lingers in the log and a
        # restart would wrongly report the old (possibly dead) tunnel URL.
        with open(self.log_path, "w", encoding="utf-8") as f:
            self.proc = subprocess.Popen(
                [self.bin, "tunnel", "--url", f"http://127.0.0.1:{self.local_port}",
                 "--no-autoupdate"],
                stdout=f, stderr=subprocess.STDOUT,
            )

    def is_alive(self) -> bool:
        """True while the cloudflared process is running and has a URL."""
        if self.proc is None:
            # We may not have started yet; treat as alive once URL known.
            return bool(self.public_url)
        if self.proc.poll() is not None:
            return False
        if not self.public_url:
            self.public_url = parse_quick_url(self.log_path)
        return bool(self.public_url)

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self.proc = None

    def wait_for_url(self, timeout_s: int = 60) -> Optional[str]:
        """Poll the log until a URL appears. Re-start once if it stalls."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            url = parse_quick_url(self.log_path)
            if url:
                self.public_url = url
                return url
            if self.proc and self.proc.poll() is not None:
                log.warning("cloudflared died; restarting")
                self.start()
            time.sleep(2)
        urls = parse_quick_url(self.log_path)
        if urls:
            self.public_url = urls
            return urls
        return None


def named_tunnel_url(sub_id: str, domain: str) -> str:
    """Deterministic public URL for a named tunnel  with one known hostname."""
    return f"https://{sub_id}.{domain}"


class NamedTunnel:
    """Run a Cloudflare *locally-managed* named tunnel on a worker.

    A locally-managed tunnel needs BOTH pieces (unlike remotely-managed
    ``--token``, whose ingress lives in the dashboard):

      * ``credentials.json`` — the tunnel's JWT credentials (equiv. token);
      * ``config.yml`` — ingress rules mapping the public hostname to the
        local origin port (without this, cloudflared falls back to :8080).

    Both are generated once on a trusted machine by
    ``scripts/create-worker-tunnel.sh`` and injected into the worker as env
    values; the worker writes them to disk and runs
    ``cloudflared tunnel run --config <config.yml>``. No browser login, no
    ``cert.pem``, no ``~/.cloudflared`` state on the worker.

    The public URL is deterministic — ``https://<worker-id>.<domain>`` — so
    the controller can compute it without parsing logs, and it survives
    worker restarts.
    """

    CRED_PATH = "/tmp/cloudflared-tunnel/credentials.json"
    CONFIG_PATH = "/tmp/cloudflared-tunnel/config.yml"

    def __init__(self, config_content: str, credentials_content: str,
                 log_path: str, public_url: str,
                 bin_path: str = "/tmp/cloudflared") -> None:
        if not config_content or not credentials_content:
            raise ValueError(
                "named tunnel requires both config.yml and credentials content")
        if not public_url:
            raise ValueError("named tunnel requires a deterministic public_url")
        self.config_content = config_content
        self.credentials_content = credentials_content
        self.log_path = log_path
        self.public_url = public_url
        self.bin = ensure_cloudflared(bin_path)
        self.proc: Optional[subprocess.Popen] = None

    def _write_files(self) -> None:
        os.makedirs(os.path.dirname(self.CRED_PATH), exist_ok=True)
        with open(self.CRED_PATH, "w", encoding="utf-8") as f:
            f.write(self.credentials_content)
        os.makedirs(os.path.dirname(self.CONFIG_PATH), exist_ok=True)
        with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(self.config_content)

    def _cmd(self) -> list[str]:
        return [
            self.bin, "tunnel", "run",
            "--config", self.CONFIG_PATH,
            "--no-autoupdate",
        ]

    def start(self) -> None:
        self.stop()
        self._write_files()
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            self.proc = subprocess.Popen(
                self._cmd(), stdout=f, stderr=subprocess.STDOUT,
            )

    def is_alive(self) -> bool:
        if self.proc is None:
            return bool(self.public_url)
        if self.proc.poll() is not None:
            return False
        # 进程活着 ≠ 隧道连上了：必须看到 cloudflared 日志里的
        # "Registered tunnel connection" 才算真正连上 Cloudflare 边缘。
        # 凭证错误/边缘拒绝时进程会持续重试不退出，process-alive 会误判。
        return self._registered_in_log()

    def _registered_in_log(self) -> bool:
        try:
            with open(self.log_path, "r", errors="ignore") as f:
                text = f.read()
        except FileNotFoundError:
            return False
        return "Registered tunnel connection" in text

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self.proc = None

    def wait_for_url(self, timeout_s: int = 60) -> Optional[str]:
        """Wait until cloudflared reports a successful edge registration.

        Named tunnels have a deterministic URL, but the URL is only reachable
        once cloudflared actually connects to the edge. A live process is NOT
        enough (bad credentials make it retry forever while staying alive),
        so we poll the log for "Registered tunnel connection".
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                log.warning("named cloudflared exited early; restarting")
                self.start()
            if self.proc and self.proc.poll() is None and self._registered_in_log():
                return self.public_url
            time.sleep(2)
        # 超时但进程还活着：大概率凭证/配置有问题，日志里有真实报错。
        tail = ""
        try:
            with open(self.log_path, "r", errors="ignore") as f:
                tail = "".join(f.readlines()[-5:])
        except Exception:
            pass
        if self.proc is not None and self.proc.poll() is None:
            log.error("named_tunnel_no_registration tail=%s", tail.strip()[:400])
        return None