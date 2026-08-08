"""Cloudflare tunnel management, worker-side.

Two supported modes (selected via ``TUNNEL_MODE`` env on the worker):

  * ``quick``   (default) - ``cloudflared tunnel --url http://127.0.0.1:<port>``
    launches a random ``*.trycloudflare.com`` public URL. The URLs are
    discovered by parsing the cloudflared log (rapid timer, no token needed).
    This is the zero-config mode used by the reference notebook.

  * ``named``   — requires CLOUDFLARE_TOKEN + TUNNEL_DOMAIN. The worker runs a
    *named* tunnel and (re)connects it; the public URL is deterministic
    (``https://<worker-id>.<domain>``) so the controller can compute it without
    reading logs. Best for production: stable URLs survive restarts.

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
        with open(self.log_path, "a", encoding="utf-8") as f:
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