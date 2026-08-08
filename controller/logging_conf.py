"""Structured logging setup.

Logs key=value pairs on a single line:
2026-08-08T16:00:01Z INFO: scheduler job_created job_id=abc123

A ``RedactingFormatter`` formats the record normally and then scrubs every
configured secret from the *final output line*, so stdlib-style lazy ``%``
formatting (``log.info("job=%s", id)``) keeps working and secrets never hit any
handler even if a caller forgets ``redact()``.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from .config import redact, settings


class RedactingFormatter(logging.Formatter):
    """Format one line, then strip configured secrets from that line."""

    def __init__(self, fmt: str, datefmt: str) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enabled = bool(settings)

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)   # <- one pass; % and args handled here
        return redact(line, settings) if self.enabled else line


_DEF_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _make_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingFormatter(_DEF_FMT, "%Y-%m-%dT%H:%M:%SZ"))
    return handler


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if getattr(root, "_minimax_configured", False):
        return
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(_make_handler())
    root._minimax_configured = True  # type: ignore


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")