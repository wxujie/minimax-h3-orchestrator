"""Typed configuration for the controller.

Reads from environment variables and a `.env` file (python-dotenv). Secrets
are never logged; the `redact()` helper scrubs request bodies.

Accounts are declared as KAGGLE_ACCOUNT_<N>_USERNAME / KAGGLE_ACCOUNT_<N>_KEY
pairs plus KAGGLE_ACCOUNT_<N>_ENABLED. The count is discovered by scanning the
environment rather than hard-coded so accounts can be added without code edits.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional
    def load_dotenv(*a, **k) -> None:
        return None


# Load .env from project root (cwd when launched via `uvicorn controller.main`).
_here = Path(__file__).resolve().parent.parent
load_dotenv(_here / ".env")


@dataclass(frozen=True)
class AccountConfig:
    id: str
    username: str
    key: str
    enabled: bool = True


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_accounts() -> List[AccountConfig]:
    """Discover accounts from KAGGLE_ACCOUNT_*_USERNAME/KEY env pairs."""
    accounts: List[AccountConfig] = []
    pattern = re.compile(r"^KAGGLE_ACCOUNT_(\d+)_(USERNAME|KEY|ENABLED)$")
    by_index: dict[str, dict[str, str]] = {}
    for key, val in os.environ.items():
        m = pattern.match(key)
        if not m:
            continue
        idx, field_ = m.group(1), m.group(2)
        by_index.setdefault(idx, {})[field_.lower()] = val
    for idx in sorted(by_index, key=lambda s: int(s)):
        cfg = by_index[idx]
        user = cfg.get("username", "")
        k = cfg.get("key", "")
        if not user:  # incomplete declaration - skip, never fabricate
            continue
        enabled = cfg.get("enabled", "true")
        accounts.append(
            AccountConfig(
                id=f"kaggle-account-{idx}",
                username=user,
                key=k,
                enabled=enabled.strip().lower() in {"1", "true", "yes", "on", ""},
            )
        )
    # Deterministic fallback ordering if only unindexed vars are provided.
    if not accounts:
        accounts = [AccountConfig(id="kaggle-account-1",
                                   username=_env("KAGGLE_USERNAME"),
                                   key=_env("KAGGLE_KEY"))]
    return accounts


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env("CONTROLLER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("CONTROLLER_PORT", 8000))

    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "sqlite:///./storage/controller.db"))

    max_job_retries: int = field(default_factory=lambda: _env_int("MAX_JOB_RETRIES", 2))
    job_output_retention_hours: int = field(
        default_factory=lambda: _env_int("JOB_OUTPUT_RETENTION_HOURS", 24)
    )
    worker_health_timeout_s: int = field(default_factory=lambda: _env_int("WORKER_HEALTH_TIMEOUT_S", 30))
    worker_heartbeat_timeout_s: int = field(default_factory=lambda: _env_int("WORKER_HEARTBEAT_TIMEOUT_S", 60))
    schedule_poll_s: float = field(default_factory=lambda: _env_float("SCHEDULER_POLL_S", 5.0))
    job_poll_s: float = field(default_factory=lambda: _env_float("JOB_POLL_S", 3.0))
    notebook_start_wait_s: int = field(default_factory=lambda: _env_int("NOTEBOOK_START_WAIT_S", 30))
    max_concurrent_notebooks: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_NOTEBOOKS", 2)
    )

    # Worker agent shared secret. Required. Use a single strong secret for the
    # whole pool; a real deployment could derive per-worker secrets from it.
    worker_auth_secret: str = field(default_factory=lambda: _env("WORKER_AUTH_SECRET", ""))
    # Internal callers (controller<->worker) transmit this header value.
    cloudflare_token: str = field(default_factory=lambda: _env("CLOUDFLARE_TOKEN", ""))

    storage_dir: Path = field(default_factory=lambda: Path(_env("STORAGE_DIR", "./storage")))

    workflow_path: Path = field(default_factory=lambda: Path(_env("WORKFLOW_PATH", "./workflows/workflow.json")))
    notebook_path: Path = field(default_factory=lambda: Path(_env("NOTEBOOK_PATH", "./notebooks/minimax-h3-comfyui.ipynb")))
    # Publicly reachable root of THIS controller, used by the pushed notebook to
    # POST /api/v1/agents/register back. Required for real notebook registration;
    # must be reachable from Kaggle (a deployed host or a controller-side tunnel).
    controller_public_url: str = field(
        default_factory=lambda: _env("CONTROLLER_PUBLIC_URL", "")
    )
    # The notebook runner clones this repo to load the worker package + workflow.
    orchestrator_repo_url: str = field(
        default_factory=lambda: _env(
            "ORCHESTRATOR_REPO_URL",
            "https://github.com/msaadakram/minimax-h3-orchestrator",
        )
    )

    # Kaggle credentials, in-memory only.
    accounts: List[AccountConfig] = field(default_factory=_parse_accounts)

    auth_required: bool = field(
        default_factory=lambda: _env_bool("WORKER_AUTH_REQUIRED", True)
    )

    def log_secrets(self) -> list[str]:
        """Everything that must never appear in logs."""
        secrets = [self.worker_auth_secret, self.cloudflare_token]
        secrets += [a.key for a in self.accounts]
        secrets += [a.username for a in self.accounts]
        return [s for s in secrets if s]


# Module-level singleton, but module import must not fail if optional deps are
# missing in a headless context (we only want config for the controller).
try:
    settings = Settings()
except Exception as exc:  # pragma: no cover - rarely called before deps installed
    settings = None  # type: ignore


def redact(text: str, settings: Optional["Settings"] = None) -> str:
    """Replace every configured secret occurrence in `text` with ***REDACTED***."""
    if not text:
        return text
    s = settings or globals().get("settings")
    if s is None:
        return text
    scrubbed = text
    for secret in s.log_secrets():
        if secret:
            scrubbed = scrubbed.replace(secret, "***REDACTED***")
    return scrubbed