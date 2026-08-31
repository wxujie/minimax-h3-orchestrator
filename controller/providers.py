"""Notebook provider registry.

The scheduler talks to a single ``provider`` object whose only job is
``start_notebook(account_id, notebook_name) -> bool``. This module turns that
into a multi-provider dispatch: each configured account declares which backend
it uses (``provider="kaggle"`` or ``provider="colab"``), and the registry routes
``start_notebook`` to the matching manager.

Both managers implement the same ``NotebookProvider`` protocol so capacity /
status / lifecycle calls stay uniform:

    capacity()                 -> ProviderCapacity
    ensure_notebook(slug, doc) -> bool
    status(slug)               -> str  (\"running\" / \"stopped\" / ...)
    poll_until_running(...)    -> bool
    stop(slug)                 -> bool  (Colab can; Kaggle cannot)

The worker side is platform-agnostic: whichever provider actually starts the
notebook, the injected runner cell reads the same NOTEBOOK_ID /
CONTROLLER_PUBLIC_URL / WORKER_AUTH_SECRET env and registers back the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .accounts import AccountManager
from .config import AccountConfig
from .kaggle_manager import KaggleManager
from .logging_conf import get_logger

log = get_logger("providers")


@dataclass
class ProviderCapacity:
    """What we can know about a provider account's usable capacity."""

    usable: bool
    gpu_available: Optional[int] = None   # None == UNKNOWN
    quota_unknown: bool = True
    reason: str = ""


@runtime_checkable
class NotebookProvider(Protocol):
    """Uniform lifecycle surface a backend manager must implement."""

    def capacity(self) -> ProviderCapacity: ...

    def ensure_notebook(self, slug: str, notebook_doc: dict) -> bool: ...

    def status(self, slug: str) -> str: ...

    def poll_until_running(self, slug: str, timeout_s: int = 600,
                           interval_s: int = 20) -> bool: ...

    def stop(self, slug: str) -> bool: ...


class ProviderRegistry:
    """Routes ``start_notebook`` to the manager matching an account's provider.

    Managers are cached per account id so the underlying CLI login state /
    credential env is reused across starts.
    """

    def __init__(self, accounts: AccountManager) -> None:
        self.accounts = accounts
        self._kaggle: dict[str, KaggleManager] = {}
        self._colab: dict[str, object] = {}   # ColabManager, imported lazily

    def start_notebook(self, account_id: str, notebook_name: str) -> bool:
        ac = self.accounts.credential(account_id)
        if ac is None:
            log.warning("provider_account_missing account=%s", account_id)
            return False
        if ac.provider == "colab":
            return self._start_colab(ac, notebook_name)
        return self._start_kaggle(ac, notebook_name)

    @staticmethod
    def _build_body(notebook_name: str, gpu_count: int = 2,
                    account: Optional[AccountConfig] = None):
        """Build the registerable ipynb document (shared by both providers).

        ``gpu_count`` tells the runner how many GPU workers to bring up on the
        host. Kaggle notebooks get 2 T4s; a Colab session has 1.

        When ``account`` provides named-tunnel credentials (locally-managed
        Cloudflare tunnel), those are injected per worker so the worker runs a
        deterministic public URL without any browser login.
        """
        from .config import settings
        from .notebook_builder import build_notebook  # local import

        if settings is None or not settings.controller_public_url:
            log.warning("controller_public_url not configured; cannot build "
                        "registerable notebook (set CONTROLLER_PUBLIC_URL)")
            return None
        tunnel_config = (account.tunnel_config if account else "") or ""
        tunnel_credentials = (account.tunnel_credentials if account else "") or ""
        return build_notebook(
            notebook_id=notebook_name,
            controller_public_url=settings.controller_public_url,
            worker_auth_secret=settings.worker_auth_secret,
            repo_url=settings.orchestrator_repo_url,
            template_path=settings.notebook_path,
            gpu_count=gpu_count,
            job_timeout_s=settings.job_timeout_s,
            tunnel_mode=settings.tunnel_mode,
            tunnel_domain=settings.tunnel_domain,
            cloudflare_tunnel_config=tunnel_config,
            cloudflare_tunnel_credentials=tunnel_credentials,
        )

    # ------------------------------------------------------------- kaggle ---
    def _start_kaggle(self, ac: AccountConfig, notebook_name: str) -> bool:
        try:
            mgr = self._kaggle.get(ac.id) or KaggleManager(ac)
            self._kaggle[ac.id] = mgr
            cap = mgr.capacity()
            if not cap.usable:
                log.warning("kaggle_capacity_unusable account=%s reason=%s",
                            ac.id, cap.reason)
                return False
            body = self._build_body(notebook_name, account=ac)
            if body is None:
                return False
            slug = f"{ac.username}/{notebook_name}"
            return mgr.ensure_notebook(slug, body)
        except Exception as exc:  # noqa: BLE001
            log.error("kaggle_start_notebook_failed account=%s err=%s",
                      ac.id, exc)
            return False

    # -------------------------------------------------------------- colab ---
    def _start_colab(self, ac: AccountConfig, notebook_name: str) -> bool:
        try:
            mgr = self._colab.get(ac.id)
            if mgr is None:
                from .colab_manager import ColabManager  # lazy: colab CLI may be absent
                mgr = ColabManager(ac)
                self._colab[ac.id] = mgr
            cap = mgr.capacity()
            if not cap.usable:
                log.warning("colab_capacity_unusable account=%s reason=%s",
                            ac.id, cap.reason)
                return False
            body = self._build_body(notebook_name, gpu_count=1, account=ac)
            if body is None:
                return False
            # Colab notebooks are addressed by their document name, no user slug.
            return mgr.ensure_notebook(notebook_name, body)
        except Exception as exc:  # noqa: BLE001
            log.error("colab_start_notebook_failed account=%s err=%s",
                      ac.id, exc)
            return False