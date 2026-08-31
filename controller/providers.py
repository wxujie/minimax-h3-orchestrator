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

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from .accounts import AccountManager
from .config import AccountConfig, settings
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
    def _ensure_named_tunnel(account: AccountConfig, notebook_name: str,
                             gpu_count: int, agent_port_base: int) -> Optional[AccountConfig]:
        """为 named 模式自动检测/补建隧道凭证。

        返回一个新的 AccountConfig（可能带上了刚生成的 tunnel 凭证）；
        若无法生成（未登录 cloudflared、脚本缺失等），返回 None 并记日志，
        由调用方决定是否中止本次启动。
        """
        if settings is None or settings.tunnel_mode != "named":
            return account
        # 已有凭证则直接用
        if account.tunnel_config and account.tunnel_credentials:
            return account
        script = Path(__file__).resolve().parent.parent / "scripts" / "create-worker-tunnel.sh"
        if not script.exists():
            log.error("named_tunnel_script_missing path=%s", script)
            return None
        if not settings.tunnel_domain:
            log.error("TUNNEL_MODE=named 但 TUNNEL_DOMAIN 未配置")
            return None
        log.info("named_tunnel_auto_create account=%s notebook=%s gpu_count=%s",
                 account.id, notebook_name, gpu_count)
        try:
            r = subprocess.run(
                [str(script), account.id, notebook_name, str(gpu_count),
                 str(agent_port_base), settings.tunnel_domain],
                capture_output=True, text=True, timeout=600,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("named_tunnel_auto_create_failed account=%s err=%s",
                      account.id, exc)
            return None
        if r.returncode != 0:
            # 未登录 cloudflared 会走这里（脚本 exit 1）
            log.error("named_tunnel_auto_create_failed account=%s rc=%s stderr=%s",
                      account.id, r.returncode, (r.stderr or "").strip()[:500])
            return None
        # 脚本成功后，从产物目录读取凭证（比解析 .env 多行值更可靠）
        script_dir = script.parent
        cfg_parts: list[str] = []
        cred_parts: list[str] = []
        for gpu in range(gpu_count):
            wid = f"{notebook_name}-gpu{gpu}"
            out_dir = script_dir / ".tunnels" / wid
            cfg_path = out_dir / "config.yml"
            cred_path = out_dir / "credentials.json"
            if not cfg_path.exists() or not cred_path.exists():
                log.error("named_tunnel_artifact_missing worker=%s", wid)
                return None
            cfg_parts.append(cfg_path.read_text(encoding="utf-8"))
            cred_parts.append(cred_path.read_text(encoding="utf-8").strip())
        if not cfg_parts:
            return None
        # 合并多 worker 的 config（一个 config 文件带多个 hostname 规则）
        merged_config = "\n".join(cfg_parts)
        merged_creds = cred_parts[0] if len(cred_parts) == 1 else "\n".join(cred_parts)
        from dataclasses import replace
        return replace(account,
                       tunnel_config=merged_config,
                       tunnel_credentials=merged_creds)

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
            gpu_count = 1  # 单卡（2026-08-30 已改为单卡单 worker）
            if settings is not None and settings.tunnel_mode == "named":
                ac2 = self._ensure_named_tunnel(ac, notebook_name, gpu_count, 8000)
                if ac2 is None:
                    log.error("named_tunnel_unavailable account=%s; 请先登录 cloudflared", ac.id)
                    return False
                ac = ac2
            mgr = self._kaggle.get(ac.id) or KaggleManager(ac)
            self._kaggle[ac.id] = mgr
            cap = mgr.capacity()
            if not cap.usable:
                log.warning("kaggle_capacity_unusable account=%s reason=%s",
                            ac.id, cap.reason)
                return False
            body = self._build_body(notebook_name, gpu_count=gpu_count, account=ac)
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
            gpu_count = 1
            if settings is not None and settings.tunnel_mode == "named":
                ac2 = self._ensure_named_tunnel(ac, notebook_name, gpu_count, 8000)
                if ac2 is None:
                    log.error("named_tunnel_unavailable account=%s; 请先登录 cloudflared", ac.id)
                    return False
                ac = ac2
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
            body = self._build_body(notebook_name, gpu_count=gpu_count, account=ac)
            if body is None:
                return False
            # Colab notebooks are addressed by their document name, no user slug.
            return mgr.ensure_notebook(notebook_name, body)
        except Exception as exc:  # noqa: BLE001
            log.error("colab_start_notebook_failed account=%s err=%s",
                      ac.id, exc)
            return False