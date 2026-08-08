"""Account registry and status.

Accounts are configured from config (credentials loaded from .env, held only
in memory) and are never rendered into public API responses or logs. The
single public listing strips credentials by construction.

Kaggle's official API does NOT expose live GPU/session quota, so each account's
quota is represented as ``UNKNOWN`` and scheduling stays conservative: at most
``max_concurrent_notebooks`` running pool-wide and never more than one notebook
per account until it signs a worker ready.
"""
from __future__ import annotations

from typing import Optional

from . import db
from .config import AccountConfig, settings
from .constants import AccountStatus, NotebookStatus
from .logging_conf import get_logger

log = get_logger("accounts")


def _notebook_public(n) -> dict:
    return {
        "id": n.id,
        "status": n.status,
        "gpu_count": n.gpu_count,
        "kaggle_kernel_slug": n.kaggle_kernel_slug,
        "last_error": n.last_error,
    }


class AccountManager:
    def __init__(self, store: db.Store) -> None:
        self.store = store
        self._config: list[AccountConfig] = (
            settings.accounts if settings else []
        )

    def sync_from_config(self) -> int:
        """Ensure configured accounts exist as rows; returns account count."""
        with self.store.session() as s:
            for ac in self._config:
                row = s.query(db.Account).get(ac.id)
                if row is None:
                    s.add(db.Account(
                        id=ac.id,
                        username=ac.username,
                        enabled=1 if ac.enabled else 0,
                        status=(
                            AccountStatus.ACCOUNT_AVAILABLE.value
                            if ac.enabled
                            else AccountStatus.ACCOUNT_DISABLED.value
                        ),
                    ))
                else:
                    row.enabled = 1 if ac.enabled else 0
                    row.username = ac.username
                    if not ac.enabled:
                        row.status = AccountStatus.ACCOUNT_DISABLED.value
            s.flush()
            return s.query(db.Account).count()

    def list_accounts(self, include_credentials: bool = False) -> list[dict]:
        """Public listing. ``include_credentials`` only ever used internally and
        trims to transient fields; the public routers never pass True."""
        out: list[dict] = []
        with self.store.session() as s:
            for row in s.query(db.Account).order_by(db.Account.id).all():
                d = {
                    "id": row.id,
                    "username": row.username,
                    "enabled": bool(row.enabled),
                    "status": row.status,
                    "quota_status": row.quota_status,
                    "gpu_available": row.quota_gpu_available,
                    "code_usage_hours": row.quota_code_usage_hours,
                    "last_checked_at": row.last_checked_at.isoformat()
                    if row.last_checked_at else None,
                    "notebooks": [_notebook_public(n) for n in row.notebooks],
                }
                if include_credentials:
                    cfg = self._find_config(row.id)
                    d["_credentials"] = (
                        {"username": cfg.username, "key": cfg.key} if cfg else {}
                    )
                out.append(d)
        return out

    def credential(self, account_id: str) -> Optional[AccountConfig]:
        return self._find_config(account_id)

    def _find_config(self, account_id: str) -> Optional[AccountConfig]:
        for a in self._config:
            if a.id == account_id:
                return a
        return None

    def set_status(self, account_id: str, status: AccountStatus, **kw) -> None:
        with self.store.session() as s:
            row = s.query(db.Account).get(account_id)
            if row:
                row.status = status
                for k, v in kw.items():
                    if hasattr(row, k):
                        setattr(row, k, v)

    def any_enabled(self) -> bool:
        with self.store.session() as s:
            return s.query(db.Account).filter(db.Account.enabled == 1).count() > 0

    def running_notebooks(self, account_id: Optional[str] = None) -> int:
        with self.store.session() as s:
            q = s.query(db.Notebook).filter(db.Notebook.status.in_([
                NotebookStatus.NOTEBOOK_STARTING.value,
                NotebookStatus.NOTEBOOK_RUNNING.value,
            ]))
            if account_id:
                q = q.filter(db.Notebook.account_id == account_id)
            return q.count()