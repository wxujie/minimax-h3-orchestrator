"""Account management endpoints (never expose credentials)."""
from __future__ import annotations

from fastapi import APIRouter

from .. import main as pmain
from ..accounts import AccountManager

router = APIRouter()


def _accounts() -> AccountManager:
    store = pmain.get_store()
    return AccountManager(store)


@router.get("/accounts")
def list_accounts():
    # Public listing always strips credentials.
    return _accounts().list_accounts(include_credentials=False)