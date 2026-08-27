"""ProviderRegistry dispatch tests — no real Kaggle/Colab calls."""
from __future__ import annotations

from controller.config import AccountConfig
from controller.providers import ProviderRegistry


class _FakeAccounts:
    def __init__(self, accounts):
        self._accounts = {a.id: a for a in accounts}

    def credential(self, account_id):
        return self._accounts.get(account_id)


def test_registry_routes_kaggle_account():
    acct = AccountConfig(id="kaggle-account-1", username="u", key="k",
                         provider="kaggle")
    reg = ProviderRegistry(_FakeAccounts([acct]))

    # monkeypatch kaggle start path
    started = {"v": False}

    def fake_kaggle(ac, name):
        started["v"] = True
        return True

    reg._start_kaggle = fake_kaggle
    assert reg.start_notebook("kaggle-account-1", "nb-x") is True
    assert started["v"] is True


def test_registry_routes_colab_account():
    acct = AccountConfig(id="colab-account-1", username="", key="",
                         provider="colab")
    reg = ProviderRegistry(_FakeAccounts([acct]))

    started = {"v": False}

    def fake_colab(ac, name):
        started["v"] = True
        return True

    reg._start_colab = fake_colab
    assert reg.start_notebook("colab-account-1", "nb-x") is True
    assert started["v"] is True


def test_registry_missing_account_returns_false():
    reg = ProviderRegistry(_FakeAccounts([]))
    assert reg.start_notebook("nope", "nb-x") is False