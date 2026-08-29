"""ColabManager unit tests — fake the colab CLI, no real network/auth."""
from __future__ import annotations

from controller.colab_manager import ColabManager
from controller.config import AccountConfig


def _acct(provider="colab"):
    return AccountConfig(id="colab-account-1", username="", key="",
                         provider=provider)


def test_capacity_when_cli_missing(monkeypatch):
    mgr = ColabManager(_acct())
    monkeypatch.setattr(mgr, "_bin", None)
    cap = mgr.capacity()
    assert cap.usable is False
    assert "not installed" in cap.reason


def test_capacity_when_authorized(monkeypatch):
    mgr = ColabManager(_acct())
    monkeypatch.setattr(mgr, "_bin", "/usr/bin/colab")

    class _R:
        returncode = 0
        stdout = "session-1  T4  running"
        stderr = ""

    monkeypatch.setattr(mgr, "_run", lambda args, timeout=180: _R())
    cap = mgr.capacity()
    assert cap.usable is True
    assert cap.gpu_available is None  # UNKNOWN
    assert cap.quota_unknown is True


def test_capacity_when_not_authorized(monkeypatch):
    mgr = ColabManager(_acct())
    monkeypatch.setattr(mgr, "_bin", "/usr/bin/colab")

    class _R:
        returncode = 1
        stdout = ""
        stderr = "visit https://accounts.google.com/o/oauth2/auth"

    monkeypatch.setattr(mgr, "_run", lambda args, timeout=180: _R())
    cap = mgr.capacity()
    assert cap.usable is False
    assert "not authorized" in cap.reason


def test_ensure_notebook_reuses_completed_session(monkeypatch):
    """Session 存在且 bootstrap marker 已写 -> 直接复用，不重启、不重跑。"""
    mgr = ColabManager(_acct())
    monkeypatch.setattr(mgr, "_session_exists", lambda slug: True)
    monkeypatch.setattr(mgr, "_bootstrap_done", lambda slug: True)
    called_start = {"v": False}
    monkeypatch.setattr(mgr, "_start_session", lambda slug: called_start.update(v=True))
    ran = {"v": False}
    monkeypatch.setattr(mgr, "_run_notebook",
                        lambda slug, nb: ran.update(v=True) or True)
    assert mgr.ensure_notebook("nb-x", {"cells": []}) is True
    assert called_start["v"] is False
    assert ran["v"] is False  # 不重跑 bootstrap


def test_ensure_notebook_redrives_incomplete_session(monkeypatch):
    """Session 存在但 marker 缺失（上次 bootstrap 失败/没跑完）-> 重跑 bootstrap。

    这是历史 bug 的回归测试：以前 `_session_exists` 为真就直接 return True，
    空壳 session 被当成就绪，worker 永远不注册。
    """
    mgr = ColabManager(_acct())
    monkeypatch.setattr(mgr, "_session_exists", lambda slug: True)
    monkeypatch.setattr(mgr, "_bootstrap_done", lambda slug: False)
    called_start = {"v": False}
    monkeypatch.setattr(mgr, "_start_session", lambda slug: called_start.update(v=True))
    ran = {"v": False}
    monkeypatch.setattr(mgr, "_run_notebook",
                        lambda slug, nb: ran.update(v=True) or True)
    assert mgr.ensure_notebook("nb-x", {"cells": []}) is True
    assert called_start["v"] is False  # session 在，不重复 new
    assert ran["v"] is True  # 但必须重跑 bootstrap


def test_ensure_notebook_starts_new_session(monkeypatch):
    mgr = ColabManager(_acct())
    monkeypatch.setattr(mgr, "_session_exists", lambda slug: False)
    monkeypatch.setattr(mgr, "_start_session", lambda slug: True)
    ran = {"v": False}
    monkeypatch.setattr(mgr, "_run_notebook",
                        lambda slug, nb: ran.update(v=True) or True)
    assert mgr.ensure_notebook("nb-x", {"cells": []}) is True
    assert ran["v"] is True


def test_start_session_new_command(monkeypatch):
    mgr = ColabManager(_acct())
    calls = []

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, timeout=180):
        calls.append(args)
        return _R()

    monkeypatch.setattr(mgr, "_run", fake_run)
    monkeypatch.setattr(mgr, "poll_until_running", lambda *a, **k: True)
    assert mgr._start_session("nb-x") is True
    assert calls and calls[0][:4] == ["new", "--session", "nb-x", "--gpu"]


def test_status_parses_running(monkeypatch):
    mgr = ColabManager(_acct())

    class _R:
        returncode = 0
        stdout = "Session nb-x is running"
        stderr = ""

    monkeypatch.setattr(mgr, "_run", lambda args, timeout=60: _R())
    assert mgr.status("nb-x") == "running"


def test_stop_returns_true(monkeypatch):
    mgr = ColabManager(_acct())

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(mgr, "_run", lambda args, timeout=120: _R())
    assert mgr.stop("nb-x") is True


def test_api_limitations_mentions_oauth():
    lims = ColabManager.api_limitations()
    joined = " ".join(lims).lower()
    assert "oauth" in joined


def test_account_home_isolation():
    a1 = AccountConfig(id="colab-account-1", username="", key="", provider="colab")
    a2 = AccountConfig(id="colab-account-2", username="", key="", provider="colab")
    h1 = ColabManager._account_home(a1)
    h2 = ColabManager._account_home(a2)
    assert h1 != h2
    assert "colab-account-1" in h1
    assert "colab-account-2" in h2