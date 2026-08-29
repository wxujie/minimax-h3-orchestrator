"""Shared fixtures: use a throwaway SQLite DB and the real workflow file.

No Kaggle / Cloudflare / GPU network is touched. The scheduler is exercised in
process via ``Scheduler.tick()`` with a FakeProvider and fake worker clients.
We set module-level env so importing controller.config builds a Settings with
three test accounts and a small max_concurrent_notebooks.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 测试隔离真实 .env：config.py 在 import 时 load_dotenv(.env) 会把本机真实
# 凭据（含 Colab 账号）带进 settings 单例，破坏账号数量断言。这里把
# load_dotenv 打成 no-op，并清除进程环境里残留的 Colab 后端配置。
try:
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: None
    for _k in list(os.environ):
        if _k.startswith("COLAB_ACCOUNT_"):
            os.environ.pop(_k, None)
except ImportError:
    pass

os.environ.setdefault("WORKER_AUTH_SECRET", "test-secret")
os.environ.setdefault("MAX_JOB_RETRIES", "2")
os.environ.setdefault("MAX_CONCURRENT_NOTEBOOKS", "2")
os.environ.setdefault("JOB_OUTPUT_RETENTION_HOURS", "24")
for i in (1, 2, 3):
    os.environ.setdefault(f"KAGGLE_ACCOUNT_{i}_USERNAME", f"acct-user-{i}")
    os.environ.setdefault(f"KAGGLE_ACCOUNT_{i}_KEY", f"acct-key-{i}")


@pytest.fixture(scope="function")
def store(tmp_path: Path):
    from controller import db as _db
    st = _db.Store(f"sqlite:///{tmp_path / 'test.db'}")
    yield st
    st.engine.dispose()


@pytest.fixture(scope="function")
def workflow():
    from controller.workflow import WorkflowAdapter
    return WorkflowAdapter(ROOT / "workflows" / "workflow.json")


@pytest.fixture(scope="function")
def provider():
    from tests.fakes import FakeProvider
    return FakeProvider(ok=True)


@pytest.fixture(scope="function")
def storage(tmp_path: Path):
    """A Storage whose dirs point into tmp_path (never the real ./storage)."""
    from controller.storage import Storage
    st = Storage()
    st.uploads_dir = tmp_path / "uploads"
    st.artifacts_dir = tmp_path / "artifacts"
    st.uploads_dir.mkdir(parents=True, exist_ok=True)
    st.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return st


@pytest.fixture(scope="function")
def managers(store):
    """Real JobManager / WorkerManager / AccountManager over the test store."""
    from controller.accounts import AccountManager
    from controller.jobs import JobManager
    from controller.workers import WorkerManager
    accs = AccountManager(store)
    accs.sync_from_config()
    return {
        "jobs": JobManager(store),
        "workers": WorkerManager(store),
        "accounts": accs,
        "store": store,
    }