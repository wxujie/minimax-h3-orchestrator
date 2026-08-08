"""Jobs REST router exercised directly (no scheduler thread, no network).

The module-level singletons ``main._store`` / ``main._storage_singleton`` are
pointed at the throwaway store/storage, then the route functions are called as
plain functions.
"""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException


@pytest.fixture(scope="function")
def api_ctx(managers, storage, monkeypatch):
    from controller import main as pmain
    monkeypatch.setattr(pmain, "_store", managers["store"], raising=False)
    monkeypatch.setattr(pmain, "_storage_singleton", storage, raising=False)
    return pmain


class _UF:
    """UploadFile stand-in exposing ``.filename`` and ``.file.read()``."""
    def __init__(self, filename, content=b""):
        self.filename = filename
        self._io = io.BytesIO(content or b"\x89PNG upload-bytes")

    @property
    def file(self):
        return self._io


def test_json_job_lifecycle(api_ctx):
    from controller.api import jobs as api
    from controller.api.jobs import JobCreate
    created = api.create_job_json(JobCreate(prompt="a bird", duration=2.0))
    jid = created["job_id"]
    assert created["position"] == 0
    got = api.get_job(jid)
    assert got["status"] == "QUEUED"
    assert got["download_url"] is None
    api.cancel_job(jid)
    assert api.get_job(jid)["status"] == "CANCELLED"


def test_multipart_upload_stages_file(api_ctx, storage):
    from controller.api import jobs as api
    created = api.create_job_multipart(
        prompt="zoom", duration=2.0, first_frame=_UF("clip.png"),
        last_frame=None, width=None, height=None, priority=0,
        # mirror FastAPI's DI-injected values for the Form/File defaults
    )
    jid = created["job_id"]
    assert api.get_job(jid)["status"] == "QUEUED"
    # the upload was persisted to the storage uploads dir
    assert storage.upload_path("clip.png").exists()


def test_create_honors_max_job_retries(api_ctx, managers, monkeypatch):
    from controller import db
    from controller.config import settings
    from controller.api import jobs as api
    from controller.api.jobs import JobCreate
    monkeypatch.setattr(settings, "max_job_retries", 9)
    jid = api.create_job_json(JobCreate(prompt="x"))["job_id"]
    with managers["store"].session() as s:
        assert s.query(db.Job).get(jid).max_retries == 9


def test_result_requires_completed(api_ctx):
    from controller.api import jobs as api
    from controller.api.jobs import JobCreate
    jid = api.create_job_json(JobCreate(prompt="x"))["job_id"]
    with pytest.raises(HTTPException) as e:
        api.get_result(jid)
    assert e.value.status_code == 409