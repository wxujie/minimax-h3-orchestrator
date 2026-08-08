"""Client SDK tests against an in-memory httpx MockTransport (no network)."""
from __future__ import annotations

import httpx
import pytest

from client.sdk import VideoClient, VideoJob


def _handler(store):
    """Return an httpx.MockTransport handler backed by a dict store."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = request.url.path
        if request.method == "POST" and url == "/api/v1/jobs":
            body = request.read().decode()
            jid = f"job-{len(store['jobs'])}"
            store["jobs"][jid] = {"job_id": jid, "status": "QUEUED"}
            return httpx.Response(200, json={"job_id": jid, "status": "QUEUED"})
        for jid, job in store["jobs"].items():
            if url == f"/api/v1/jobs/{jid}":
                return httpx.Response(200, json=job)
            if url == f"/api/v1/jobs/{jid}/cancel":
                job["status"] = "CANCELLED"
                return httpx.Response(200, json={"job_id": jid, "status": "CANCELLED"})
            if url == f"/api/v1/jobs/{jid}/result":
                return httpx.Response(200, content=b"\x00\x00\x00\x18ftypmp42", headers={})
        return httpx.Response(404, json={"detail": "not found"})
    return handler


@pytest.fixture
def client():
    store = {"jobs": {}}
    transport = httpx.MockTransport(_handler(store))
    c = VideoClient("http://ctl.local")
    c._http = httpx.Client(transport=transport)
    return c, store


def test_create_and_get_roundtrip(client):
    c, store = client
    job = c.create_video(prompt="hello world")
    assert isinstance(job, VideoJob)
    assert job.status == "QUEUED"
    # mark it completed server-side then read back
    store["jobs"][job.job_id]["status"] = "COMPLETED"
    got = c.get_job(job.job_id)
    assert got.status == "COMPLETED"


def test_download(client):
    c, store = client
    job = c.create_video(prompt="x")
    store["jobs"][job.job_id]["status"] = "COMPLETED"
    out = c.download(job.job_id, "/tmp/vid-test.mp4")
    assert out.read_bytes().startswith(b"\x00\x00\x00\x18")


def test_cancel(client):
    c, store = client
    job = c.create_video(prompt="x")
    c.cancel(job.job_id)
    assert store["jobs"][job.job_id]["status"] == "CANCELLED"


def test_wait_for_result_times_out(client):
    c, store = client
    job = c.create_video(prompt="x")
    with pytest.raises(TimeoutError):
        c.wait_for_result(job.job_id, timeout=0.05, poll=0.01)