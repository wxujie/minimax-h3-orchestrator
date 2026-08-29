"""Worker agent route + execution tests.

A fake ComfyClient and a fake (non-downloading) QuickTunnel are injected so no
network/GPU/cloudflared is touched. The agent's FastAPI app is driven with
starlette's TestClient. ``_execute`` runs on a thread; tests poll job status.
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from worker import agent as ag
from worker import cloudflare as cfmodule


class FakeComfy:
    """Scripted ComfyUI stand-in returning a prompt id and a real output file."""

    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.uploaded: list[str] = []

    def health(self) -> dict:
        return {"system_stats": {"system": {"gpu_driver_version": "fake"}}}

    def upload_image(self, filepath: str, subfolder: str = "input") -> str:
        self.uploaded.append(filepath)
        return f"{subfolder}/{os.path.basename(filepath)}"

    def submit_prompt(self, graph: dict) -> str:
        self.submitted.append(graph)
        return "prompt-123"

    def wait_for_history(self, prompt_id, timeout_s=3600, poll_s=5.0,
                         on_progress=None):
        return {"status": {"status_str": "success"}}

    def download_output(self, history, dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        out = os.path.join(dest_dir, "video.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42 fake-video")
        return out

    def cancel(self, prompt_id) -> None:
        pass


class FakeTunnel:
    def __init__(self, *a, **k) -> None:
        self.public_url = "http://abc.trycloudflare.com"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def wait_for_url(self, timeout=60):
        return self.public_url


def _deadline(seconds: float = 6.0) -> float:
    return time.time() + seconds


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(cfmodule, "QuickTunnel", FakeTunnel)
    monkeypatch.setattr(cfmodule, "ensure_cloudflared", lambda *a, **k: "/bin/true")
    spec = ag.WorkerSpec(
        worker_name="worker-0", gpu_index=0,
        comfy_port=8188, agent_port=8899,
        secret="test-secret",
        tunnel_log_path=str(tmp_path / "cf.log"),
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
    )
    os.makedirs(spec.input_dir, exist_ok=True)
    a = ag.WorkerAgent(spec, workflow_factory=lambda **kw: {"graph": kw})

    # inject fakes — never touches a real ComfyUI or downloads cloudflared
    a.comfy = FakeComfy()
    a.tunnel = FakeTunnel()
    return a


TOKEN = {"Authorization": "Bearer test-secret"}


def test_health_leaks_no_secret(agent):
    r = TestClient(agent.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["worker_id"] == "worker-0"
    assert body["status"] in ("ready", "starting")


def test_requires_token(agent):
    c = TestClient(agent.app)
    # body is valid but no token -> 401 (not 422)
    body = {"prompt_text": "x", "duration": 2.0, "first_frame": "frame.png"}
    assert c.post("/jobs", json=body).status_code == 401
    assert c.get("/status").status_code == 401
    assert c.get("/jobs/nope").status_code == 401


def test_job_lifecycle_end_to_end(agent):
    c = TestClient(agent.app)
    up = c.post("/jobs/input", headers=TOKEN,
                files={"file": ("frame.png", b"\x89PNG\x0d\x0a frame")})
    assert up.status_code == 200 and up.json() == {"filename": "frame.png"}

    r = c.post("/jobs", headers=TOKEN, json={
        "prompt_text": "a cat jumping", "duration": 2.0,
        "first_frame": "frame.png",
    })
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]

    deadline = _deadline()
    status = "QUEUED"
    while time.time() < deadline:
        j = c.get(f"/jobs/{jid}", headers=TOKEN).json()
        status = j["status"]
        if status in ("DONE", "FAILED"):
            break
        time.sleep(0.02)
    assert status == "DONE", agent.jobs[jid].error

    res = c.get(f"/jobs/{jid}/result", headers=TOKEN)
    assert res.status_code == 200
    assert res.content.startswith(b"\x00\x00\x00\x18")


def test_upload_stages_sanitized_filename(agent):
    c = TestClient(agent.app)
    up = c.post("/jobs/input", headers=TOKEN,
                files={"file": ("../../escape.png", b"x")})
    assert up.status_code == 200
    assert up.json() == {"filename": "escape.png"}

def test_job_timeout_override_and_fallback(agent):
    """任务级 timeout_s 覆盖 worker 默认；不传则回落 spec.job_timeout_s。"""
    seen = {}
    orig_wait = agent.comfy.wait_for_history

    def recording_wait(prompt_id, timeout_s=3600, poll_s=5.0, on_progress=None):
        seen["timeout"] = timeout_s
        return orig_wait(prompt_id, timeout_s=timeout_s, poll_s=poll_s,
                         on_progress=on_progress)

    agent.comfy.wait_for_history = recording_wait
    c = TestClient(agent.app)

    # stage first_frame（和 end-to-end 测试一样，否则 job 会在 staging 失败）
    up = c.post("/jobs/input", headers=TOKEN,
                files={"file": ("frame.png", b"\x89PNG\x0d\x0a frame")})
    assert up.status_code == 200

    # 1. 带 timeout_s=123 -> 用它
    r = c.post("/jobs", headers=TOKEN, json={
        "prompt_text": "x", "duration": 2.0, "first_frame": "frame.png",
        "timeout_s": 123,
    })
    assert r.status_code == 200, r.text
    jid1 = r.json()["job_id"]
    deadline = _deadline()
    while time.time() < deadline:
        if agent.jobs[jid1].status in ("DONE", "FAILED"):
            break
        time.sleep(0.02)
    assert agent.jobs[jid1].status == "DONE"
    assert seen["timeout"] == 123

    # 2. 不传 -> 回落 spec.job_timeout_s（默认 7200）
    seen.clear()
    r = c.post("/jobs", headers=TOKEN, json={
        "prompt_text": "y", "duration": 2.0, "first_frame": "frame.png",
    })
    assert r.status_code == 200, r.text
    jid2 = r.json()["job_id"]
    deadline = _deadline()
    while time.time() < deadline:
        if agent.jobs[jid2].status in ("DONE", "FAILED"):
            break
        time.sleep(0.02)
    assert agent.jobs[jid2].status == "DONE"
    assert seen["timeout"] == agent.spec.job_timeout_s
