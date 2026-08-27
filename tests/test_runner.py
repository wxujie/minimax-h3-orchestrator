"""Tests for the notebook-side runner wiring (worker/runner.py).

Focus is the pieces that make a real notebook register with the controller:
the workflow factory bound to the shared adapter, the register POST contract,
and that the worker id scheme matches the placeholder rows the scheduler
provisions (so register() upserts the right row).
"""
from __future__ import annotations

from pathlib import Path

from controller.constants import worker_id
from worker.runner import build_workflow_factory

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_factory_builds_real_prompt(workflow):
    """The factory the notebook uses must emit a MiniMax-H3 submit-ready graph."""
    factory = build_workflow_factory(repo_root=ROOT)
    graph = factory(prompt_text="a cat", duration=2.0,
                    first_frame="frame.png", width=1344, height=768, seed=7)
    types = {v.get("class_type") for v in graph.values()}
    assert "MiniMaxH3ImageToVideo" in types
    assert "UNETLoader" in types and "CLIPLoader" in types
    assert "SaveVideo" in types
    assert graph["104"]["inputs"]["prompt"] == "a cat"
    assert graph["104"]["inputs"]["first_frame"] == ["_load_first", 0]


def test_workflow_factory_requires_first_frame(workflow):
    factory = build_workflow_factory(repo_root=ROOT)
    try:
        factory(prompt_text="no frame")
    except Exception as exc:
        assert "first_frame" in str(exc)
    else:
        raise AssertionError("expected WorkflowError for missing first frame")


def test_worker_id_matches_provisioned_slots(managers):
    """runner's worker_id must equal the placeholder row the scheduler creates."""
    store = managers["store"]
    # scheduler provisions worker rows named "<notebook_id>-gpu<g>" after it
    # records the notebook as STARTING (provisioned() only fills missing rows).
    from controller import db
    from controller.constants import NotebookStatus
    from controller.workers import WorkerManager
    nb = "nb-kaggle-account-1"
    with store.session() as s:
        s.merge(db.Notebook(id=nb, account_id="kaggle-account-1", gpu_count=2,
                            status=NotebookStatus.NOTEBOOK_STARTING.value))
        s.flush()
    WorkerManager(store).provisioned(nb, 2)
    with store.session() as s:
        rows = {w.id for w in s.query(db.Worker).all()}
    assert rows == {worker_id("nb-kaggle-account-1", g) for g in (0, 1)}


def test_register_posts_correct_contract(monkeypatch):
    from worker.runner import _register

    calls: list[tuple[str, dict, dict]] = []

    class _FakePost:
        def __init__(self, status_code=200):
            self.status_code = status_code

    def fake_post(url, json, headers, timeout, trust_env):
        calls.append((url, json, headers))
        return _FakePost(200)

    monkeypatch.setattr("worker.runner.httpx.post", fake_post)

    ok = _register("https://ctrl.example.com/", "tok", worker_id="nb-1-gpu0",
                   notebook_id="nb-1", gpu=0, tunnel_url="https://w.trycloudflare.com")
    assert ok is True
    url, body, headers = calls[0]
    assert url == "https://ctrl.example.com/api/v1/agents/register"
    assert body == {"worker_id": "nb-1-gpu0", "notebook_id": "nb-1",
                    "gpu": 0, "tunnel_url": "https://w.trycloudflare.com"}
    assert headers["Authorization"] == "Bearer tok"


def test_register_handles_transient_failure(monkeypatch):
    from worker.runner import _register

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("worker.runner.httpx.post", boom)
    assert _register("https://c/", "", worker_id="w", notebook_id="nb",
                     gpu=0, tunnel_url="u") is False


class _Proc:
    def __init__(self, dead: bool):
        self.dead = dead

    def poll(self):
        return 1 if self.dead else None


class _FakeTunnel:
    def __init__(self, *, alive: bool, dead_proc: bool, pub: str):
        self._alive = alive
        self.proc = _Proc(dead_proc)
        self.public_url = pub

    def is_alive(self):
        return self._alive


class _FakeAgent:
    def __init__(self, tunnel):
        self.tunnel = tunnel
        self.starts = 0

    def start_tunnel(self, timeout=40):
        self.starts += 1
        self.tunnel._alive = True
        self.tunnel.proc = _Proc(False)
        self.tunnel.public_url = "https://new.trycloudflare.com"
        return self.tunnel.public_url


def _gate(tunnel, *, registered):
    return {
        "agent": _FakeAgent(tunnel), "worker_id": "nb-1-gpu0",
        "notebook_id": "nb-1", "gpu": 0,
        "controller_url": "https://c.example.com", "secret": "tok",
        "url": tunnel.public_url, "registered": registered,
    }


def test_reconcile_restarts_dead_tunnel(monkeypatch):
    """A dead cloudflared must be restarted and re-registered with the NEW url."""
    from worker.runner import _reconcile_worker
    calls = []
    monkeypatch.setattr("worker.runner._register",
                        lambda *a, tunnel_url=None, **k: calls.append(tunnel_url) or True)
    gate = _gate(_FakeTunnel(alive=False, dead_proc=True, pub="https://old.example.com"),
                 registered=True)
    ok = _reconcile_worker(gate)
    assert ok is True
    assert gate["agent"].starts == 1, "dead tunnel must be restarted"
    # must not reuse the old dead URL
    assert calls and calls[0] == "https://new.trycloudflare.com"
    assert gate["registered"] is True


def test_reconcile_reregisters_healthy_unregistered(monkeypatch):
    """Healthy tunnel whose registration never landed is re-registered, not restarted."""
    from worker.runner import _reconcile_worker
    calls = []
    monkeypatch.setattr("worker.runner._register",
                        lambda *a, tunnel_url=None, **k: calls.append(tunnel_url) or True)
    gate = _gate(_FakeTunnel(alive=True, dead_proc=False, pub="https://live.example.com"),
                 registered=False)
    ok = _reconcile_worker(gate)
    assert ok is True
    assert gate["agent"].starts == 0, "healthy tunnel must not be restarted"
    assert calls == ["https://live.example.com"]  # existing URL re-posted


def test_reconcile_noop_when_healthy_and_registered(monkeypatch):
    from worker.runner import _reconcile_worker
    calls = []
    monkeypatch.setattr("worker.runner._register",
                        lambda *a, tunnel_url=None, **k: calls.append(tunnel_url) or True)
    gate = _gate(_FakeTunnel(alive=True, dead_proc=False, pub="https://live.example.com"),
                 registered=True)
    assert _reconcile_worker(gate) is True
    assert gate["agent"].starts == 0
    assert calls == []