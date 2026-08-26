"""Tests for the real notebook push: builder + ensure_notebook wiring.

Covers the step that turns a queued job into a registerable worker: the
built ipynb carries the ComfyUI bootstrap plus the runner cell (clone repo,
inject identity, run), and `KaggleManager.ensure_notebook` writes that ipynb
(as the kernel code file) into the push temp dir instead of a stub.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from controller.config import AccountConfig, settings
from controller.kaggle_manager import KaggleManager
from controller.notebook_builder import DEFAULT_REPO_URL, build_notebook


def _runner_cell(doc: dict) -> dict:
    for c in doc["cells"]:
        if "worker.runner" in "".join(c.get("source", [])):
            return c
    raise AssertionError("runner cell not found in built notebook")


_TEMPLATE = {"cells": [], "metadata": {"kaggle": {"accelerator": "none"}}}


def test_build_notebook_injects_identity():
    doc = build_notebook(
        notebook_id="nb-kaggle-account-1",
        controller_public_url="https://ctrl.example.com",
        worker_auth_secret="secret-123",
        gpu_count=2,
        template=_TEMPLATE,
    )
    src = _runner_cell(doc)["source"]
    assert "nb-kaggle-account-1" in src
    assert "https://ctrl.example.com" in src
    assert "secret-123" in src
    assert '"2"' in src
    assert '"git", "clone"' in src
    assert "from worker.runner import run" in src
    assert 'os.environ.setdefault("NOTEBOOK_ID"' in src
    assert DEFAULT_REPO_URL in src


def test_build_notebook_is_idempotent():
    kw = dict(notebook_id="nb-1", controller_public_url="https://c.example.com",
              worker_auth_secret="s", gpu_count=1, template=_TEMPLATE)
    doc1 = build_notebook(**kw)
    doc2 = build_notebook(**kw)
    assert len(doc1["cells"]) == 2  # install + runner
    assert len(doc2["cells"]) == 2  # no double-append on rebuild


def test_build_notebook_flips_accelerator_and_keeps_template():
    doc = build_notebook(notebook_id="nb-1", controller_public_url="https://c",
                         worker_auth_secret="s", template=_TEMPLATE)
    assert doc["metadata"]["kaggle"]["accelerator"] == "GPU"
    assert doc["metadata"]["kaggle"]["isGpuEnabled"] is True
    # the caller-supplied template object must not be mutated
    assert _TEMPLATE["metadata"]["kaggle"]["accelerator"] == "none"


def test_real_template_gets_runner_cell():
    """The on-disk reference notebook, when built, ends up with the runner."""
    from pathlib import Path as _P
    doc = build_notebook(
        notebook_id="nb-1", controller_public_url="https://c.example.com",
        worker_auth_secret="s", template_path=_P("notebooks/minimax-h3-comfyui.ipynb"),
    )
    # reference notebook: 24 base cells + Multishot pack cell = 25;
    # +pip-install +runner = 27
    assert len(doc["cells"]) == 27
    _runner_cell(doc)


def test_build_pushed_notebook_needs_public_url(monkeypatch):
    """Without CONTROLLER_PUBLIC_URL the notebook cannot register -> refuse build."""
    from controller import main as mainmod
    monkeypatch.setattr(settings, "controller_public_url", "")
    assert mainmod.build_pushed_notebook("nb-1") is None


def test_build_pushed_notebook_uses_settings(monkeypatch):
    from controller import main as mainmod
    monkeypatch.setattr(settings, "controller_public_url", "https://ctrl.example.com")
    monkeypatch.setattr(settings, "worker_auth_secret", "w-secret")
    doc = mainmod.build_pushed_notebook("nb-kaggle-account-1")
    assert doc is not None
    src = _runner_cell(doc)["source"]
    assert "nb-kaggle-account-1" in src
    assert "https://ctrl.example.com" in src
    assert "w-secret" in src


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _capturing_run(self, args, **kw):
    """Fake _run that records the push temp dir's contents for assertions."""
    # args == ['kaggle', 'kernels', 'push', '--path', <tmpdir>]
    td = Path(args[-1])
    captured["dir"] = td
    captured["files"] = sorted(p.name for p in td.iterdir())
    meta = json.loads((td / "kernel-metadata.json").read_text())
    captured["meta"] = meta
    code = td / meta["code_file"]
    captured["code_json"] = json.loads(code.read_text()) if code.exists() else None
    return _FakeProc()


captured: dict = {}


def test_ensure_notebook_pushes_real_ipynb(monkeypatch):
    """ensure_notebook writes the built ipynb (not a stub) as the code file."""
    monkeypatch.setattr(KaggleManager, "_run", _capturing_run)
    mgr = KaggleManager(AccountConfig(id="k1", username="acct-user-1", key="k-key"))
    body = build_notebook(
        notebook_id="nb-1", controller_public_url="https://c.example.com",
        worker_auth_secret="s", template=_TEMPLATE,
    )
    assert mgr.ensure_notebook("acct-user-1/nb-1", body) is True

    assert captured["code_json"] is not None, "expected a real ipynb code file"
    assert captured["meta"]["id"] == "acct-user-1/nb-1"
    assert captured["meta"]["kernel_type"] == "notebook"
    assert captured["meta"]["code_file"] == "minimax-h3-comfyui.ipynb"
    assert captured["meta"]["enable_gpu"] is True
    # the pushed document must be the real builder output, not the old stub
    assert any("worker.runner" in "".join(c.get("source", []))
               for c in captured["code_json"]["cells"])
    assert not any(p.endswith(".py") for p in captured["files"])


def test_kaggle_provider_start_builds_real_notebook(monkeypatch):
    """Full provisioning path: a job -> provider -> real registerable notebook pushed."""
    from controller import main as mainmod
    from controller.config import AccountConfig as _AC

    monkeypatch.setattr(mainmod.settings, "controller_public_url", "https://ctrl.example.com")
    monkeypatch.setattr(mainmod.settings, "worker_auth_secret", "w-secret")

    class _Accts:
        def credential(self, aid):
            return _AC(id=aid, username="acct-user-1", key="k-key")

    monkeypatch.setattr(mainmod, "_accounts", _Accts())
    captured: dict = {}

    def fake_run(self, args, **kw):
        if "push" not in args:
            return _FakeProc()  # capacity() -> `config view` rc 0
        td = Path(args[-1])
        captured["meta"] = json.loads((td / "kernel-metadata.json").read_text())
        captured["doc"] = json.loads((td / captured["meta"]["code_file"]).read_text())
        return _FakeProc()

    monkeypatch.setattr(KaggleManager, "_run", fake_run)
    prov = mainmod.KaggleProvider()
    assert prov.start_notebook("kaggle-account-1", "nb-kaggle-account-1") is True
    assert captured["meta"]["id"] == "acct-user-1/nb-kaggle-account-1"
    assert captured["meta"]["kernel_type"] == "notebook"
    assert captured["meta"]["code_file"] == "minimax-h3-comfyui.ipynb"
    assert captured["meta"]["enable_gpu"] is True
    src = "".join("".join(c.get("source", [])) for c in captured["doc"]["cells"])
    assert "nb-kaggle-account-1" in src
    assert "https://ctrl.example.com" in src


def test_kaggle_provider_refuses_without_public_url(monkeypatch):
    """Without CONTROLLER_PUBLIC_URL the scheduler must not push an unroutable notebook."""
    from controller import main as mainmod

    def _run_ok(self, args, **kw):
        return _FakeProc()  # capacity gives green; build itself is what refuses

    monkeypatch.setattr(mainmod.settings, "controller_public_url", "")
    monkeypatch.setattr(mainmod.settings, "worker_auth_secret", "w-secret")
    monkeypatch.setattr(KaggleManager, "_run", _run_ok)

    class _Accts:
        def credential(self, aid):
            return AccountConfig(id=aid, username="acct-user-1", key="k-key")

    monkeypatch.setattr(mainmod, "_accounts", _Accts())
    prov = mainmod.KaggleProvider()
    assert prov.start_notebook("kaggle-account-1", "nb-kaggle-account-1") is False


def test_ensure_notebook_accepts_first_push_success(monkeypatch):
    """A brand-new kernel push succeeds with rc 0 even on the first run."""
    def _run_ok(self, args, **kw):
        return _FakeProc(returncode=0, stdout="Creating new kernel...")
    monkeypatch.setattr(KaggleManager, "_run", _run_ok)
    mgr = KaggleManager(AccountConfig(id="k1", username="u", key="k"))
    assert mgr.ensure_notebook("u/nb-1", {"cells": []}) is True