"""Notebook-side entrypoint: turn one Kaggle GPU slot into a registered worker.

Run *after* a ComfyUI instance is already listening on ``127.0.0.1:8188 + gpu``
(the reference notebook's bootstrap does this). For each GPU index this:

  1. builds a ``WorkerSpec`` and a ``WorkerAgent`` (agent_FastAPI on
     ``127.0.0.1:8000 + gpu``, talking only to its local ComfyUI),
  2. serves the agent with uvicorn,
  3. opens a Cloudflare quick tunnel in front of the **agent** (never ComfyUI),
  4. registers the worker with the controller via ``POST /api/v1/agents/register``
     so the scheduler can dispatch video jobs to its tunnel URL.

After all workers are registered the loop keeps running (watching tunnels +
re-registering) so the notebook session stays alive, acting as the replacement
for the reference notebook's trailing ``sleep`` cell.

Identity comes from env, injected into the built notebook by
``controller/notebook_builder.py``:

    NOTEBOOK_ID             the controller's notebook_name ("nb-...").
    CONTROLLER_PUBLIC_URL   controller root, publicly reachable from Kaggle.
    WORKER_AUTH_SECRET      shared controller<->worker secret.
    GPU_COUNT               how many GPU workers to bring up (default 2).

``workflow_factory`` is bound to the real ``controller.workflow.WorkflowAdapter``
(part of the orchestrator repo this package is imported from) so the worker and
the controller share one definition of the workflow.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import httpx
except Exception:  # notebook bootstrap installs it before runner is imported
    httpx = None  # type: ignore

# FastAPI/uvicorn are imported lazily so `--selftest` / unit tests can import
# this module without them.
from . import cloudflare as cf
from .agent import WorkerAgent, WorkerSpec

log = logging.getLogger("worker.runner")


# ------------------------------------------------------------------ factory --
def build_workflow_factory(repo_root: Optional[Path] = None):
    """Bind the real MiniMax-H3 workflow adapter as the agent's factory.

    ``repo_root`` lets callers point at a cloned orchestrator checkout (the
    notebook layout) instead of relying on cwd. When omitted we try the repo
    layout produced by a runtime clone.
    """
    from controller.workflow import WorkflowAdapter  # lazy import
    from controller.workflow_r2v import R2VWorkflowAdapter
    from controller.workflow_multishot import MultishotWorkflowAdapter

    def _find(rel: str) -> Path:
        if repo_root is not None:
            return repo_root / "workflows" / rel
        return Path(f"workflows/{rel}")

    fl2va_path = _find("workflow.json")
    r2v_path = _find("workflow_r2v.json")
    multishot_path = _find("H3_Seamless_Chain_CORE.json")
    if not fl2va_path.exists():
        raise RuntimeError(f"workflow.json not found at {fl2va_path}")

    fl2va = WorkflowAdapter(path=fl2va_path)
    r2v = R2VWorkflowAdapter(path=r2v_path) if r2v_path.exists() else None
    multishot = MultishotWorkflowAdapter(path=multishot_path) if multishot_path.exists() else None

    def workflow_factory(**kwargs: object) -> dict:
        # Dispatch on the presence of R2V-only parameters.
        if "ref_images" in kwargs or kwargs.get("mode") == "r2v":
            if r2v is None:
                raise RuntimeError("R2V workflow adapter not available")
            kwargs.pop("mode", None)
            return r2v.build_prompt(**kwargs)
        if "script" in kwargs:
            if multishot is None:
                raise RuntimeError("Multishot workflow adapter not available")
            kwargs.pop("mode", None)
            kwargs.pop("ref_images", None)
            return multishot.build_prompt(**kwargs)
        kwargs.pop("mode", None)
        kwargs.pop("ref_images", None)
        return fl2va.build_prompt(**kwargs)

    return workflow_factory


# ------------------------------------------------------------------- server --
def _serve(agent: WorkerAgent, host: str, port: int) -> None:
    """Run an agent's FastAPI app with uvicorn in the current thread."""
    import uvicorn  # lazy

    server = uvicorn.Server(uvicorn.Config(
        agent.app, host=host, port=port, log_level="warning"))
    server.run()


def _wait_server(port: int, timeout_s: float = 30) -> bool:
    """Poll until the agent FastAPI answers on its local port."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# --------------------------------------------------------------- register ---
def _register(controller_url: str, secret: str, *, worker_id: str,
              notebook_id: str, gpu: int, tunnel_url: str) -> bool:
    url = f"{controller_url.rstrip('/')}/api/v1/agents/register"
    payload = {
        "worker_id": worker_id,
        "notebook_id": notebook_id,
        "gpu": gpu,
        "tunnel_url": tunnel_url,
    }
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=30)
        return r.status_code < 400
    except Exception as exc:  # noqa: BLE001 - transient network
        log.warning("register_failed worker=%s err=%s", worker_id, exc)
        return False


def _reconcile_worker(gate: dict) -> bool:
    """Ensure one worker is tunnelled and registered with the controller.

    Returns True when the worker is *good*: live tunnel **and** registered.

    Heals two failure modes:

    * a dead cloudflared process — restart it (``start_tunnel`` truncates the
      log, so only the *new* URL is read; the stale/dead URL is not resurrected)
      and clear the registration so the fresh URL is posted;
    * a healthy tunnel whose registration never landed (the controller was
      briefly down) — re-register the existing URL on the next pass.
    """
    agent = gate["agent"]
    proc = agent.tunnel.proc
    proc_dead = proc is not None and proc.poll() is not None
    need_tunnel = proc_dead or not agent.tunnel.is_alive()
    if need_tunnel:
        url = agent.start_tunnel(timeout=40)
        gate["url"] = url or gate.get("url")
        gate["registered"] = False
    else:
        gate["url"] = agent.tunnel.public_url or gate.get("url")
    if not gate.get("url"):
        return False
    if not gate.get("registered"):
        gate["registered"] = _register(
            gate["controller_url"], gate["secret"],
            worker_id=gate["worker_id"], notebook_id=gate["notebook_id"],
            gpu=gate["gpu"], tunnel_url=gate["url"])
        log.info("worker_reconcile worker=%s url=%s registered=%s",
                 gate["worker_id"], gate["url"], gate["registered"])
    return gate["registered"]


def _keepalive_loop(gates: list[dict], interval_s: float = 30) -> None:
    """Watch every worker's tunnel + registration; heal and re-register forever.

    Blocks indefinitely (keeps the notebook session alive). On a recycled/dead
    quick tunnel or a missed registration it re-establishes the worker so the
    controller's next health check finds it READY again.
    """
    while True:
        time.sleep(interval_s)
        for gate in gates:
            _reconcile_worker(gate)


# ------------------------------------------------------------------- launch --
def run(*, notebook_id: Optional[str] = None, controller_url: Optional[str] = None,
        secret: Optional[str] = None, gpu_count: Optional[int] = None,
        comfy_port_base: int = 8188, agent_port_base: int = 8000,
        repo_root: Optional[Path] = None, input_dir: Optional[str] = None,
        output_dir: Optional[str] = None, keepalive: bool = True) -> list[dict]:
    """Bring up ``gpu_count`` workers and register them with the controller.

    Returns the list of registered workers (dicts) after first registration.
    With ``keepalive=True`` (default) the function then blocks forever and
    keeps the session alive, so call it last in a notebook cell.
    """
    notebook_id = notebook_id or os.environ.get("NOTEBOOK_ID", "")
    controller_url = controller_url or os.environ.get("CONTROLLER_PUBLIC_URL", "")
    secret = secret or os.environ.get("WORKER_AUTH_SECRET", "")
    gpu_count = gpu_count if gpu_count is not None else int(
        os.environ.get("GPU_COUNT", "2"))

    if not notebook_id or not controller_url:
        raise RuntimeError("NOTEBOOK_ID and CONTROLLER_PUBLIC_URL are required")

    input_dir = input_dir or os.environ.get("WORKER_INPUT_DIR", "/kaggle/working/inputs")
    output_dir = output_dir or os.environ.get("WORKER_OUTPUT_DIR", "/kaggle/working/outputs")
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from controller.constants import worker_id  # stable notebook-name scheme

    # Resolve the orchestrator checkout from the importable `controller` package
    # (cloned by the notebook runner cell) so workflow.json is found no matter
    # where the repo landed (cwd on Kaggle is /kaggle/working, not the clone dir).
    if repo_root is None:
        import controller as _ctrl  # noqa: PLC0415
        repo_root = Path(_ctrl.__file__).resolve().parent.parent

    workflow_factory = build_workflow_factory(repo_root)

    agents: list[WorkerAgent] = []
    for gpu in range(gpu_count):
        spec = WorkerSpec(
            worker_name=worker_id(notebook_id, gpu),
            gpu_index=gpu,
            comfy_port=comfy_port_base + gpu,
            agent_port=agent_port_base + gpu,
            secret=secret,
            tunnel_log_path=f"/tmp/worker_gpu{gpu}.log",
            input_dir=input_dir,
            output_dir=output_dir,
            job_timeout_s=3600.0,
        )
        agent = WorkerAgent(spec, workflow_factory)
        threading.Thread(
            target=_serve, args=(agent, "127.0.0.1", spec.agent_port),
            daemon=True, name=f"uvicorn-{spec.worker_name}").start()
        agents.append(agent)
        log.info("agent_starting worker=%s port=%s", spec.worker_name, spec.agent_port)

    # Wait for each agent to answer locally, expose it through a tunnel, and
    # register it. Reconciliation (restart dead tunnels, re-register on a
    # missed registration) is driven by the keepalive loop below; the first
    # per-worker pass here just brings the initial state up.
    gates: list[dict] = []
    for gpu, agent in enumerate(agents):
        if not _wait_server(agent.spec.agent_port):
            log.error("agent_not_listening worker=%s", agent.spec.worker_name)
            continue
        gates.append({
            "controller_url": controller_url, "secret": secret,
            "notebook_id": notebook_id, "gpu": gpu,
            "worker_id": agent.spec.worker_name, "agent": agent,
            "url": None, "registered": False,
        })
        _reconcile_worker(gates[-1])

    registered = [{
        "worker_id": g["worker_id"], "notebook_id": g["notebook_id"],
        "gpu": g["gpu"], "tunnel_url": g.get("url"), "registered": g["registered"],
    } for g in gates]

    if keepalive:
        _keepalive_loop(gates)
    return registered


# ------------------------------------------------------------------- cli -----
def main(argv: Optional[list[str]] = None) -> int:
    """Self-test / standalone launcher (``python -m worker.runner``).

    For an actual notebook the ``run()`` entrypoint is called from a cell; this
    CLI exists so the runner can be exercised locally (without a controller) for
    packaging sanity. ``--selftest`` only checks imports/config, no servers.
    """
    ap = argparse.ArgumentParser(description="MiniMax-H3 worker runner")
    ap.add_argument("--selftest", action="store_true", help="import/wiring check only")
    ap.add_argument("--notebook-id", default=os.environ.get("NOTEBOOK_ID", ""))
    ap.add_argument("--controller-url", default=os.environ.get("CONTROLLER_PUBLIC_URL", ""))
    ap.add_argument("--secret", default=os.environ.get("WORKER_AUTH_SECRET", ""))
    ap.add_argument("--gpu-count", type=int,
                    default=int(os.environ.get("GPU_COUNT", "1")))
    args = ap.parse_args(argv)

    if args.selftest:
        print("selftest: imports ok")
        print("controller_url configured:", bool(args.controller_url))
        from controller.workflow import WorkflowAdapter
        from controller.config import settings
        status = WorkflowAdapter(
            path=Path(settings.workflow_path) if settings else None
        ).describe()["status"]
        print("workflow adapter status:", status)
        return 0

    run(notebook_id=args.notebook_id, controller_url=args.controller_url,
        secret=args.secret, gpu_count=args.gpu_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())