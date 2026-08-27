"""Build the ipynb that actually gets pushed to Kaggle.

The reference notebook (``notebooks/minimax-h3-comfyui.ipynb``) bootstraps a
full ComfyUI + model download but stops there — it never registers a worker with
the controller, so a queued job can never be handed to it. This builder takes
that template and appends two cells:

  1. install the worker agent's HTTP deps (fastapi/uvicorn/httpx/...), and
  2. clone the orchestrator repo, inject this notebook's identity
     (``NOTEBOOK_ID``, ``CONTROLLER_PUBLIC_URL``, ``WORKER_AUTH_SECRET``) into the
     environment, and run ``worker.runner.run()`` — which serves the agent,
     opens a tunnel, and POSTs ``/api/v1/agents/register`` back to the controller.

The built notebook is written to a temp dir at push time and is **never** part of
the git repo, so the injected ``WORKER_AUTH_SECRET`` / public URL stay out of
version control. The Kaggle account key is never embedded here — only the
orchestrator's own worker secret and controller URL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .constants import GPU_PER_NOTEBOOK
from .logging_conf import get_logger

log = get_logger("notebook_builder")

DEFAULT_REPO_URL = "https://github.com/msaadakram/minimax-h3-orchestrator"


def _pipe_install_cell() -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (
            "# --- MiniMax H3 orchestrator: worker agent HTTP deps ---\n"
            "!pip install -q fastapi uvicorn httpx pydantic python-multipart\n"
        ),
    }


def _runner_cell(*, notebook_id: str, controller_public_url: str,
                 worker_auth_secret: str, gpu_count: int,
                 repo_url: str) -> dict:
    nb = json.dumps(notebook_id)
    url = json.dumps(controller_public_url)
    sec = json.dumps(worker_auth_secret)
    gpu = json.dumps(str(gpu_count))
    repo = json.dumps(repo_url)

    source = (
        "# --- MiniMax H3 orchestrator: register GPU workers with the controller ---\n"
        "import os, sys, subprocess, pathlib, shutil\n"
        "\n"
        f"REPO_URL = {repo}\n"
        'REPO = "/tmp/minimax-h3-orchestrator"\n'
        "\n"
        "# Always re-clone: Kaggle does not reliably clear /tmp between kernel\n"
        "# restarts, and a stale checkout would run old worker code (e.g. the\n"
        "# zombie-tunnel keepalive fix). A fresh clone guarantees the notebook\n"
        "# runs exactly the code pushed to the repo.\n"
        "if pathlib.Path(REPO).exists():\n"
        "    shutil.rmtree(REPO, ignore_errors=True)\n"
        "subprocess.run(\n"
        '    ["git", "clone", "--depth", "1", REPO_URL, REPO],\n'
        "    check=True, capture_output=True,\n"
        ")\n"
        "sys.path.insert(0, REPO)\n"
        "\n"
        f"os.environ.setdefault(\"NOTEBOOK_ID\", {nb})\n"
        f"os.environ.setdefault(\"CONTROLLER_PUBLIC_URL\", {url})\n"
        f"os.environ.setdefault(\"WORKER_AUTH_SECRET\", {sec})\n"
        f"os.environ.setdefault(\"GPU_COUNT\", {gpu})\n"
        "\n"
        "from worker.runner import run\n"
        "run()  # serves agents, opens tunnels, registers; keeps the session alive\n"
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def build_notebook(
    *,
    notebook_id: str,
    controller_public_url: str,
    worker_auth_secret: str,
    gpu_count: Optional[int] = None,
    repo_url: Optional[str] = None,
    template_path: Optional[Path] = None,
    template: Optional[dict] = None,
) -> dict:
    """Return the customized ipynb document for one notebook push.

    ``template`` (a parsed ipynb dict) or ``template_path`` supplies the ComfyUI
    bootstrap. Identity values are injected into the trailing runner cell.
    """
    if template is None:
        tpath = template_path or Path("./notebooks/minimax-h3-comfyui.ipynb")
        with open(tpath, encoding="utf-8") as f:
            template = json.load(f)

    # Clone the document so the committed template file is never mutated.
    nb = json.loads(json.dumps(template))
    cells = list(nb.get("cells", []))

    # Only append the worker wiring once (idempotent for re-pushes).
    has_runner = any(
        "worker.runner" in "".join(c.get("source", []))
        for c in cells
    )
    if not has_runner:
        cells.append(_pipe_install_cell())
        cells.append(_runner_cell(
            notebook_id=notebook_id,
            controller_public_url=controller_public_url,
            worker_auth_secret=worker_auth_secret,
            gpu_count=gpu_count if gpu_count is not None else GPU_PER_NOTEBOOK,
            repo_url=repo_url or DEFAULT_REPO_URL,
        ))
    nb["cells"] = cells

    # The metadata's accelerator governs the Kaggle runtime; align it with the
    # GPU notebook this is built for (kernel-metadata.json also sets enable_gpu).
    meta = nb.setdefault("kaggle", {})
    meta["accelerator"] = "GPU"
    meta["isGpuEnabled"] = True
    meta["dockerImageVersionId"] = 28755  # cur/CPU-compatible CUDA image

    nb["metadata"] = nb.get("metadata", {})
    nb["metadata"]["kaggle"] = meta
    return nb


def write_notebook(path: Path, doc: dict) -> Path:
    """Write a built notebook document to disk (used by the push temp dir)."""
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path