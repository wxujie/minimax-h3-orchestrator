"""Worker agent: the authenticated control plane for one GPU's ComfyUI.

Runs *inside the Kaggle notebook*, one process per GPU worker. The controller
reaches this agent through the worker's Cloudflare tunnel. The agent talks only
to its own local ComfyUI instance — ComfyUI itself is never exposed.

Routes
------
    GET  /health                 liveness + state (unauthenticated *)
    GET  /status                 detailed worker/GPU/ComfyUI/tunnel status
    POST /jobs                   create + run a video job
    GET  /jobs/{id}              job status / progress
    GET  /jobs/{id}/result       download the finished video
    POST /jobs/{id}/cancel       abort a queued/running job
    POST /shutdown               stop the tunnel for this worker

    (*) /health intentionally needs no token so the controller's readiness
        probe works while secrets are still being wired; it leaks only whether
        ComfyUI and the tunnel are up, never secrets.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import cloudflare as cf
from . import comfy_client
from . import gpu as gpumod

log = logging.getLogger("worker.agent")

JOB_QUEUED, JOB_RUNNING, JOB_DONE, JOB_FAILED, JOB_CANCELLED = (
    "QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"
)


@dataclass
class WorkerSpec:
    """A worker's immutable identity + per-worker paths (from notebook env)."""

    worker_name: str
    gpu_index: int
    comfy_port: int
    agent_port: int
    secret: str
    tunnel_log_path: str
    input_dir: str
    output_dir: str
    job_timeout_s: float = 7200.0

    @property
    def comfy_base_url(self) -> str:
        return f"http://127.0.0.1:{self.comfy_port}"


class JobRequest(BaseModel):
    prompt_text: str
    duration: float = 2.0
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None
    ref_images: Optional[list[str]] = None
    ref_image_size: str = "match"
    turbo: bool = False
    turbo_steps: Optional[int] = None
    turbo_lora_strength: Optional[float] = None
    use_pdd: bool = False
    pdd_nfe: str = "8"
    pdd_lora_strength: float = 1.0
    pdd_head_strength: float = 1.0
    script: Optional[str] = None
    frames_per_shot: Optional[int] = None
    shot_count: int = 0
    start_image: Optional[str] = None
    reference_images: Optional[list[str]] = None
    workflow: str = "minimax-h3"
    graph: Optional[dict] = None
    model_overrides: Optional[dict] = None
    # 单任务最大渲染秒数；None = 用 worker 的 spec.job_timeout_s（全局默认）。
    timeout_s: Optional[float] = None


@dataclass
class RunnableJob:
    id: str
    payload: JobRequest
    status: str = JOB_QUEUED
    prompt_id: Optional[str] = None
    created: float = field(default_factory=time.time)
    progress: int = 0
    error: str = ""
    result_path: Optional[str] = None


class WorkerAgent:
    """One GPU's agent. Bound to exactly one local ComfyUI instance."""

    def __init__(self, spec: WorkerSpec,
                 workflow_factory: Callable[..., dict]) -> None:
        self.spec = spec
        self.comfy = comfy_client.ComfyClient(base_url=spec.comfy_base_url)
        self.workflow_factory = workflow_factory
        self.secret = spec.secret or os.environ.get("WORKER_AUTH_SECRET", "")
        self.tunnel = self._make_tunnel(spec)
        self.lock = threading.Lock()
        self.jobs: dict[str, RunnableJob] = {}
        self.app = self._build_app()

    @staticmethod
    def _make_tunnel(spec: WorkerSpec):
        """Select tunnel backend by TUNNEL_MODE env (quick=default, named)."""
        mode = os.environ.get("TUNNEL_MODE", "quick").strip().lower()
        bin_path = os.environ.get("CLOUDFLARED_BIN", "/tmp/cloudflared")
        if mode == "named":
            config_content = os.environ.get("CLOUDFLARE_TUNNEL_CONFIG", "")
            creds_content = os.environ.get("CLOUDFLARE_TUNNEL_CREDENTIALS", "")
            domain = os.environ.get("TUNNEL_DOMAIN", "")
            if not config_content or not creds_content or not domain:
                log.warning(
                    "TUNNEL_MODE=named but CLOUDFLARE_TUNNEL_CONFIG / "
                    "CLOUDFLARE_TUNNEL_CREDENTIALS / TUNNEL_DOMAIN unset; "
                    "falling back to quick tunnel")
                return cf.QuickTunnel(spec.agent_port, spec.tunnel_log_path,
                                      bin_path=bin_path)
            url = cf.named_tunnel_url(spec.worker_name, domain)
            return cf.NamedTunnel(
                config_content=config_content,
                credentials_content=creds_content,
                log_path=spec.tunnel_log_path,
                public_url=url,
                bin_path=bin_path,
            )
        return cf.QuickTunnel(spec.agent_port, spec.tunnel_log_path,
                              bin_path=bin_path)

    # ------------------------------------------------------------- security --
    def _auth(self, authorization: Optional[str]) -> bool:
        if not self.secret:
            return True
        if not authorization or not authorization.lower().startswith("bearer "):
            return False
        token = authorization.split(" ", 1)[1].strip()
        return hmac.compare_digest(token, self.secret)

    def _require(self, authorization: Optional[str]) -> None:
        if not self._auth(authorization):
            raise HTTPException(status_code=401, detail="invalid worker token")

    # ----------------------------------------------------------------- is_alive
    def _comfy_ok(self) -> bool:
        try:
            self.comfy.health()
            return True
        except Exception:
            return False

    # --------------------------------------------------------------- tunnels --
    def start_tunnel(self, timeout: int = 30) -> Optional[str]:
        self.tunnel.start()
        return self.tunnel.wait_for_url(timeout_s=timeout)

    def tunnel_ok(self) -> bool:
        return self.tunnel.is_alive()

    # ------------------------------------------------------------- routes ----
    def _build_app(self) -> FastAPI:
        app = FastAPI(title=f"{self.spec.worker_name}", version="1.0")

        @app.get("/health")
        def health():
            ok = self._comfy_ok()
            return {
                "worker_id": self.spec.worker_name,
                "gpu": self.spec.gpu_index,
                "status": "ready" if (ok and self.tunnel_ok()) else "starting",
                "comfyui": ok,
                "cloudflare": self.tunnel_ok(),
                "public_url": self.tunnel.public_url,
            }

        @app.get("/status")
        def worker_status(authorization: Optional[str] = Header(default=None)):
            self._require(authorization)
            return {
                "worker_id": self.spec.worker_name,
                "gpu": self._gpu(),
                "comfyui_ok": self._comfy_ok(),
                "tunnel_ok": self.tunnel_ok(),
                "tunnel_url": self.tunnel.public_url,
                "busy": bool(self.jobs),
                "running": [j.id for j in self.jobs.values()
                            if j.status == JOB_RUNNING],
            }

        @app.get("/debug")
        def worker_debug(authorization: Optional[str] = Header(default=None)):
            """诊断端点：线程栈 + ComfyUI 队列 + 渲染日志 tail。

            卡死时可用它定位：采样线程卡在哪个调用、ComfyUI 是否还在队列中
            执行、以及 ComfyUI 最近的报错/进度日志。
            """
            self._require(authorization)
            import sys as _sys
            import traceback as _traceback
            frames = {}
            for t in threading.enumerate():
                f = _sys._current_frames().get(t.ident)
                if f is None:
                    frames[t.name] = []
                    continue
                frames[t.name] = [
                    f"{fr.filename}:{fr.lineno} in {fr.name}"
                    for fr in _traceback.extract_stack(f)
                ][-8:]
            # ComfyUI queue (queue_running / queue_pending)
            queue_info = None
            try:
                queue_info = self.comfy.queue()
            except Exception as e:  # noqa: BLE001
                queue_info = {"error": str(e)}
            # ComfyUI 日志 tail（单卡：gpu_index=0 -> /tmp/comfyui_gpu0.log）
            log_tail = []
            log_path = f"/tmp/comfyui_gpu{self.spec.gpu_index}.log"
            try:
                with open(log_path, "r", errors="replace") as f:
                    lines = f.readlines()[-30:]
                log_tail = [ln.rstrip() for ln in lines]
            except Exception as e:  # noqa: BLE001
                log_tail = [f"<无法读取 {log_path}: {e}>"]
            return {
                "worker_id": self.spec.worker_name,
                "threads": frames,
                "comfy_queue": queue_info,
                "comfy_log_tail": log_tail,
                "jobs": {jid: {"status": j.status, "progress": j.progress,
                                 "prompt_id": j.prompt_id}
                         for jid, j in self.jobs.items()},
            }

        @app.post("/jobs")
        def run_job(job: JobRequest,
                    authorization: Optional[str] = Header(default=None)):
            self._require(authorization)
            with self.lock:
                busy = any(j.status == JOB_RUNNING for j in self.jobs.values())
                if busy:
                    raise HTTPException(status_code=409, detail="worker busy")
                job_id = f"{self.spec.worker_name}-{uuid.uuid4().hex[:8]}"
                self.jobs[job_id] = RunnableJob(id=job_id, payload=job)
            threading.Thread(target=self._execute, args=(job_id,), daemon=True).start()
            return {"job_id": job_id, "status": "STARTING"}

        @app.post("/jobs/input")
        async def upload_input(request: Request,
                               authorization: Optional[str] = Header(default=None)):
            """Stage a client file (image) into the worker's input dir.

            Body: multipart with one field named ``file``. Filename is
            sanitized and written under ``input_dir`` only (no traversal).
            Returns the safe name to reference in JobRequest.
            """
            self._require(authorization)
            form = await request.form()
            up = form.get("file")
            if up is None:
                raise HTTPException(status_code=400, detail="missing file field")
            import os.path as _p
            base = _p.basename(up.filename or "file.png")[:120]
            os.makedirs(self.spec.input_dir, exist_ok=True)
            safe = _p.join(self.spec.input_dir, base)
            with open(safe, "wb") as f:
                f.write(up.file.read())
            return {"filename": base}

        @app.get("/jobs/{job_id}")
        def job_status(job_id: str, authorization: Optional[str] = Header(default=None)):
            self._require(authorization)
            rj = self.jobs.get(job_id)
            if not rj:
                raise HTTPException(status_code=404, detail="unknown job")
            return {"job_id": job_id, "status": rj.status,
                    "progress": rj.progress, "error": rj.error}

        @app.get("/jobs/{job_id}/result")
        def job_result(job_id: str, authorization: Optional[str] = Header(default=None)):
            self._require(authorization)
            rj = self.jobs.get(job_id)
            if not rj or rj.status != JOB_DONE or not rj.result_path:
                raise HTTPException(status_code=404, detail="result not ready")
            if not os.path.exists(rj.result_path):
                raise HTTPException(status_code=404, detail="result file missing")
            return StreamingResponse(
                open(rj.result_path, "rb"),
                media_type="video/mp4",
                headers={"Content-Disposition":
                         f'attachment; filename="{os.path.basename(rj.result_path)}"'},
            )

        @app.post("/jobs/{job_id}/cancel")
        def job_cancel(job_id: str, authorization: Optional[str] = Header(default=None)):
            self._require(authorization)
            rj = self.jobs.get(job_id)
            if not rj:
                raise HTTPException(status_code=404, detail="unknown job")
            if rj.status == JOB_RUNNING and rj.prompt_id:
                self.comfy.cancel(rj.prompt_id)
            rj.status = JOB_CANCELLED
            return {"job_id": job_id, "status": "CANCELLED"}

        @app.post("/shutdown")
        def shutdown(authorization: Optional[str] = Header(default=None)):
            self._require(authorization)
            self.tunnel.stop()
            return {"ok": True}

        return app

    # --------------------------------------------------------------- helpers --
    def _gpu(self) -> dict:
        gpus = gpumod.list_gpus()
        if gpus and self.spec.gpu_index < len(gpus):
            return gpus[self.spec.gpu_index].to_dict()
        return {"index": self.spec.gpu_index, "name": "unknown"}

    def _stage_path(self, raw: Optional[str]) -> Optional[str]:
        """Map a client filename to a staged input file in the work dir."""
        if not raw:
            return None
        import os
        base = os.path.basename(raw)  # sanitize: no traversal
        candidate = os.path.join(self.spec.input_dir, base)
        return candidate if os.path.exists(candidate) else None

    def _execute(self, job_id: str) -> None:
        rj = self.jobs[job_id]
        rj.status = JOB_RUNNING
        try:
            p = rj.payload
            if p.workflow == "minimax-h3-multishot":
                if p.graph is not None:
                    graph = p.graph
                else:
                    # fallback: upload ref images then build via factory
                    start_comfy = None
                    ref_comfy = []
                    if p.start_image:
                        sl = self._stage_path(p.start_image)
                        if sl:
                            start_comfy = self.comfy.upload_image(sl)
                    for raw in (p.reference_images or []):
                        local = self._stage_path(raw)
                        if local:
                            ref_comfy.append(self.comfy.upload_image(local))
                    graph = self.workflow_factory(
                        script=p.script or "",
                        width=p.width, height=p.height,
                        frames_per_shot=p.frames_per_shot or 243,
                        steps=4, seed=p.seed,
                        shot_count=p.shot_count,
                        start_image=start_comfy,
                        reference_images=ref_comfy or None,
                    )
            elif p.workflow == "minimax-h3-r2v":
                if p.graph is not None:
                    # Controller already built the full ComfyUI graph.
                    graph = p.graph
                else:
                    # Reference-to-Video: stage + upload ref images, call R2V factory.
                    ref_comfy = []
                    for raw in (p.ref_images or []):
                        local = self._stage_path(raw)
                        if local:
                            ref_comfy.append(self.comfy.upload_image(local))
                    graph = self.workflow_factory(
                        prompt_text=p.prompt_text, duration=p.duration,
                        width=p.width, height=p.height, seed=p.seed,
                        ref_images=ref_comfy,
                        ref_image_size=p.ref_image_size or "match",
                        turbo=p.turbo,
                        turbo_steps=p.turbo_steps,
                        turbo_lora_strength=p.turbo_lora_strength,
                        use_pdd=p.use_pdd,
                        pdd_nfe=p.pdd_nfe,
                        pdd_lora_strength=p.pdd_lora_strength,
                        pdd_head_strength=p.pdd_head_strength,
                        model_overrides=p.model_overrides,
                    )
            else:
                first_local = self._stage_path(p.first_frame)
                last_local = self._stage_path(p.last_frame) if p.last_frame else None
                if not first_local:
                    raise comfy_client.ComfyError(
                        "first_frame not present in staged input (was it uploaded?)",
                        transient=False)
                # Upload staged files into ComfyUI; get back its filename.
                first_comfy = self.comfy.upload_image(first_local)
                last_comfy = (self.comfy.upload_image(last_local)
                              if last_local else first_comfy)
                graph = self.workflow_factory(
                    prompt_text=p.prompt_text, duration=p.duration,
                    width=p.width, height=p.height, seed=p.seed,
                    first_frame=first_comfy, last_frame=last_comfy,
                    model_overrides=p.model_overrides,
                )
            rj.prompt_id = self.comfy.submit_prompt(graph)
        except comfy_client.ComfyError as e:
            rj.status = JOB_FAILED
            rj.error = e.message
            return
        except Exception as e:  # noqa: BLE001
            rj.status = JOB_FAILED
            rj.error = str(e)
            return

        try:
            # job 级超时优先，否则回落 worker 全局超时。
            effective_timeout = (
                p.timeout_s if p.timeout_s is not None
                else self.spec.job_timeout_s
            )
            # 渲染期间用 elapsed/timeout 估算真实推进（0→90%），完成时跳 100。
            # 多镜头自定义 sampler 不向 ComfyUI /prompt 上报 step 级进度，
            # 时间估算是唯一能反映「还在跑、跑到哪」的可靠信号。
            def _on_progress(_status_info: dict) -> None:
                elapsed = time.time() - rj.created
                est = min(90, int(elapsed / max(1.0, effective_timeout) * 90))
                if est > rj.progress:
                    rj.progress = est
            history = self.comfy.wait_for_history(
                rj.prompt_id, timeout_s=effective_timeout,
                on_progress=_on_progress)
            rj.progress = 100
            out_dir = os.path.join(self.spec.output_dir, job_id)
            path = self.comfy.download_output(history, out_dir)
            if path is None:
                rj.status, rj.error = JOB_FAILED, "no video output produced"
                return
            rj.result_path = path
            rj.status = JOB_DONE
        except comfy_client.ComfyError as e:
            rj.status = JOB_FAILED
            rj.error = e.message


class InputAgent:  # keep a stable public name
    pass


# Compatibility alias used by some notebook snippets.
make_agent = WorkerAgent