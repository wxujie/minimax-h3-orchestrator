"""Job REST endpoints.

POST /api/v1/jobs                create a job (JSON or multipart)
GET  /api/v1/jobs                list jobs
GET  /api/v1/jobs/{id}           job status
GET  /api/v1/jobs/{id}/result    download the generated video
POST /api/v1/jobs/{id}/cancel    cancel queued/running work

Files are staged into the controller's upload store, then handed to a worker by
filename after it is copied into the worker's own input dir. A JSON body takes
filename references; a multipart body accepts the files directly.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import main as pmain
from ..config import settings
from ..constants import JobStatus
from ..logging_conf import get_logger
from ..storage import Storage, sanitize_filename
from ..jobs import JobManager

log = get_logger("api_jobs")
router = APIRouter()


def _manager() -> tuple[JobManager, Storage]:
    store = pmain.get_store()
    jobs = JobManager(store)
    # Storage is a shared singleton on the app; re-build is cheap/harmless here.
    storage = getattr(pmain, "_storage_singleton", None)
    if storage is None:
        storage = Storage()
        pmain._storage_singleton = storage
    return jobs, storage


class JobCreate(BaseModel):
    prompt: str = ""
    duration: float = 2.0
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None
    model_overrides: Optional[dict] = None
    priority: int = 0


@router.post("/jobs")
def create_job_json(req: JobCreate):
    jobs, _ = _manager()
    r = jobs.create(input_data={
        "prompt": req.prompt,
        "duration": req.duration,
        "width": req.width, "height": req.height,
        "seed": req.seed,
        "first_frame": req.first_frame, "last_frame": req.last_frame,
        "model_overrides": req.model_overrides or {},
    }, priority=req.priority, max_retries=settings.max_job_retries)
    log.info("job_created job_id=%s", r["job_id"])
    return r


@router.post("/jobs/multipart")
def create_job_multipart(
    prompt: str = Form(""),
    duration: float = Form(2.0),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    priority: int = Form(0),
    first_frame: UploadFile = File(...),
    last_frame: Optional[UploadFile] = File(None),
):
    jobs, storage = _manager()
    fname = sanitize_filename(first_frame.filename or "first.png")
    storage.save_upload(fname, first_frame.file.read())
    lf = None
    if last_frame and last_frame.filename:
        lname = sanitize_filename(last_frame.filename or "last.png")
        storage.save_upload(lname, last_frame.file.read())
        lf = lname
    r = jobs.create(input_data={
        "prompt": prompt, "duration": duration,
        "width": width, "height": height,
        "first_frame": fname, "last_frame": lf,
    }, priority=priority, max_retries=settings.max_job_retries)
    log.info("job_created job_id=%s multipart", r["job_id"])
    return r


@router.get("/jobs")
def list_jobs(status: Optional[str] = None, limit: int = 100):
    jobs, _ = _manager()
    return jobs.list(status=status, limit=limit)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    jobs, _ = _manager()
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    st = j["status"]
    progress = (100 if st == JobStatus.COMPLETED.value
                else 0 if st == JobStatus.QUEUED.value else 45)
    return {
        "job_id": j["job_id"],
        "status": st,
        "worker_id": j["worker_id"],
        "progress": progress,
        "error": j["last_error"],
        "download_url": f"/api/v1/jobs/{job_id}/result"
        if st == JobStatus.COMPLETED.value else None,
        "created_at": j["created_at"],
    }


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    jobs, storage = _manager()
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    if j["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail="job not finished")
    path = storage.look_for_video(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="result missing on disk")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    jobs, _ = _manager()
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=400, detail="job cannot be cancelled")
    log.info("job_cancelled job_id=%s", job_id)
    return {"job_id": job_id, "status": JobStatus.CANCELLED.value}