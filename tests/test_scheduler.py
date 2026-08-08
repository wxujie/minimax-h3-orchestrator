"""Scheduler tests: dispatch, provisioning, completion, retry — all in-process.

Real JobManager/WorkerManager/AccountManager over a throwaway SQLite DB, a
FakeProvider for notebook provisioning, and FakeWorkerClient for job lifecycle.
``WorkerManager.poll_health`` is monkeypatched so no network is touched. The
scheduler is driven deterministically by calling ``Scheduler.tick()`` directly
(the background thread is never started).
"""
from __future__ import annotations

from controller import db
from controller.constants import JobStatus
from controller.scheduler import Scheduler
from tests.fakes import FakeWorkerScript, make_client_factory


def _add_worker(store, wid: str) -> None:
    with store.session() as s:
        s.merge(db.Worker(
            id=wid, notebook_id="nb-1", gpu_index=0, comfy_port=8188,
            comfy_url="http://127.0.0.1:8188", tunnel_url="http://w.local",
            status="WORKER_READY", token_id="t1",
        ))
        s.flush()


def _make(managers, provider, storage, *, scripts=None) -> Scheduler:
    sched = Scheduler(
        managers["store"], managers["jobs"], managers["workers"],
        managers["accounts"], provider=provider, storage=storage,
        client_factory=make_client_factory(scripts),
    )
    # Health is scripted: any tunnelled worker looks green, no HTTP.
    sched.workers.poll_health = (
        lambda w: {"status": "ready"} if w.get("tunnel_url") else None)
    return sched


def _queue(managers, **input_data) -> str:
    input_data.setdefault("first_frame", "frame.png")
    return managers["jobs"].create(input_data=input_data)["job_id"]


def test_dispatch_to_ready_worker_completes(managers, provider, storage):
    store = managers["store"]
    _add_worker(store, "worker-0")
    sched = _make(managers, provider, storage)

    # ensure the referenced first_frame exists on disk for staging
    f = storage.upload_path("frame.png")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x89PNG\r\n\x1a\nframe")

    jid = _queue(managers, prompt="a cat", duration=2.0, first_frame="frame.png")
    sched.tick()  # heartbeat + dispatch -> RUNNING
    assert managers["jobs"].get(jid)["status"] == JobStatus.RUNNING.value

    sched.tick()  # monitor -> DONE -> COMPLETED
    assert managers["jobs"].get(jid)["status"] == JobStatus.COMPLETED.value
    assert managers["jobs"].get(jid)["result_artifact_id"]
    # worker returned to ready
    assert managers["workers"].get("worker-0")["status"] == "WORKER_READY"
    # output persisted under the job dir
    out = storage.artifacts_dir / jid / "output"
    assert out.exists() and any(out.iterdir())


def test_no_ready_worker_provisions_notebook(managers, provider, storage):
    sched = _make(managers, provider, storage)
    jid = _queue(managers, prompt="x")  # no workers at all
    sched.tick()
    assert provider.started, "provider.start_notebook not called"
    acct_id, nb = provider.started[0]
    assert acct_id == "kaggle-account-1"
    # notebook row + GPU_PER_NOTEBOOK placeholders created
    with managers["store"].session() as s:
        wcount = s.query(db.Worker).count()
        nb_status = s.query(db.Notebook).filter(db.Notebook.id == nb).first().status
    assert wcount == 2
    assert nb_status == "NOTEBOOK_STARTING"
    # job can't run until the notebooks actually register ready workers
    assert managers["jobs"].get(jid)["status"] == JobStatus.QUEUED.value


def test_provider_failure_exhausts_account_quota(managers, provider, storage):
    provider.fail_next = True
    sched = _make(managers, provider, storage)
    _queue(managers, prompt="x")
    sched.tick()
    with managers["store"].session() as s:
        nb = s.query(db.Notebook).first()
        assert nb is not None
        assert nb.status == "QUOTA_EXHAUSTED"
    assert managers["accounts"].running_notebooks() == 0


def test_worker_failure_requeues_transient(managers, provider, storage):
    store = managers["store"]
    _add_worker(store, "worker-0")
    sched = _make(managers, provider, storage,
                  scripts={"worker-0": FakeWorkerScript(job_status="FAILED",
                                                        error="vae oom")})
    jid = _queue(managers, prompt="hi")
    sched.tick()  # dispatch
    sched.tick()  # worker reports FAILED -> requeue (attempt 1 of max 2)
    st = managers["jobs"].get(jid)
    assert st["status"] == JobStatus.QUEUED.value
    # start_attempt incremented attempt_count past 0 on dispatch
    with managers["store"].session() as conn:
        assert conn.query(db.Job).get(jid).attempt_count >= 1


def test_persistent_failure_marks_job_failed(managers, provider, storage):
    store = managers["store"]
    _add_worker(store, "worker-0")
    sched = _make(managers, provider, storage,
                  scripts={"worker-0": FakeWorkerScript(job_status="FAILED",
                                                        error="bad request")})
    jid = _queue(managers, prompt="hi")
    # drive enough ticks to exceed MAX_JOB_RETRIES (=2)
    for _ in range(6):
        sched.tick()
    assert managers["jobs"].get(jid)["status"] == JobStatus.FAILED.value


def test_cancel(managers, provider, storage):
    jobs = managers["jobs"]
    sid = jobs.create(input_data={"prompt": "x"})
    assert jobs.get(sid["job_id"])["status"] == JobStatus.QUEUED.value
    assert jobs.cancel(sid["job_id"]) is True
    assert jobs.get(sid["job_id"])["status"] == JobStatus.CANCELLED.value
    assert jobs.cancel(sid["job_id"]) is False