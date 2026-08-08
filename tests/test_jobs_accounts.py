"""Job registry state machine + account listing safety."""
from __future__ import annotations

from controller.constants import ErrorClass, JobStatus
from controller.jobs import JobManager


# -------------------------------------------------------------------- jobs ----
def test_create_is_queued_and_positioned(managers):
    jobs = managers["jobs"]
    a = jobs.create(input_data={"prompt": "a"})
    b = jobs.create(input_data={"prompt": "b"}, priority=5)
    assert a["status"] == "QUEUED"
    # higher priority appears first in queue
    assert jobs.next_in_queue()["job_id"] == b["job_id"]


def test_transient_failure_retries_then_fails(managers):
    jobs = managers["jobs"]
    jid = jobs.create(input_data={})["job_id"]
    assert jobs.retryable(jid) is True
    max_retries = 2
    # attempts 1..2 use up retries but stay retryable
    for _ in range(max_retries):
        jobs.start_attempt(jid)
        jobs.fail(jid, "boom", error_class=ErrorClass.TRANSIENT)
        assert jobs.retryable(jid) is True
    # attempt 3 exceeds max_retries -> permanent fail
    jobs.start_attempt(jid)
    jobs.fail(jid, "boom", error_class=ErrorClass.TRANSIENT)
    assert jobs.retryable(jid) is False
    # the scheduler turns a past-due transient failure into FAILED
    jobs.set_status(jid, JobStatus.FAILED)
    assert jobs.get(jid)["status"] == JobStatus.FAILED.value


def test_permanent_failure_does_not_retry(managers):
    jobs = managers["jobs"]
    jid = jobs.create(input_data={})["job_id"]
    jobs.fail(jid, "bad workflow", error_class=ErrorClass.PERMANENT)
    assert jobs.get(jid)["status"] == JobStatus.FAILED.value
    assert jobs.retryable(jid) is False


def test_count_and_list(managers):
    jobs = managers["jobs"]
    jobs.create(input_data={"a": 1})
    jobs.create(input_data={"b": 2})
    c = jobs.counts()
    assert c[JobStatus.QUEUED.value] == 2
    assert len(jobs.list()) == 2


# ----------------------------------------------------------------- accounts ---
def test_sync_from_config_creates_accounts(managers):
    accs = managers["accounts"]
    n = accs.sync_from_config()
    assert n == 3
    listed = accs.list_accounts()
    assert len(listed) == 3
    # credentials never leak into the public listing
    for row in listed:
        assert "key" not in row
        assert "username" in row  # username is not a secret (login id)


def test_disabled_account_excluded_from_provisioning(store, managers, provider):
    from controller.constants import AccountStatus
    accs = managers["accounts"]
    accs.set_status("kaggle-account-2", AccountStatus.ACCOUNT_DISABLED)
    # only enabled accounts are candidates
    from controller.scheduler import Scheduler
    from controller.constants import JobStatus
    sched = Scheduler(store, managers["jobs"], managers["workers"], accs,
                      provider=provider)
    sched.workers.poll_health = lambda w: None
    managers["jobs"].create(input_data={"prompt": "x"})
    sched.tick()
    assert provider.started[0][0] == "kaggle-account-1"