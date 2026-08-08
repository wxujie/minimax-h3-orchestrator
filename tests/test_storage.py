"""Storage safety + lifecycle tests."""
from __future__ import annotations

from controller.storage import Storage, safe_join, sanitize_filename


def test_sanitize_strips_paths():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("a/b/c.png") == "c.png"
    assert sanitize_filename(".hidden") == "hidden"
    assert sanitize_filename("") == "file.bin"


def test_safe_join_never_escapes(tmp_path):
    # sanitize_filename strips traversal before safe_join, so a hostile name is
    # always contained inside the directory.
    target = safe_join(tmp_path, "../../escape.png")
    assert str(target).startswith(str(tmp_path.resolve()))


def test_write_and_read_output(storage, tmp_path):
    job_id = "job_x"
    p = storage.write_output(job_id, "out.mp4", b"data")
    assert p.exists() and p.read_bytes() == b"data"
    got = storage.look_for_video(job_id)
    assert got and got.read_bytes() == b"data"


def test_sweep_removes_old_job(storage):
    import os
    import time
    job_id = "job_old"
    storage.write_output(job_id, "out.mp4", b"x")
    assert storage.look_for_video(job_id)
    # backdate the artifact dir mtime so it falls outside retention
    os.utime(storage.artifacts_dir / job_id, (time.time() - 7200, time.time() - 7200))
    removed = storage.sweep(hours=1)
    assert removed == 1
    assert storage.look_for_video(job_id) is None