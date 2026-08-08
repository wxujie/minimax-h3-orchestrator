"""Local artifact storage.

Layout (storage_dir):
    storage/
        uploads/            controller-side inbound client uploads
        artifacts/<job_id>/  per-job input/output, unique names
        retention sweep removes artifacts older than JOB_OUTPUT_RETENTION_HOURS

All filenames are sanitized (basename-only, stripped of path separators) and
writes always stay inside the job directory: path traversal is impossible.
Results and uploads reference stored files, never client-supplied paths.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from .config import settings
from .db import Store


def sanitize_filename(name: str) -> str:
    """Force a filename to a safe basename. No slashes, no '..', no control chars."""
    name = os.path.basename(name)
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = name.strip().lstrip(".")
    if not name:
        name = "file.bin"
    return name[:120]


def safe_join(directory: Path, filename: str) -> Path:
    """Join a directory with a sanitized filename, asserting containment."""
    safe = sanitize_filename(filename)
    target = (directory / safe).resolve()
    root = directory.resolve()
    if not str(target).startswith(str(root)):
        raise ValueError(f"refusing path outside storage: {filename}")
    return target


class Storage:
    def __init__(self) -> None:
        base = Path(settings.storage_dir if settings else "./storage").resolve()
        self.uploads_dir = base / "uploads"
        self.artifacts_dir = base / "artifacts"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- client uploads ----------------
    def save_upload(self, filename: str, data: bytes) -> Path:
        safe = safe_join(self.uploads_dir, filename)
        safe.write_bytes(data)
        return safe

    def open_upload(self, filename: str) -> Path:
        return safe_join(self.uploads_dir, filename)

    def upload_path(self, filename: str) -> Path:
        return safe_join(self.uploads_dir, os.path.basename(filename))

    def exists(self, path: Path) -> bool:
        return Path(path).exists()

    # ---------------- job artifacts ----------------
    def job_dir(self, job_id: str, sub: str = "") -> Path:
        d = self.artifacts_dir / job_id
        if sub:
            d = d / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def store_input(self, job_id: str, filename: str, data: bytes) -> Path:
        target = safe_join(self.job_dir(job_id, "input"), filename)
        target.write_bytes(data)
        return target

    def read_artifact(self, job_id: str, filename: str) -> Optional[Path]:
        target = safe_join(self.artifacts_dir / job_id, filename)
        return target if target.exists() else None

    def look_for_video(self, job_id: str) -> Optional[Path]:
        """Discover the generated video for a job (first non-empty .mp4)."""
        outdir = self.job_dir(job_id, "output")
        if not outdir.exists():
            return None
        for f in sorted(outdir.iterdir()):
            if f.is_file() and f.suffix.lower() in {".mp4", ".mkv", ".webm"} and f.stat().st_size > 0:
                return f
        return None

    def retain_output(self, job_id: str, filepath: Path) -> Path:
        """Copy a downloaded video into the job's persistent output dir."""
        target = safe_join(self.job_dir(job_id, "output"), filepath.name)
        shutil.copyfile(filepath, target)
        return target

    def write_output(self, job_id: str, filename: str, data: bytes) -> Path:
        """Persist raw bytes (e.g. the downloaded mp4) into the job output dir."""
        target = safe_join(self.job_dir(job_id, "output"), filename)
        target.write_bytes(data)
        return target

    # ---------------- retention ----------------
    def sweep(self, hours: Optional[float] = None) -> int:
        """Delete completed jobs older than ``hours``. Returns count removed."""
        hours = hours if hours is not None else (
            settings.job_output_retention_hours if settings else 24
        )
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - float(hours) * 3600
        removed = 0
        for entry in self.artifacts_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except (FileNotFoundError, OSError):
                continue
            if mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed