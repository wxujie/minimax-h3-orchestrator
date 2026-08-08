"""GPU detection and isolation helpers.

The notebook starts one ComfyUI per physical GPU with ``CUDA_VISIBLE_DEVICES``
pin-pointing exactly one device. ``gpu_info`` reads nvidia-smi to build a
human/API-visible map of which physical GPU each worker owns.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
        }


def list_gpus() -> list[GpuInfo]:
    """Query nvidia-smi for physical GPUs (in the Kaggle runtime)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    gpus: list[GpuInfo] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GpuInfo(
                index=int(parts[0]), name=parts[1],
                memory_total_mb=int(parts[2]), memory_used_mb=int(parts[3]),
            ))
        except ValueError:
            continue
    return gpus


def env_for_gpu(gpu_index: int) -> dict:
    """Return an env copy with CUDA_VISIBLE_DEVICES pinned to one GPU."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env