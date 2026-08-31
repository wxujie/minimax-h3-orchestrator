#!/usr/bin/env python3
"""探测 Kaggle 账号能申请到哪些 GPU 加速器。

对每个目标 accelerator 构造一个最小 notebook + kernel-metadata.json，
用 `kaggle kernels push --accelerator <ACC>` 试提交。能成功 push 的
就是有权限的卡；报 quota/entitlement 错误的就是拿不到的。

用法:
    python scripts/gpu_probe.py [accelerator ...]

默认探测列表: NvidiaL4 NvidiaTeslaT4Highmem NvidiaA100 NvidiaH100 NvidiaRtxPro6000
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 从 scripts/ 运行时也要能找到 controller 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.config import settings

TARGETS = sys.argv[1:] or [
    "NvidiaL4",
    "NvidiaTeslaT4Highmem",
    "NvidiaA100",
    "NvidiaH100",
    "NvidiaRtxPro6000",
]

NOTEBOOK = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["import subprocess\n",
                   "print(subprocess.run(['nvidia-smi'], capture_output=True, "
                   "text=True).stdout[:300])\n"],
    }],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _account():
    for a in settings.accounts:
        if a.provider == "kaggle" and a.enabled:
            return a
    raise SystemExit("没有启用的 Kaggle 账号")


def _env(acct) -> dict:
    env = dict(__import__("os").environ)
    env["KAGGLE_USERNAME"] = acct.username
    env["KAGGLE_KEY"] = acct.key
    env["KAGGLE_API_TOKEN"] = acct.key
    return env


def probe(acct, accelerator: str, slug: str) -> tuple[bool, str]:
    """试 push 一次，返回 (是否成功, 摘要)。"""
    import os
    env = _env(acct)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        meta = {
            "id": slug,
            "title": slug.split("/", 1)[-1],  # title 必须等于 id 的 basename，否则 409
            "code_file": "probe.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "competition_sources": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        (td / "probe.ipynb").write_text(json.dumps(NOTEBOOK))
        r = subprocess.run(
            ["kaggle", "kernels", "push", "--path", str(td),
             "--accelerator", accelerator],
            capture_output=True, text=True, timeout=180, env=env,
        )
        out = (r.stdout + r.stderr).strip()
        low = out.lower()
        # 真正的权限拒绝：quota / entitlement / 403 / 权限类错误
        denied_kw = ("quota", "entitle", "forbidden", "403", "not available",
                     "you do not", "no permission", "not entitled", "unavailable")
        denied = any(k in low for k in denied_kw)
        # 成功：exit 0 且没有权限类拒绝词
        ok = r.returncode == 0 and not denied
        return ok, out[:600]


def main():
    acct = _account()
    print(f"账号: {acct.username}")
    print("=" * 60)
    for acc in TARGETS:
        slug = f"{acct.username}/gpu-probe-{acc.lower()}"
        print(f"\n>>> {acc}")
        try:
            ok, msg = probe(acct, acc, slug)
        except subprocess.TimeoutExpired:
            print("  ⏱ 超时（可能卡在排队）")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ 异常: {e}")
            continue
        status = "✅ 可申请" if ok else "❌ 拒绝"
        print(f"  {status}")
        # 只打印关键行
        for line in msg.splitlines():
            low = line.lower()
            if any(k in low for k in ("error", "quota", "entitle", "forbidden",
                                      "403", "accelerator", "not", "invalid",
                                      "available", "creating", "exists")):
                print(f"    {line.strip()}")
        time.sleep(3)  # 避免打太快触发限流


if __name__ == "__main__":
    main()