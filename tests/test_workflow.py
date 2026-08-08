"""Workflow adapter tests against the real workflow.json."""
from __future__ import annotations

import pytest

from controller.workflow import (
    WorkflowAdapter, WorkflowError, duration_to_frames,
)


def test_duration_to_frames_grid_snap():
    # 2s @24fps = 48 frames -> snap to 17k+5 grid: 48 -> 56
    assert duration_to_frames(2.0) == 56


def test_validate_passes_on_real_file(workflow):
    assert workflow.validate() is None
    desc = workflow.describe()
    assert desc["status"] == "ok"


def test_build_prompt_shape(workflow):
    g = workflow.build_prompt(first_frame="a.png", last_frame="b.png",
                              prompt_text="hello", duration=2.0,
                              width=672, height=384, seed=7)
    # loaders present
    assert g["6"]["class_type"] == "UNETLoader"
    assert g["13"]["class_type"] == "CLIPLoader"
    # core node
    miv = g["104"]
    assert miv["class_type"] == "MiniMaxH3ImageToVideo"
    assert miv["inputs"]["prompt"] == "hello"
    assert miv["inputs"]["width"] == 672
    assert miv["inputs"]["height"] == 384
    # frame ordering: first and last both wired
    assert miv["inputs"]["first_frame"][0] == "_load_first"
    assert miv["inputs"]["last_frame"][0] == "_load_last"


def test_build_prompt_requires_first_frame(workflow):
    with pytest.raises(WorkflowError):
        workflow.build_prompt(prompt_text="missing frame")


def test_resolution_clamped_and_bounds(workflow):
    g = workflow.build_prompt(first_frame="a.png", width=100, height=100)
    assert g["104"]["inputs"]["width"] == 96   # 100 -> //32*32
    with pytest.raises(WorkflowError):
        workflow.build_prompt(first_frame="a.png", width=4000, height=4000)


def test_duration_out_of_range(workflow):
    with pytest.raises(WorkflowError):
        workflow.build_prompt(first_frame="a.png", duration=99.0)


def test_model_overrides(workflow):
    g = workflow.build_prompt(
        first_frame="a.png",
        model_overrides={"unet_name": "custom.safetensors"},
    )
    assert g["6"]["inputs"]["unet_name"] == "custom.safetensors"


def test_frame_missing_uses_first_for_last(workflow):
    g = workflow.build_prompt(first_frame="only.png")
    assert g["104"]["inputs"]["last_frame"] == g["104"]["inputs"]["first_frame"]