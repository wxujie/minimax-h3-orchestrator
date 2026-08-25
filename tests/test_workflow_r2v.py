"""Tests for the R2V (reference-to-video + turbo) workflow adapter."""
from __future__ import annotations

import pytest

from controller.workflow_r2v import (
    R2VWorkflowAdapter, WorkflowError, duration_to_frames,
    TURBO_LORA_NAME, UNET_REF2VA, N_CORE, N_LORA, N_SW_MODEL, N_SW_STEPS,
)


@pytest.fixture(scope="module")
def adapter():
    return R2VWorkflowAdapter()


def test_duration_grid_snap():
    assert duration_to_frames(5.0) == 124


def test_validate(adapter):
    assert adapter.validate() is None


def test_build_prompt_plain(adapter):
    g = adapter.build_prompt(prompt_text="anime city", duration=5.0,
                             width=672, height=384, seed=1)
    core = g[N_CORE]
    assert core["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert core["inputs"]["prompt"] == "anime city"
    assert core["inputs"]["width"] == 672
    assert core["inputs"]["height"] == 384
    # plain mode: no lora/switch nodes, 20 steps
    assert N_LORA not in g
    assert N_SW_MODEL not in g
    assert g["124"]["inputs"]["steps"] == 20
    # ref images left unset when none provided
    assert core["inputs"]["ref_images.ref_image_0"] is None


def test_build_prompt_turbo(adapter):
    g = adapter.build_prompt(prompt_text="anime", duration=5.0, turbo=True)
    # lora + switches present
    assert g[N_LORA]["class_type"] == "LoraLoaderModelOnly"
    assert g[N_LORA]["inputs"]["lora_name"] == TURBO_LORA_NAME
    assert g[N_SW_MODEL]["class_type"] == "ComfySwitchNode"
    assert g[N_SW_STEPS]["class_type"] == "ComfySwitchNode"
    # guider + scheduler routed through switch model
    assert g["126"]["inputs"]["model"][0] == N_SW_MODEL
    assert g["124"]["inputs"]["model"][0] == N_SW_MODEL
    # turbo steps = 4 (switch true branch)
    assert g[N_SW_STEPS]["inputs"]["on_true"] == 4
    assert g[N_SW_STEPS]["inputs"]["on_false"] == 20


def test_build_prompt_ref_images(adapter):
    g = adapter.build_prompt(prompt_text="x", duration=5.0,
                             ref_images=["a.png", "b.png"])
    core = g[N_CORE]["inputs"]
    assert core["ref_images.ref_image_0"] == ["_ref0", 0]
    assert core["ref_images.ref_image_1"] == ["_ref1", 0]
    assert core["ref_images.ref_image_2"] is None
    assert g["_ref0"]["class_type"] == "LoadImage"
    assert g["_ref0"]["inputs"]["image"] == "a.png"


def test_unet_is_ref2va(adapter):
    g = adapter.build_prompt(prompt_text="x")
    assert g["127"]["inputs"]["unet_name"] == UNET_REF2VA


def test_resolution_clamp(adapter):
    g = adapter.build_prompt(prompt_text="x", width=100, height=100)
    assert g[N_CORE]["inputs"]["width"] == 96
    with pytest.raises(WorkflowError):
        adapter.build_prompt(prompt_text="x", width=4000, height=4000)


def test_duration_out_of_range(adapter):
    with pytest.raises(WorkflowError):
        adapter.build_prompt(prompt_text="x", duration=99.0)