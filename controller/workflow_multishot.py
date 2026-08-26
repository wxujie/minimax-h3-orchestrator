"""Workflow adapter for MiniMax-H3 Multishot (seamless chained shots).

Builds the flat ComfyUI ``/prompt`` API graph for the
``H3_Seamless_Chain_CORE`` workflow from joeygambino/MiniMax-H3-Multishot.
The graph chains N shots into one continuous take: the ``H3MultishotSampler``
node loops internally, feeding each shot's last frame + audio into the next
shot (``continuity=first_frame``, the model's own trained hand-off — no extra
third-party Motion-Context pack required).

This adapter is a sibling of ``controller.workflow`` (FL2VA) and
``controller.workflow_r2v`` (R2V). Node ids below match the CORE workflow JSON
exactly.

Required custom pack: ComfyUI-H3-Multishot (installed into ComfyUI's
``custom_nodes/`` by the Kaggle notebook).

Models (already downloaded by the existing notebook):
- unet      minimax_h3_ref2va_pruned_int8_convrot.safetensors  (diffusion_models)
- clip      qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      (text_encoders)
- video vae minimax_h3_video_vae_fp16.safetensors             (vae)
- audio vae minimax_h3_audio_vae_fp32.safetensors             (vae)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import settings


class WorkflowError(Exception):
    """Validation / conversion failure -> Permanent job error."""


# Node ids from H3_Seamless_Chain_CORE.json
N_CONTROLS = "2"      # H3StudioControls
N_MODEL = "3"         # H3ModelLoaderAny
N_CLIP = "4"          # H3ClipLoaderAny
N_LORA = "5"          # H3LoraStack
N_VAE_V = "6"         # VAELoader (video)
N_VAE_A = "7"         # VAELoader (audio)
N_SAMPLER = "8"       # H3MultishotSampler
N_CREATE = "10"       # CreateVideo
N_SAVE_V = "11"       # SaveVideo
N_SAVE_A = "12"       # SaveAudio

# Models
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

_LORA_NONE = "None"


class MultishotWorkflowAdapter:
    """Builds a submit-ready flat API graph for the Multishot CORE workflow."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or self._default_path()
        self.raw: dict = {}
        self._load()

    @staticmethod
    def _default_path() -> Path:
        if settings is not None and getattr(settings, "workflow_multishot_path", None):
            return Path(settings.workflow_multishot_path)
        return Path("./workflows/H3_Seamless_Chain_CORE.json")

    def _load(self) -> None:
        if not self.path.exists():
            raise WorkflowError(f"Multishot workflow file not found: {self.path}")
        try:
            self.raw = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Multishot workflow JSON malformed: {e}") from e

    # ------------------------------------------------------------------ build --
    def build_prompt(
        self,
        *,
        script: str,
        width: int = 768,
        height: int = 768,
        frames_per_shot: int = 243,
        steps: int = 20,
        seed: Optional[int] = None,
        shot_count: int = 0,
        model_name: str = UNET_REF2VA,
        clip_name: str = CLIP_NAME,
        video_vae: str = VAE_VIDEO,
        audio_vae: str = VAE_AUDIO,
        sampler_name: str = "res_multistep",
        scheduler: str = "simple",
        seed_per_shot: bool = True,
        lora_1: str = _LORA_NONE,
        lora_strength_1: float = 1.0,
        lora_2: str = _LORA_NONE,
        lora_strength_2: float = 1.0,
        lora_3: str = _LORA_NONE,
        lora_strength_3: float = 1.0,
        lora_4: str = _LORA_NONE,
        lora_strength_4: float = 1.0,
    ) -> dict:
        """Return a flat ComfyUI ``/prompt`` graph.

        ``script`` must be the Multishot script: one prompt per shot separated
        by ``---`` on its own line, or a JSON ``{"prompts": [...]}`` string.
        """
        if not script or not script.strip():
            raise WorkflowError("script is required for Multishot")
        w = max(32, (int(round(width)) // 32) * 32)
        h = max(32, (int(round(height)) // 32) * 32)
        if w > 4096 or h > 4096:
            raise WorkflowError(f"resolution too large: {w}x{h}")
        seed_val = seed if seed is not None else 0
        fps = max(5, (int(round(frames_per_shot)) // 17) * 17)
        fps = min(fps, 481)

        graph = {
            N_CONTROLS: {
                "class_type": "H3StudioControls",
                "inputs": {
                    "width": w,
                    "height": h,
                    "frames_per_shot": fps,
                    "steps": steps,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                },
            },
            N_MODEL: {
                "class_type": "H3ModelLoaderAny",
                "inputs": {"model_name": model_name},
            },
            N_CLIP: {
                "class_type": "H3ClipLoaderAny",
                "inputs": {"clip_name": clip_name, "type": "minimax"},
            },
            N_LORA: {
                "class_type": "H3LoraStack",
                "inputs": {
                    "model": [N_MODEL, 0],
                    "lora_1": lora_1, "strength_1": lora_strength_1,
                    "lora_2": lora_2, "strength_2": lora_strength_2,
                    "lora_3": lora_3, "strength_3": lora_strength_3,
                    "lora_4": lora_4, "strength_4": lora_strength_4,
                },
            },
            N_VAE_V: {
                "class_type": "VAELoader",
                "inputs": {"vae_name": video_vae},
            },
            N_VAE_A: {
                "class_type": "VAELoader",
                "inputs": {"vae_name": audio_vae},
            },
            N_SAMPLER: {
                "class_type": "H3MultishotSampler",
                "inputs": {
                    "model": [N_LORA, 0],
                    "clip": [N_CLIP, 0],
                    "video_vae": [N_VAE_V, 0],
                    "audio_vae": [N_VAE_A, 0],
                    "script": script,
                    "shot_count": shot_count,
                    "width": [N_CONTROLS, 0],
                    "height": [N_CONTROLS, 1],
                    "frames_per_shot": [N_CONTROLS, 2],
                    "steps": [N_CONTROLS, 3],
                    "seed": seed_val,
                    "seed_per_shot": seed_per_shot,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                },
            },
            N_CREATE: {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": [N_SAMPLER, 0],
                    "audio": [N_SAMPLER, 1],
                    "fps": 24,
                    "bit_depth": 8,
                },
            },
            N_SAVE_V: {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": [N_CREATE, 0],
                    "filename_prefix": "video/H3CHAIN",
                    "format": "auto",
                    "codec": "auto",
                },
            },
            N_SAVE_A: {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": [N_SAMPLER, 1],
                    "filename_prefix": "audio/H3CHAIN",
                },
            },
        }
        return graph