"""Workflow adapter for MiniMax H3 Reference-to-Video (R2V) with Turbo LoRA.

This is a sibling of ``controller.workflow`` (which drives the FL2VA
image-to-video graph). It builds the flat ComfyUI ``/prompt`` API graph for the
R2V mode by reading node ids/types/model names from the official R2V template
file (``workflows/workflow_r2v.json``), then emitting a faithful API payload.

Graph shape (flat API format):

    UNETLoader(ref2va unet) ──┬─> LoraLoaderModelOnly ──┐
                              └────────────────────────┼─> ComfySwitch ─> BasicGuider.model
    CLIPLoader(qwen3vl)  ──────────────> MiniMaxH3ReferenceToVideo ── positive ─┐
    VAELoader(video)     ──────────────> MiniMaxH3ReferenceToVideo ── LATENT ──┐│
    VAELoader(audio)     ──────────────> MiniMaxH3ReferenceToVideo (audio_vae) ││
    LoadImage(ref0..N)   ──────────────> MiniMaxH3ReferenceToVideo (ref_images) ││
    RandomNoise ─> SamplerCustomAdvanced.noise                                   ││
    BasicScheduler ─> SamplerCustomAdvanced.sigmas                               ││
    KSamplerSelect ─> SamplerCustomAdvanced.sampler                              ││
    SamplerCustomAdvanced.latent_image <── MiniMaxH3ReferenceToVideo.LATENT ─────││
    SamplerCustomAdvanced.output ─> VAEDecode / VAEDecodeAudio ─> CreateVideo ─> SaveVideo

Turbo mode: when enabled, the unet is routed through ``LoraLoaderModelOnly``
with the ref2v turbo LoRA (``minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16``)
and ``turbo_steps`` replaces the normal 20 steps. Disabled = plain unet + 20
steps. The switch nodes mirror the template's ComfySwitchNode wiring exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import settings


class WorkflowError(Exception):
    """Validation / conversion failure -> Permanent job error."""


def _snap_to_grid(frames: int) -> int:
    base = max(5, round(frames))
    return base + (5 - (base % 17)) % 17


def duration_to_frames(seconds: float) -> int:
    return _snap_to_grid(seconds * 24.0)


# ------------------------------------------------------------------ node ids --
# These ids match the official R2V template. They are read from the template
# file at runtime, so a template update only requires the JSON to stay valid;
# these constants are fallback identifiers for the builder.
N_UNET = "127"
N_CLIP = "128"
N_VAE_V = "119"
N_VAE_A = "120"
N_NOISE = "129"
N_CORE = "136"          # MiniMaxH3ReferenceToVideo
N_GUIDER = "126"
N_SCHED = "124"
N_SEL = "123"
N_SAMPLER = "125"
N_VDEC = "122"
N_ADEC = "121"
N_CREATE = "130"
N_SAVE = "92"
N_LORA = "145"          # LoraLoaderModelOnly
N_SW_MODEL = "141"      # ComfySwitchNode (unet vs lora-unet)
N_SW_STEPS = "142"      # ComfySwitchNode (20 vs turbo steps)
N_PDD = "150"           # UC_MiniMaxH3PDDAcc (阿里 PDD Acc LoRA)

# Turbo LoRA constants (official ref2v turbo)
TURBO_LORA_NAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
TURBO_LORA_STRENGTH = 1.0
NORMAL_STEPS = 20
TURBO_STEPS = 4

# 阿里 PDD Acc LoRA (Parallel Decoding Distillation, 8-step)
PDD_LORA_NAME = "minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors"
PDD_NFE = "8"
PDD_LORA_STRENGTH = 1.0
PDD_HEAD_STRENGTH = 1.0

# T4 (15GB VRAM) safe default resolution. The ref2va unet (~20GB int8) + CLIP
# + dual VAE + audio decoder exceed T4 VRAM at the native 1344x768 canvas,
# causing OOM / "no video output produced". 832x480 is the largest size that
# renders reliably on T4. Bigger GPUs (T4x2/A100) can request full res.
T4_SAFE_RESOLUTION = (832, 480)

# model files (official Comfy-Org/MiniMax-H3 ref2va set)
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"


class R2VWorkflowAdapter:
    """Reads workflow_r2v.json and emits a submit-ready flat API prompt."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or self._default_path()
        self.raw: dict = {}
        self.model = {
            "unet_name": UNET_REF2VA,
            "clip_name": CLIP_NAME,
            "vae_name": VAE_VIDEO,
            "vae_name_1": VAE_AUDIO,
            "turbo_lora": TURBO_LORA_NAME,
        }
        self.resolution = (1344, 768)
        self._load()
        self._extract()

    @staticmethod
    def _default_path() -> Path:
        if settings is not None and getattr(settings, "workflow_r2v_path", None):
            return Path(settings.workflow_r2v_path)
        return Path("./workflows/workflow_r2v.json")

    # ----------------------------------------------------------------- load ----
    def _load(self) -> None:
        if not self.path.exists():
            raise WorkflowError(f"R2V workflow file not found: {self.path}")
        try:
            self.raw = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            raise WorkflowError(f"R2V workflow JSON malformed: {e}") from e

    def _extract(self) -> None:
        """Read model names + default resolution from the template, if present.

        The real template carries these as ``nodes[].widgets_values`` and
        ``nodes[].properties.models``. We keep hard-coded official defaults as
        fallback but prefer the file when it has real data.
        """
        nodes = {n["id"]: n for n in self.raw.get("nodes", [])}

        def model_name(nid: int, field: str, fallback: str) -> str:
            n = nodes.get(str(nid)) or nodes.get(nid)
            if not n:
                return fallback
            wv = n.get("widgets_values") or []
            if wv and wv[0]:
                return wv[0]
            props = n.get("properties") or {}
            for m in props.get("models") or []:
                if m.get("name"):
                    return m["name"]
            return fallback

        self.model["unet_name"] = model_name(N_UNET, "unet_name", UNET_REF2VA)
        self.model["clip_name"] = model_name(N_CLIP, "clip_name", CLIP_NAME)
        self.model["vae_name"] = model_name(N_VAE_V, "vae_name", VAE_VIDEO)
        self.model["vae_name_1"] = model_name(N_VAE_A, "vae_name_1", VAE_AUDIO)
        self.model["turbo_lora"] = model_name(
            N_LORA, "turbo_lora", TURBO_LORA_NAME)

        core = nodes.get(str(N_CORE)) or nodes.get(N_CORE)
        if core:
            wv = core.get("widgets_values") or []
            # core widgets: [prompt, width, height, length, alignment]
            try:
                w = int(wv[1]); h = int(wv[2])
                if w > 0 and h > 0:
                    self.resolution = (w, h)
            except (TypeError, ValueError, IndexError):
                pass

    # -------------------------------------------------------------- validate ---
    def validate(self) -> Optional[list[str]]:
        nodes = {str(n["id"]): n for n in self.raw.get("nodes", [])}
        errs: list[str] = []
        for nid, want in (
            (N_UNET, "UNETLoader"), (N_CLIP, "CLIPLoader"),
            (N_VAE_V, "VAELoader"), (N_VAE_A, "VAELoader"),
            (N_CORE, "MiniMaxH3ReferenceToVideo"),
            (N_SAMPLER, "SamplerCustomAdvanced"),
            (N_CREATE, "CreateVideo"), (N_SAVE, "SaveVideo"),
        ):
            n = nodes.get(nid)
            got = n.get("type") if n else None
            if got != want:
                errs.append(f"node {nid} type {got} != {want}")
        if errs:
            raise WorkflowError("R2V workflow validation failed: " + "; ".join(errs))
        return None

    # -------------------------------------------------------------- build -----
    def build_prompt(
        self,
        *,
        prompt_text: str = "",
        duration: float = 2.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        ref_images: Optional[list[str]] = None,
        ref_videos: Optional[list[str]] = None,
        ref_audios: Optional[list[str]] = None,
        ref_image_size: str = "match",
        turbo: bool = False,
        turbo_steps: int = TURBO_STEPS,
        turbo_lora_strength: float = TURBO_LORA_STRENGTH,
        use_pdd: bool = False,
        pdd_nfe: str = PDD_NFE,
        pdd_lora_strength: float = PDD_LORA_STRENGTH,
        pdd_head_strength: float = PDD_HEAD_STRENGTH,
        model_overrides: Optional[dict] = None,
    ) -> dict:
        """Return a flat API graph for ComfyUI ``/prompt``.

        ``ref_images`` is a list of ComfyUI-side image filenames (already
        uploaded via /upload/image) to attach as ref_image_0..N. When empty,
        R2V still works text-only.
        """
        self.validate()
        model = dict(self.model)
        if model_overrides:
            for k, v in model_overrides.items():
                if v:
                    model[k] = v
        dur = self.parse_duration(duration)
        frames = duration_to_frames(dur)
        w, h = self._resolve_resolution(width, height)
        seed_val = seed if seed is not None else 0
        if turbo_steps is None:
            turbo_steps = TURBO_STEPS
        if turbo_lora_strength is None:
            turbo_lora_strength = TURBO_LORA_STRENGTH
        # PDD 模式忽略 turbo；它自带 8 步 sigmas，且官方要求不叠 turbo LoRA
        if use_pdd:
            turbo = False
        steps = pdd_nfe if use_pdd else (turbo_steps if turbo else NORMAL_STEPS)

        loaders = {
            N_UNET: {"class_type": "UNETLoader",
                     "inputs": {"unet_name": model["unet_name"],
                                "weight_dtype": "default"}},
            N_CLIP: {"class_type": "CLIPLoader",
                     "inputs": {"clip_name": model["clip_name"],
                                "type": "minimax", "device": "default"}},
            N_VAE_V: {"class_type": "VAELoader",
                      "inputs": {"vae_name": model["vae_name"]}},
            N_VAE_A: {"class_type": "VAELoader",
                      "inputs": {"vae_name": model["vae_name_1"]}},
        }

        # ref image LoadImage nodes (uploaded filenames)
        imgs: dict[str, dict] = {}
        refs = ref_images or []
        for i, fname in enumerate(refs[:3]):
            imgs[f"_ref{i}"] = {"class_type": "LoadImage",
                                "inputs": {"image": fname}}

        # core node inputs (flat API): clip/vae/audio_vae + ref_images.* + prompt/w/h/length
        core_inputs: dict[str, Any] = {
            "clip": [N_CLIP, 0],
            "vae": [N_VAE_V, 0],
            "audio_vae": [N_VAE_A, 0],
            "prompt": prompt_text,
            "width": w, "height": h, "length": int(frames),
            "ref_image_size": ref_image_size if ref_image_size in ("match", "max") else "match",
        }
        for i in range(3):
            key = f"ref_images.ref_image_{i}"
            core_inputs[key] = [f"_ref{i}", 0] if i < len(refs) else None
        # ref videos/audios leave unset (None) unless caller supplied
        # (the flat API accepts these only via uploaded paths; we keep them None
        #  to avoid referencing non-existent inputs, matching the template.)

        # turbo: lora loader on the unet branch, then a switch
        lora = {}
        switches = {}
        pdd = {}
        if turbo:
            lora[N_LORA] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [N_UNET, 0],
                    "lora_name": model["turbo_lora"],
                    "strength_model": turbo_lora_strength,
                },
            }
            switches[N_SW_MODEL] = {
                "class_type": "ComfySwitchNode",
                "inputs": {
                    "on_false": [N_UNET, 0],
                    "on_true": [N_LORA, 0],
                    "switch": True,
                },
            }
            model_src = [N_SW_MODEL, 0]
            # steps switch: on_false=normal(20) / on_true=turbo(4)
            switches[N_SW_STEPS] = {
                "class_type": "ComfySwitchNode",
                "inputs": {
                    "on_false": NORMAL_STEPS,
                    "on_true": TURBO_STEPS,
                    "switch": True,
                },
            }
            steps_src = [N_SW_STEPS, 0]
        elif use_pdd:
            # 阿里 PDD：unet -> UC_MiniMaxH3PDDAcc，用它的 model + sigmas 输出
            pdd[N_PDD] = {
                "class_type": "UC_MiniMaxH3PDDAcc",
                "inputs": {
                    "model": [N_UNET, 0],
                    "pdd_lora": PDD_LORA_NAME,
                    "nfe": pdd_nfe,
                    "partition": "",
                    "lora_strength": pdd_lora_strength,
                    "head_strength": pdd_head_strength,
                    "on_off_grid": "error",
                },
            }
            model_src = [N_PDD, 0]
            steps_src = [N_PDD, 1]  # sigmas 输出
        else:
            model_src = [N_UNET, 0]
            steps_src = NORMAL_STEPS

        if use_pdd:
            # PDD 模式：sampler 直接用 PDD 的 sigmas 输出（其步进边界），
            # 采样器必须 euler，且绕过 BasicScheduler
            sampler_name = "euler"
            sigmas_src = [N_PDD, 1]
            sched_node = {}
        else:
            sampler_name = "res_multistep"
            sigmas_src = [N_SCHED, 0]
            sched_node = {
                N_SCHED: {"class_type": "BasicScheduler",
                          "inputs": {"model": model_src, "scheduler": "simple",
                                     "steps": steps_src, "denoise": 1.0}},
            }

        graph = {
            **loaders, **imgs, **lora, **switches, **pdd, **sched_node,
            N_NOISE: {"class_type": "RandomNoise",
                      "inputs": {"noise_seed": seed_val}},
            N_CORE: {"class_type": "MiniMaxH3ReferenceToVideo",
                     "inputs": core_inputs},
            N_GUIDER: {"class_type": "BasicGuider",
                       "inputs": {"model": model_src,
                                  "conditioning": [N_CORE, 0]}},
            N_SEL: {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": sampler_name}},
            N_SAMPLER: {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": [N_NOISE, 0], "guider": [N_GUIDER, 0],
                "sampler": [N_SEL, 0], "sigmas": sigmas_src,
                "latent_image": [N_CORE, 1]}},
            N_VDEC: {"class_type": "VAEDecode",
                     "inputs": {"samples": [N_SAMPLER, 0],
                                "vae": [N_VAE_V, 0]}},
            N_ADEC: {"class_type": "VAEDecodeAudio",
                     "inputs": {"samples": [N_SAMPLER, 0],
                                "vae": [N_VAE_A, 0]}},
            N_CREATE: {"class_type": "CreateVideo",
                       "inputs": {"images": [N_VDEC, 0], "audio": [N_ADEC, 0],
                                  "fps": 24, "bit_depth": 8}},
            N_SAVE: {"class_type": "SaveVideo",
                     "inputs": {"video": [N_CREATE, 0],
                                "filename_prefix": "video/MiniMax_H3_R2V",
                                "format": "auto", "codec": "auto"}},
        }
        return graph

    def parse_duration(self, value: Any) -> float:
        try:
            dur = float(value)
        except (TypeError, ValueError) as e:
            raise WorkflowError(f"invalid duration: {value!r}") from e
        if not (1.0 <= dur <= 60.0):
            raise WorkflowError(f"duration out of range 1..60s: {dur}")
        return dur

    def _resolve_resolution(self, width, height):
        w, h = width, height
        if w is None or h is None:
            # Default to the T4-safe canvas instead of the template's native
            # 1344x768, which OOMs the 15GB T4 with the ref2va model.
            dw, dh = T4_SAFE_RESOLUTION if (w is None and h is None) else self.resolution
            w, h = w or dw, h or dh
        w = max(32, (int(round(w)) // 32) * 32)
        h = max(32, (int(round(h)) // 32) * 32)
        if w > 1920 or h > 1088:
            raise WorkflowError(f"resolution too large: {w}x{h} (max 1920x1088)")
        return w, h

    def describe(self) -> dict:
        try:
            self.validate()
            status, err = "ok", None
        except WorkflowError as e:
            status, err = "error", str(e)
        return {
            "workflow_path": str(self.path),
            "status": status,
            "error": err,
            "mode": "reference-to-video",
            "core_node": f"MiniMaxH3ReferenceToVideo (id {N_CORE})",
            "models": dict(self.model),
            "default_resolution": self.resolution,
            "duration_limits": [1.0, 60.0],
        }