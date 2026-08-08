"""Workflow adapter for the real MiniMax-H3 workflow (ComfyUI 0.4 subgraph).

``workflow.json`` is an *editable* graph: a single top-level "subgraph-node"
(id 105) invokes an inner definition (``definitions.subgraphs[0]``) holding the
actual nodes: model loaders, ``MiniMaxH3ImageToVideo`` core node, the sampler
chain, and the video/audio decoders.

This adapter loads that editable graph, validates it against the exact node
ids/types, reads model names and resolution semantics from the file itself, and
flattens the inner chain into a faithful *API-format* prompt payload (the flat
``{key: {class_type, inputs}}`` graph POSTed to ComfyUI ``/prompt``).

Node class names are read from the subgraph's own ``nodes[].type``; loader
names are read from the top-level subgraph-invocation node's widgets. Only call
parameters are injected: prompt text, duration (seconds -> grid-snapped frame
``length``), width/height, seed, and the two frame IMAGE inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import settings


class WorkflowError(Exception):
    """Validation / conversion failure -> Permanent job error."""


# ------------------------------------------------------------------ grid ------
def _snap_to_grid(frames: int) -> int:
    """Mirror the ComfyMathExpression snap-to-17k+5-grid."""
    base = max(5, round(frames))
    return base + (5 - (base % 17)) % 17


def duration_to_frames(seconds: float) -> int:
    """seconds @24fps -> grid-snapped frame count."""
    return _snap_to_grid(seconds * 24.0)


# ------------------------------------------------------ stable subgraph ids ---
N_UNET = "6"      # UNETLoader
N_CLIP = "13"     # CLIPLoader
N_VAE_V = "11"    # VAELoader (video)
N_VAE_A = "24"    # VAELoader (audio)
N_NOISE = "15"    # RandomNoise
N_MIV = "104"     # MiniMaxH3ImageToVideo
N_GUIDER = "16"   # BasicGuider
N_SCHED = "9"     # BasicScheduler
N_SEL = "17"      # KSamplerSelect
N_SAMPLER = "14"  # SamplerCustomAdvanced
N_VDEC = "10"     # VAEDecode
N_ADEC = "23"     # VAEDecodeAudio
N_CREATE = "91"   # CreateVideo
N_SAVE = "92"     # SaveVideo

_SUBGRAPH_NODE_ID = 105

_DEFAULT_STEPS = 20
_DEFAULT_DENOISE = 1.0


class WorkflowAdapter:
    """Reads workflow.json and emits a submit-ready flat API prompt."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (
            Path(settings.workflow_path) if settings else Path("./workflows/workflow.json")
        )
        self.raw: dict = {}
        self._sub_nodes: dict[int, dict] = {}
        self._top_nodes: dict[int, dict] = {}
        self.model = {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "vae_name": "minimax_h3_video_vae_fp16.safetensors",
            "vae_name_1": "minimax_h3_audio_vae_fp32.safetensors",
        }
        self.resolution = (1344, 768)
        self._load()
        self._extract()

    # ----------------------------------------------------------------- log ----
    def _load(self) -> None:
        if not self.path.exists():
            raise WorkflowError(f"workflow file not found: {self.path}")
        try:
            self.raw = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:  # pragma: no cover
            raise WorkflowError(f"workflow JSON malformed: {e}") from e
        subs = self.raw.get("definitions", {}).get("subgraphs")
        if not subs:
            raise WorkflowError("workflow has no definitions.subgraphs")
        self._sub_nodes = {n["id"]: n for n in subs[0].get("nodes", [])}
        self._top_nodes = {n["id"]: n for n in self.raw.get("nodes", [])}

    def _extract(self) -> None:
        """Read model names + default resolution from the file (not hard-coded)."""
        inv = self._top_nodes.get(_SUBGRAPH_NODE_ID)
        if inv and inv.get("widgets_values"):
            vals = inv["widgets_values"]
            # [prompt, width, height, value_1(duration), noise_seed,
            #  unet_name, clip_name, vae_name, vae_name_1]
            order = ["prompt", "width", "height", "value_1", "noise_seed",
                     "unet_name", "clip_name", "vae_name", "vae_name_1"]
            if len(vals) >= len(order):
                for key in ("unet_name", "clip_name", "vae_name", "vae_name_1"):
                    val = vals[order.index(key)]
                    if val:
                        self.model[key] = val
                try:
                    w = int(vals[order.index("width")])
                    h = int(vals[order.index("height")])
                    if w > 0 and h > 0:
                        self.resolution = (w, h)
                except (TypeError, ValueError):
                    pass

    # -------------------------------------------------------------- validate ---
    def validate(self) -> Optional[list[str]]:
        errs: list[str] = []
        if _SUBGRAPH_NODE_ID not in self._top_nodes:
            errs.append(f"missing top-level subgraph node {_SUBGRAPH_NODE_ID}")
        # Nodes located in the inner subgraph definition.
        need_sub = {
            int(N_MIV): "MiniMaxH3ImageToVideo",
            int(N_UNET): "UNETLoader", int(N_CLIP): "CLIPLoader",
            int(N_VAE_V): "VAELoader", int(N_VAE_A): "VAELoader",
            int(N_NOISE): "RandomNoise",
            int(N_SAMPLER): "SamplerCustomAdvanced",
            int(N_CREATE): "CreateVideo",
        }
        # Nodes located at the top-level graph (post-subgraph output).
        need_top = {
            int(N_SAVE): "SaveVideo",
        }
        for nid, want in need_sub.items():
            got = self.sub_node_type(nid)
            if got != want:
                errs.append(f"subgraph node {nid} type {got} != {want}")
        for nid, want in need_top.items():
            got = self._top_nodes.get(nid)
            got_t = got.get("type") if got else None
            if got_t != want:
                errs.append(f"top-level node {nid} type {got_t} != {want}")
        if errs:
            raise WorkflowError("workflow validation failed: " + "; ".join(errs))
        return None

    def sub_node_type(self, node_id: int) -> Optional[str]:
        n = self._sub_nodes.get(node_id)
        return n.get("type") if n else None

    # ------------------------------------------------------------------ build --
    def build_prompt(
        self,
        *,
        first_frame: Optional[str] = None,
        last_frame: Optional[str] = None,
        prompt_text: str = "",
        duration: float = 2.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        model_overrides: Optional[dict] = None,
    ) -> dict:
        """Return a flat API graph for ComfyUI ``/prompt``."""
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

        loaders = {
            N_UNET: {"class_type": "UNETLoader",
                     "inputs": {"unet_name": model["unet_name"]}},
            N_CLIP: {"class_type": "CLIPLoader",
                     "inputs": {"clip_name": model["clip_name"]}},
            N_VAE_V: {"class_type": "VAELoader",
                      "inputs": {"vae_name": model["vae_name"]}},
            N_VAE_A: {"class_type": "VAELoader",
                      "inputs": {"vae_name": model["vae_name_1"]}},
        }

        imgs: dict[str, dict] = {}
        refs: dict[str, list] = {}
        if first_frame:
            imgs["_load_first"] = {"class_type": "LoadImage",
                                   "inputs": {"image": first_frame}}
            refs["first"] = ["_load_first", 0]
        if last_frame:
            imgs["_load_last"] = {"class_type": "LoadImage",
                                  "inputs": {"image": last_frame}}
            refs["last"] = ["_load_last", 0]
        if "first" not in refs:
            raise WorkflowError("a first_frame image is required by the workflow")

        miv = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": [N_CLIP, 0],
                "vae": [N_VAE_V, 0],
                "first_frame": refs["first"],
                "last_frame": refs.get("last", refs["first"]),
                "prompt": prompt_text,
                "width": w, "height": h, "length": int(frames),
            },
        }

        graph = {
            **loaders,
            **imgs,
            N_NOISE: {"class_type": "RandomNoise", "inputs": {"noise_seed": seed_val}},
            N_MIV: miv,
            N_GUIDER: {"class_type": "BasicGuider",
                       "inputs": {"model": [N_UNET, 0], "conditioning": [N_MIV, 0]}},
            N_SCHED: {"class_type": "BasicScheduler",
                      "inputs": {"model": [N_UNET, 0], "scheduler": "simple",
                                 "steps": _DEFAULT_STEPS, "denoise": _DEFAULT_DENOISE}},
            N_SEL: {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "res_multistep"}},
            N_SAMPLER: {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": [N_NOISE, 0], "guider": [N_GUIDER, 0],
                "sampler": [N_SEL, 0], "sigmas": [N_SCHED, 0],
                "latent_image": [N_MIV, 1]}},
            N_VDEC: {"class_type": "VAEDecode",
                     "inputs": {"samples": [N_SAMPLER, 0], "vae": [N_VAE_V, 0]}},
            N_ADEC: {"class_type": "VAEDecodeAudio",
                     "inputs": {"samples": [N_SAMPLER, 0], "vae": [N_VAE_A, 0]}},
            N_CREATE: {"class_type": "CreateVideo",
                       "inputs": {"images": [N_VDEC, 0], "audio": [N_ADEC, 0],
                                  "fps": 24, "bit_depth": 8}},
            N_SAVE: {"class_type": "SaveVideo",
                     "inputs": {"video": [N_CREATE, 0],
                                "filename_prefix": "video/MiniMax_H3",
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

    def _resolve_resolution(self, width: Optional[int], height: Optional[int]) -> tuple[int, int]:
        w, h = width, height
        if w is None or h is None:
            dw, dh = self.resolution
            w, h = w or dw, h or dh
        w = max(32, (int(round(w)) // 32) * 32)
        h = max(32, (int(round(h)) // 32) * 32)
        if w > 1920 or h > 1088:
            raise WorkflowError(f"resolution too large: {w}x{h} (max 1920x1088)")
        return w, h

    # ------------------------------------------------------------- describe --
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
            "subgraph_node": _SUBGRAPH_NODE_ID,
            "core_node": f"MiniMaxH3ImageToVideo (internal id {N_MIV})",
            "models": dict(self.model),
            "default_resolution": self.resolution,
            "duration_limits": [1.0, 60.0],
            "frames_for_duration_s": {
                s: duration_to_frames(float(s)) for s in (1, 2, 5, 10)
            },
        }