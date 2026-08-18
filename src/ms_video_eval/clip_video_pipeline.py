"""Shared Diffusers pipeline loading and prompt encoding for CLIP video models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def has_variant_weights(path: Path, variant: str) -> bool:
    return any(path.glob(f"*.{variant}.*"))


def has_default_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        item.is_file()
        and item.suffix in {".bin", ".safetensors"}
        and ".fp16." not in item.name
        for item in path.iterdir()
    )


def pipeline_requires_variant(model_root: Path, variant: str) -> bool:
    return all(
        not has_default_weights(model_root / component)
        and has_variant_weights(model_root / component, variant)
        for component in ("text_encoder", "unet", "vae")
    )


def load_clip_video_pipeline(
    *,
    backend: str,
    model_root: Path,
    motion_adapter: Path | None,
    dtype: torch.dtype,
) -> Any:
    """Load the same official Diffusers pipeline for export and generation."""
    if backend == "animatediff":
        if motion_adapter is None:
            raise ValueError("motion_adapter is required for AnimateDiff")
        from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

        adapter_options: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if not has_default_weights(motion_adapter) and has_variant_weights(motion_adapter, "fp16"):
            adapter_options["variant"] = "fp16"
        adapter = MotionAdapter.from_pretrained(motion_adapter, **adapter_options)

        pipeline_options: dict[str, Any] = {
            "motion_adapter": adapter,
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if pipeline_requires_variant(model_root, "fp16"):
            pipeline_options["variant"] = "fp16"
        pipe = AnimateDiffPipeline.from_pretrained(model_root, **pipeline_options)
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        return pipe

    if backend == "zeroscope":
        from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            model_root,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe

    raise ValueError(f"Unsupported backend: {backend}")


def pipeline_execution_device(pipe: Any, fallback: torch.device) -> torch.device:
    return torch.device(getattr(pipe, "_execution_device", fallback))


def encode_prompt_with_pipeline(
    pipe: Any,
    *,
    prompt: str | None,
    device: torch.device,
    guidance_scale: float,
    negative_prompt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the pipeline's public prompt encoder for both native and injected paths."""
    encode_prompt = getattr(pipe, "encode_prompt", None)
    if not callable(encode_prompt):
        raise RuntimeError(
            f"{type(pipe).__name__} does not expose public encode_prompt(); "
            "use a supported Diffusers pipeline version"
        )

    do_cfg = guidance_scale > 1.0
    encoded = encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
    )
    if isinstance(encoded, torch.Tensor):
        return encoded, None
    if not isinstance(encoded, tuple) or not encoded:
        raise RuntimeError(f"Unexpected encode_prompt() result: {type(encoded).__name__}")
    positive = encoded[0]
    negative = encoded[1] if len(encoded) > 1 else None
    if not isinstance(positive, torch.Tensor):
        raise RuntimeError("encode_prompt() did not return prompt embeddings")
    if negative is not None and not isinstance(negative, torch.Tensor):
        raise RuntimeError("encode_prompt() returned invalid negative prompt embeddings")
    return positive, negative
