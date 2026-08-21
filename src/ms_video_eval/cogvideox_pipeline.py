"""Shared official Diffusers loading and encoding for CogVideoX-2B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_cogvideox_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported CogVideoX dtype: {name}") from error


def load_cogvideox_pipeline(model_root: Path, dtype: torch.dtype) -> Any:
    from diffusers import CogVideoXPipeline

    return CogVideoXPipeline.from_pretrained(
        model_root,
        torch_dtype=dtype,
        local_files_only=True,
    )


def encode_cogvideox_prompt(
    pipe: Any,
    *,
    prompt: str | None,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt: str = "",
    max_sequence_length: int = 226,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Call CogVideoX's public encoder for native or injected conditions."""
    positive, negative = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=guidance_scale > 1.0,
        num_videos_per_prompt=1,
        prompt_embeds=prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )
    return positive, negative
