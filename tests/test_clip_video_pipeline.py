from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.clip_video_pipeline import encode_prompt_with_pipeline, resolve_dtype


class FakePipeline:
    def __init__(self) -> None:
        self.kwargs = None

    def encode_prompt(self, **kwargs):
        self.kwargs = kwargs
        positive = kwargs.get("prompt_embeds")
        if positive is None:
            positive = torch.ones(1, 77, 768)
        negative = torch.zeros_like(positive) if kwargs["do_classifier_free_guidance"] else None
        return positive, negative


def test_resolve_dtype() -> None:
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int8")


def test_encode_prompt_uses_public_pipeline_api() -> None:
    pipe = FakePipeline()
    positive, negative = encode_prompt_with_pipeline(
        pipe,
        prompt="A person kicks a ball.",
        negative_prompt="",
        device=torch.device("cpu"),
        guidance_scale=7.5,
    )

    assert positive.shape == (1, 77, 768)
    assert negative is not None
    assert pipe.kwargs["prompt"] == "A person kicks a ball."
    assert pipe.kwargs["do_classifier_free_guidance"] is True


def test_encode_prompt_preserves_injected_tensor() -> None:
    pipe = FakePipeline()
    injected = torch.randn(1, 77, 768)
    positive, negative = encode_prompt_with_pipeline(
        pipe,
        prompt=None,
        negative_prompt="",
        prompt_embeds=injected,
        device=torch.device("cpu"),
        guidance_scale=7.5,
    )

    assert positive is injected
    assert negative is not None
    assert pipe.kwargs["prompt_embeds"] is injected
