from __future__ import annotations

import sys
from pathlib import Path
import os
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.clip_video_pipeline import (
    encode_prompt_with_pipeline,
    has_default_weights,
    pipeline_requires_variant,
    resolve_dtype,
)


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


def test_clip_pipeline_import_does_not_require_opencv() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'cv2' or name.startswith('cv2.'):
        raise ModuleNotFoundError('blocked cv2 for import isolation test')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import ms_video_eval.clip_video_pipeline
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_default_weights_take_precedence_over_variant(tmp_path: Path) -> None:
    for component in ("text_encoder", "unet", "vae"):
        directory = tmp_path / component
        directory.mkdir()
        (directory / "diffusion_pytorch_model.safetensors").write_bytes(b"default")
        (directory / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"variant")

    assert has_default_weights(tmp_path / "unet")
    assert not pipeline_requires_variant(tmp_path, "fp16")


def test_variant_is_used_only_when_default_weights_are_absent(tmp_path: Path) -> None:
    for component in ("text_encoder", "unet", "vae"):
        directory = tmp_path / component
        directory.mkdir()
        (directory / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"variant")

    assert not has_default_weights(tmp_path / "unet")
    assert pipeline_requires_variant(tmp_path, "fp16")
