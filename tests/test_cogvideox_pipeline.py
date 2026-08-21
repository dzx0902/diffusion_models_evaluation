from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.cogvideox_pipeline import encode_cogvideox_prompt, resolve_cogvideox_dtype


def test_resolve_cogvideox_dtype() -> None:
    assert resolve_cogvideox_dtype("float16") == torch.float16
    with pytest.raises(ValueError, match="Unsupported CogVideoX dtype"):
        resolve_cogvideox_dtype("missing")


def test_encode_cogvideox_prompt_uses_public_pipeline_contract() -> None:
    positive = torch.randn(1, 226, 4096)
    negative = torch.randn(1, 226, 4096)
    pipe = Mock()
    pipe.encode_prompt.return_value = (positive, negative)
    actual = encode_cogvideox_prompt(
        pipe,
        prompt=None,
        device=torch.device("cpu"),
        dtype=torch.float16,
        guidance_scale=6.0,
        prompt_embeds=positive,
    )
    assert actual == (positive, negative)
    pipe.encode_prompt.assert_called_once_with(
        prompt=None,
        negative_prompt="",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=positive,
        max_sequence_length=226,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
