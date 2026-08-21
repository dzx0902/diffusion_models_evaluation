from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_clip_video_native_screen import backend_defaults, frames_to_uint8, load_captions


def test_backend_defaults() -> None:
    assert backend_defaults("animatediff") == ("bfloat16", 512, 512)
    assert backend_defaults("zeroscope") == ("float16", 320, 576)


def test_load_captions_requires_every_requested_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"video_id": "01-001", "caption": "A person kicks a ball."}) + "\n")
    assert load_captions(manifest, ["01-001"])["01-001"] == "A person kicks a ball."
    with pytest.raises(ValueError, match="Missing requested video IDs"):
        load_captions(manifest, ["02-040"])


def test_frames_to_uint8_scales_diffusers_float_frames() -> None:
    frames = [np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)]
    converted = frames_to_uint8(frames)
    assert converted.dtype == np.uint8
    assert converted.tolist() == [[[[0, 128, 255]]]]
