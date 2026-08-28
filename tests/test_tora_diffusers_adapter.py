from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "scripts" / "adapters"
if str(ADAPTERS) not in sys.path:
    sys.path.insert(0, str(ADAPTERS))

from tora_diffusers_condition_generate import load_control, validate_paths


def test_load_control_reads_full_condition(tmp_path: Path) -> None:
    path = tmp_path / "condition.pt"
    torch.save(
        {
            "video_id": "01-001",
            "caption": "A person kicks a ball.",
            "hidden_state": torch.zeros(226, 4096),
            "control": "fixture",
        },
        path,
    )
    caption, state, metadata = load_control(path)
    assert caption == "A person kicks a ball."
    assert state.shape == (226, 4096)
    assert metadata == {"video_id": "01-001", "control": "fixture"}


def test_validate_paths_rejects_invalid_frame_count(tmp_path: Path) -> None:
    repo = tmp_path / "Tora"
    model = tmp_path / "model"
    condition = tmp_path / "condition.pt"
    point = tmp_path / "point.txt"
    for path in (
        repo / "diffusers-version/tora/t2v_pipeline.py",
        repo / "diffusers-version/tora/traj_utils.py",
        model / "model_index.json",
        condition,
        point,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    args = argparse.Namespace(
        tora_repo=repo,
        model_root=model,
        condition=condition,
        point_path=[point],
        num_frames=48,
        height=480,
        width=720,
    )
    with pytest.raises(ValueError, match="num_frames"):
        validate_paths(args)
