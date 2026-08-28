from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_video_metrics import frame_diagnostics, trajectory_direction_score


def test_frame_diagnostics_are_bounded() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 10, dtype=np.uint8)]
    values = frame_diagnostics(frames)
    assert 0 <= values["temporal_consistency"] <= 1
    assert 0 <= values["sharpness_score"] <= 1


def test_stationary_trajectory_rewards_stationary_frames() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    points = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    assert trajectory_direction_score(frames, points) == 1.0
