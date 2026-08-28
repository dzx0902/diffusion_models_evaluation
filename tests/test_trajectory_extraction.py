from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.trajectory_extraction import interpolate_centers, tora_canvas_points


def test_interpolation_fills_internal_and_edge_gaps() -> None:
    points, coverage = interpolate_centers([None, (0.0, 0.0), None, (1.0, 1.0), None])
    assert coverage == 0.4
    assert np.allclose(points[0], [0, 0])
    assert np.allclose(points[2], [0.5, 0.5])
    assert np.allclose(points[4], [1, 1])


def test_empty_track_uses_center_fallback_and_tora_canvas() -> None:
    points, coverage = interpolate_centers([None, None])
    assert coverage == 0
    canvas = tora_canvas_points(points)
    assert canvas.tolist() == [[128, 128], [128, 128]]
