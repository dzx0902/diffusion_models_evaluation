"""Utilities for converting sparse entity detections into Tora point tracks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def interpolate_centers(
    centers: Sequence[tuple[float, float] | None],
    fallback: tuple[float, float] = (0.5, 0.5),
) -> tuple[np.ndarray, float]:
    """Linearly fill missing normalized centers and report detection coverage."""

    count = len(centers)
    if count < 2:
        raise ValueError("A trajectory needs at least two frames")
    valid = np.asarray([index for index, value in enumerate(centers) if value is not None])
    coverage = len(valid) / count
    if not len(valid):
        return np.tile(np.asarray(fallback, dtype=np.float32), (count, 1)), 0.0
    values = np.asarray([centers[index] for index in valid], dtype=np.float32)
    timeline = np.arange(count)
    result = np.stack(
        [np.interp(timeline, valid, values[:, axis]) for axis in range(2)], axis=1
    ).astype(np.float32)
    return np.clip(result, 0.0, 1.0), coverage


def tora_canvas_points(normalized: np.ndarray) -> np.ndarray:
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise ValueError("Expected normalized [frames,2] points")
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255).astype(np.int32)
