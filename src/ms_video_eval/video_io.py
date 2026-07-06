"""Video and image I/O helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .task_schema import TaskSpec
from .utils import ensure_dir


POSITION_BOXES = {
    "left": (0.08, 0.30, 0.34, 0.70),
    "right": (0.66, 0.30, 0.92, 0.70),
    "center": (0.36, 0.32, 0.64, 0.68),
    "upper left": (0.06, 0.08, 0.32, 0.38),
    "upper right": (0.68, 0.08, 0.94, 0.38),
    "lower left": (0.06, 0.62, 0.32, 0.92),
    "lower right": (0.68, 0.62, 0.94, 0.92),
    "upper": (0.30, 0.06, 0.70, 0.36),
    "lower": (0.30, 0.64, 0.70, 0.94),
}


def _region_box(position: str, width: int, height: int) -> tuple[int, int, int, int]:
    pos = position.strip().lower()
    for key, norm_box in POSITION_BOXES.items():
        if key in pos:
            x1 = int(norm_box[0] * width)
            y1 = int(norm_box[1] * height)
            x2 = int(norm_box[2] * width)
            y2 = int(norm_box[3] * height)
            return x1, y1, x2, y2
    return int(0.25 * width), int(0.25 * height), int(0.75 * width), int(0.75 * height)


def create_pseudo_reference_image(
    task: TaskSpec,
    output_path: Path,
    width: int = 768,
    height: int = 432,
) -> Path:
    """Create a lightweight pseudo-reference image for I2V/TI2V runs."""

    ensure_dir(output_path.parent)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (235, 242, 247)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (180, 180, 180), 2)

    title = f"Pseudo-reference for {task.id}"
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)

    for idx, subject in enumerate(task.subjects[:2]):
        x1, y1, x2, y2 = _region_box(subject.initial_position, width, height)
        color = (60 + 70 * idx, 110 + 40 * idx, 200 - 50 * idx)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"{subject.name} | {subject.initial_position}"
        cv2.putText(
            canvas,
            label,
            (x1 + 8, min(y2 - 12, y1 + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        motion = subject.motion[:48]
        cv2.putText(
            canvas,
            motion,
            (x1 + 8, min(y2 - 12, y1 + 58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (55, 55, 55),
            1,
        )

    scene_text = f"Scene: {task.scene} | Camera: {task.camera}"
    cv2.putText(canvas, scene_text, (24, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def extract_frames(video_path: Path, output_dir: Path, sample_every: int = 4) -> list[Path]:
    """Extract one frame every N frames from a video."""

    ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved: list[Path] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            frame_path = output_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved.append(frame_path)
            saved_idx += 1
        frame_idx += 1
    capture.release()
    return saved


def parse_generation_filename(path: Path) -> tuple[str, str, int]:
    """Parse model id, task id, and seed from a generated video path."""

    model_id = path.parent.name
    match = re.match(r"(.+)_seed(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse task/seed from filename: {path.name}")
    task_id = match.group(1)
    seed = int(match.group(2))
    return model_id, task_id, seed

