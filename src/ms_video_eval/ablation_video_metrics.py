"""Model-agnostic diagnostics for generated EEG reconstruction videos."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def sample_video(path: Path, sample_every: int = 4) -> list[np.ndarray]:
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    capture = cv2.VideoCapture(str(path)); frames = []; index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % sample_every == 0:
            frames.append(frame)
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"No readable frames in {path}")
    return frames


def frame_diagnostics(frames: Sequence[np.ndarray]) -> dict[str, float]:
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    sharpness = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in gray]
    brightness = [float(frame.mean()) / 255.0 for frame in gray]
    differences = [
        float(np.abs(right.astype(np.float32) - left.astype(np.float32)).mean()) / 255.0
        for left, right in zip(gray, gray[1:])
    ]
    motion = float(np.mean(differences)) if differences else 0.0
    return {
        "temporal_consistency": 1.0 - motion,
        "motion_energy": motion,
        "sharpness_score": float(np.mean([1.0 - math.exp(-value / 100.0) for value in sharpness])),
        "exposure_valid_rate": float(np.mean([0.05 <= value <= 0.95 for value in brightness])),
    }


def parse_trajectory_points(path: Path) -> np.ndarray:
    numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", path.read_text(encoding="utf-8"))]
    if len(numbers) < 4 or len(numbers) % 2:
        raise ValueError(f"Could not parse coordinate pairs from {path}")
    return np.asarray(numbers, dtype=np.float32).reshape(-1, 2)


def optical_flow_direction(frames: Sequence[np.ndarray]) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(2, dtype=np.float32)
    vectors = []
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=-1)
        mask = magnitude >= np.quantile(magnitude, 0.75)
        vectors.append(flow[mask].mean(axis=0) if mask.any() else flow.reshape(-1, 2).mean(axis=0))
        previous = current
    return np.asarray(vectors).mean(axis=0)


def trajectory_direction_score(frames: Sequence[np.ndarray], points: np.ndarray) -> float:
    expected = points[-1] - points[0]
    observed = optical_flow_direction(frames)
    if np.linalg.norm(expected) < 1e-8:
        return float(math.exp(-float(np.linalg.norm(observed))))
    if np.linalg.norm(observed) < 1e-8:
        return 0.0
    cosine = float(np.dot(expected, observed) / (np.linalg.norm(expected) * np.linalg.norm(observed)))
    return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0


class CLIPFrameScorer:
    """Lazy Transformers CLIP scorer; model can be a local directory."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch; self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score(self, frames: Sequence[np.ndarray], text: str) -> float:
        from PIL import Image
        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs)
            image = self.torch.nn.functional.normalize(output.image_embeds, dim=-1)
            text_state = self.torch.nn.functional.normalize(output.text_embeds, dim=-1)
        return float(((image @ text_state.t()).mean() + 1) / 2)
