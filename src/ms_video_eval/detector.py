"""Detection backend for frame-level object localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .utils import ensure_dir, load_yaml, write_json


@dataclass
class DetectorSettings:
    """Runtime configuration for the detector."""

    backend: str = "yolo"
    model_path: str = "yolo11x.pt"
    conf_threshold: float = 0.25
    class_aliases: dict[str, list[str]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorSettings":
        aliases = data.get("class_aliases", {}) or {}
        return cls(
            backend=str(data.get("backend", "yolo")).strip(),
            model_path=str(data.get("model_path", "yolo11x.pt")).strip(),
            conf_threshold=float(data.get("conf_threshold", 0.25)),
            class_aliases={key: list(value) for key, value in aliases.items()},
        )


def load_detector_settings(path: Path) -> DetectorSettings:
    payload = load_yaml(path) or {}
    return DetectorSettings.from_dict(payload.get("detector", {}))


def _load_ultralytics_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install requirements-ms-eval.txt on the GPU server "
            "before running detection."
        ) from exc
    return YOLO(model_path)


def _reverse_alias_map(class_aliases: dict[str, list[str]]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for canonical, aliases in class_aliases.items():
        for alias in aliases:
            reverse[alias.lower()] = canonical
        reverse[canonical.lower()] = canonical
    return reverse


def _map_class_name(raw_name: str, reverse_aliases: dict[str, str]) -> str | None:
    key = raw_name.lower().strip()
    if key in reverse_aliases:
        return reverse_aliases[key]
    return None


class FrameDetector:
    """Lazy YOLO detector reused across many frame directories."""

    def __init__(self, settings: DetectorSettings) -> None:
        if settings.backend.lower() != "yolo":
            raise ValueError(f"Unsupported detector backend: {settings.backend}")
        self.settings = settings
        self.model = None
        self.reverse_aliases = _reverse_alias_map(settings.class_aliases or {})

    def _ensure_model(self):
        if self.model is None:
            self.model = _load_ultralytics_model(self.settings.model_path)
        return self.model

    def detect_frames(self, frames_dir: Path, output_path: Path) -> dict[str, Any]:
        """Run object detection on a directory of extracted frames."""

        model = self._ensure_model()
        frame_files = sorted(
            [path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        detections: list[dict[str, Any]] = []
        for frame_idx, frame_path in enumerate(frame_files):
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            result = model.predict(source=image, conf=self.settings.conf_threshold, verbose=False)[0]
            frame_detections: list[dict[str, Any]] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    raw_name = result.names[int(box.cls)]
                    canonical = _map_class_name(raw_name, self.reverse_aliases)
                    if canonical is None:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    frame_detections.append(
                        {
                            "frame_idx": frame_idx,
                            "class_name": canonical,
                            "raw_class_name": raw_name,
                            "confidence": float(box.conf[0]),
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
            detections.append(
                {
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path),
                    "detections": frame_detections,
                }
            )

        payload = {
            "frames_dir": str(frames_dir),
            "model_path": self.settings.model_path,
            "conf_threshold": self.settings.conf_threshold,
            "detections": detections,
        }
        write_json(output_path, payload)
        return payload


def detect_frames(
    frames_dir: Path,
    output_path: Path,
    settings: DetectorSettings,
) -> dict[str, Any]:
    """Run object detection on a directory of extracted frames."""

    return FrameDetector(settings).detect_frames(frames_dir, output_path)
