"""Tracking extension points for future mask or box tracking backends.

The first benchmark version computes temporal metrics directly from frame-level
detections. This module is intentionally lightweight so future SAM2, box IoU,
or optical-flow trackers can plug into the same package without changing the
script surface.
"""

from __future__ import annotations

from typing import Any


def identity_tracks(frame_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return detections unchanged as a placeholder tracking strategy."""

    return frame_detections

