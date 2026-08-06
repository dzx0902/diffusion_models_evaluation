"""Shared validation helpers for EEG trial manifests."""

from __future__ import annotations

from typing import Any


def trial_duration(row: dict[str, Any]) -> float:
    """Return and cross-check a trial's duration metadata."""

    duration = float(row["duration_sec"])
    samples = int(row["length_samples"])
    sfreq = float(row["sfreq"])
    measured = samples / sfreq
    if abs(duration - measured) > 1e-6:
        raise ValueError(
            f"{row.get('video_id', '<unknown>')}/{row.get('session', '<unknown>')}: "
            f"duration_sec={duration} differs from length_samples/sfreq={measured}"
        )
    return duration


def filter_trial_duration(
    rows: list[dict[str, Any]],
    duration_sec: float | None,
) -> list[dict[str, Any]]:
    """Validate all rows and optionally retain one exact stimulus duration."""

    validated = [(row, trial_duration(row)) for row in rows]
    if duration_sec is None:
        return [row for row, _ in validated]
    return [row for row, duration in validated if abs(duration - duration_sec) <= 1e-6]
