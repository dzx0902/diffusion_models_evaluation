from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_eeg_ablation_generation import audit


def test_audit_accepts_matched_fixed_trajectory() -> None:
    rows = [
        {"variant": variant, "generator": "tora", "video_id": "v1", "seed": 0,
         "trajectory_sha256": "same"}
        for variant in ("a", "b")
    ]
    assert audit(rows)["status"] == "PASS"


def test_audit_rejects_changed_trajectory() -> None:
    rows = [
        {"variant": "a", "generator": "tora", "video_id": "v1", "seed": 0,
         "trajectory_sha256": "one"},
        {"variant": "b", "generator": "tora", "video_id": "v1", "seed": 0,
         "trajectory_sha256": "two"},
    ]
    assert audit(rows)["status"] == "FAIL"


def test_audit_filters_and_aliases_tora_routes() -> None:
    rows = [
        {"variant": "a", "generator": "tora", "video_id": "v1", "seed": 42,
         "generation_seed": 0, "trajectory_sha256": "same"},
        {"variant": "a", "generator": "animatediff", "video_id": "v1", "seed": 42,
         "generation_seed": 0},
        {"variant": "c", "generator": "tora_injected", "video_id": "v1", "seed": 42,
         "generation_seed": 0, "trajectory_sha256": "same"},
    ]
    result = audit(
        rows,
        {"tora", "tora_injected"},
        {"tora_injected": "tora"},
    )
    assert result["status"] == "PASS"
    assert result["matched_cells"] == 1
