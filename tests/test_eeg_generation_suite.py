from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_eeg_generation_suite import select_suite


def test_select_suite_is_balanced_and_held_out() -> None:
    rows = []
    for category in ("01", "02"):
        for number in range(1, 5):
            rows.append(
                {
                    "video_id": f"{category}-{number:03d}",
                    "session": "session3",
                    "duration_sec": "4.0",
                    "trial_index": str(number),
                    "length_samples": "800",
                }
            )
    selected = select_suite(
        rows,
        {row["video_id"] for row in rows},
        categories=["01", "02"],
        session="session3",
        duration_sec=4.0,
        per_category=1,
    )
    assert [row["category_id"] for row in selected] == ["01", "02"]
    assert [row["video_id"] for row in selected] == ["01-003", "02-003"]
