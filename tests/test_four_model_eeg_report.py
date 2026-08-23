from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_four_model_eeg_report import classify_video, validate_matched_groups


class FourModelEEGReportTest(unittest.TestCase):
    def test_classifies_generated_filenames(self) -> None:
        self.assertEqual(
            classify_video("01-001_wan_pca512_exact_seed0.mp4"),
            "wan_pca512_exact",
        )
        self.assertEqual(
            classify_video("01-001_cogvideox2b_eeg_seed0.mp4"),
            "cogvideox2b_eeg",
        )

    def test_rejects_unmatched_groups(self) -> None:
        rows = [
            {"benchmark_group": group, "video_id": "01-001"}
            for group in (
                "wan_pca512_exact",
                "wan_pca512_eeg",
                "animatediff_eeg",
                "zeroscope_eeg",
            )
        ]
        with self.assertRaisesRegex(ValueError, "Missing benchmark groups"):
            validate_matched_groups(rows)


if __name__ == "__main__":
    unittest.main()
