from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_four_model_eeg_benchmark import read_suite


class FourModelEEGBenchmarkTest(unittest.TestCase):
    def test_read_suite_accepts_unique_four_second_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"video_id": video_id, "duration_sec": 4.0, "session": "session3"})
                    for video_id in ("01-001", "02-001")
                ),
                encoding="utf-8",
            )

            rows = read_suite(path)

        self.assertEqual([row["video_id"] for row in rows], ["01-001", "02-001"])

    def test_read_suite_rejects_duplicate_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            row = json.dumps({"video_id": "01-001", "duration_sec": 4.0, "session": "session3"})
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_suite(path)


if __name__ == "__main__":
    unittest.main()
