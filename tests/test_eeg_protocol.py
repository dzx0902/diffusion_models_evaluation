import unittest

from src.ms_video_eval.eeg_protocol import filter_trial_duration, trial_duration


def trial(video_id: str, duration: float, samples: int) -> dict[str, str]:
    return {
        "video_id": video_id,
        "session": "session1",
        "duration_sec": str(duration),
        "length_samples": str(samples),
        "sfreq": "200",
    }


class EEGProtocolTest(unittest.TestCase):
    def test_filters_exact_duration_after_validation(self) -> None:
        rows = [trial("01-001", 4.0, 800), trial("07-001", 6.0, 1200)]

        selected = filter_trial_duration(rows, 4.0)

        self.assertEqual([row["video_id"] for row in selected], ["01-001"])

    def test_none_retains_all_valid_rows(self) -> None:
        rows = [trial("01-001", 4.0, 800), trial("07-001", 6.0, 1200)]

        self.assertEqual(filter_trial_duration(rows, None), rows)

    def test_rejects_inconsistent_duration_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from length_samples/sfreq"):
            trial_duration(trial("01-001", 4.0, 799))


if __name__ == "__main__":
    unittest.main()
