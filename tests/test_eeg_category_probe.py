from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_eeg_category_probe import category_from_video_id, classification_metrics


def test_category_from_video_id_accepts_both_separators() -> None:
    assert category_from_video_id("01-078") == "01"
    assert category_from_video_id("06_001") == "06"


def test_classification_metrics() -> None:
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 1], classes=2)

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
