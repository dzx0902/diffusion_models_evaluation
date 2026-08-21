from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_classifiers import BandpowerMLP, build_eeg_classifier


@pytest.mark.parametrize(
    "name",
    ["mlp", "eegnet", "shallownet", "deepnet", "tsconv", "conformer"],
)
def test_classifiers_share_raw_eeg_interface(name: str) -> None:
    model = build_eeg_classifier(name, classes=6, channels=62, samples=800)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 62, 800))
    assert output.shape == (2, 6)
    assert torch.isfinite(output).all()


def test_bandpower_mlp_uses_five_bands_per_channel() -> None:
    model = BandpowerMLP(classes=6, channels=62, samples=800, sampling_rate=200)
    features = model.extract_features(torch.randn(2, 62, 800))
    assert features.shape == (2, 310)


def test_unknown_classifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown EEG classifier"):
        build_eeg_classifier("missing", classes=6)
