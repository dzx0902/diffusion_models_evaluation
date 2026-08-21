"""Compact EEG classifiers used by the subject-dependent screening benchmark."""

from __future__ import annotations

import torch
from torch import nn


class RawEEGClassifier(nn.Module):
    """Base class for classifiers receiving normalized ``[B, C, T]`` EEG."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.classes = classes


class Conv2DClassifier(RawEEGClassifier):
    def __init__(self, features: nn.Module, classes: int, channels: int, samples: int) -> None:
        super().__init__(classes)
        self.features = features
        with torch.no_grad():
            size = features(torch.zeros(1, 1, channels, samples)).numel()
        self.head = nn.Linear(size, classes)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        features = self.features(eeg.unsqueeze(1)).flatten(1)
        return self.head(features)


def build_shallow(classes: int, channels: int, samples: int) -> Conv2DClassifier:
    features = nn.Sequential(
        nn.Conv2d(1, 40, (1, 25)),
        nn.Conv2d(40, 40, (channels, 1), bias=False),
        nn.BatchNorm2d(40),
        nn.ELU(),
        nn.AvgPool2d((1, 51), (1, 5)),
        nn.Dropout(0.5),
    )
    return Conv2DClassifier(features, classes, channels, samples)


def build_deep(classes: int, channels: int, samples: int) -> Conv2DClassifier:
    features = nn.Sequential(
        nn.Conv2d(1, 25, (1, 10)),
        nn.Conv2d(25, 25, (channels, 1), bias=False),
        nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.5),
        nn.Conv2d(25, 50, (1, 10)),
        nn.BatchNorm2d(50), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.5),
        nn.Conv2d(50, 100, (1, 10)),
        nn.BatchNorm2d(100), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.5),
        nn.Conv2d(100, 200, (1, 10)),
        nn.BatchNorm2d(200), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.5),
    )
    return Conv2DClassifier(features, classes, channels, samples)


def build_eegnet(classes: int, channels: int, samples: int) -> Conv2DClassifier:
    f1, depth, f2 = 8, 2, 16
    features = nn.Sequential(
        nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
        nn.BatchNorm2d(f1),
        nn.Conv2d(f1, f1 * depth, (channels, 1), groups=f1, bias=False),
        nn.BatchNorm2d(f1 * depth), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
        nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8), groups=f1 * depth, bias=False),
        nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
        nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5),
    )
    return Conv2DClassifier(features, classes, channels, samples)


def build_tsconv(classes: int, channels: int, samples: int) -> Conv2DClassifier:
    features = nn.Sequential(
        nn.Conv2d(1, 40, (1, 25)),
        nn.AvgPool2d((1, 51), (1, 5)),
        nn.BatchNorm2d(40), nn.ELU(),
        nn.Conv2d(40, 40, (channels, 1), bias=False),
        nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.5),
    )
    return Conv2DClassifier(features, classes, channels, samples)


class BandpowerMLP(RawEEGClassifier):
    BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

    def __init__(self, classes: int, channels: int, samples: int, sampling_rate: int) -> None:
        super().__init__(classes)
        frequencies = torch.fft.rfftfreq(samples, d=1.0 / sampling_rate)
        masks = torch.stack([(frequencies >= low) & (frequencies < high) for low, high in self.BANDS])
        self.register_buffer("band_masks", masks, persistent=False)
        feature_dim = channels * len(self.BANDS)
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, classes),
        )

    def extract_features(self, eeg: torch.Tensor) -> torch.Tensor:
        power = torch.fft.rfft(eeg.float(), dim=-1).abs().square()
        features = [torch.log1p(power[..., mask].mean(dim=-1)) for mask in self.band_masks]
        return torch.stack(features, dim=-1).flatten(1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.network(self.extract_features(eeg))


class EEGConformer(RawEEGClassifier):
    def __init__(self, classes: int, channels: int, hidden: int = 40) -> None:
        super().__init__(classes)
        self.patch = nn.Sequential(
            nn.Conv2d(1, hidden, (1, 25)),
            nn.Conv2d(hidden, hidden, (channels, 1), bias=False),
            nn.BatchNorm2d(hidden), nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)), nn.Dropout(0.5),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=5,
            dim_feedforward=hidden * 4,
            dropout=0.5,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, classes))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(eeg.unsqueeze(1)).squeeze(2).transpose(1, 2)
        return self.head(self.encoder(tokens).mean(dim=1))


def build_eeg_classifier(
    name: str,
    classes: int,
    channels: int = 62,
    samples: int = 800,
    sampling_rate: int = 200,
) -> nn.Module:
    """Build one model under the common raw-EEG classification interface."""
    if name == "mlp":
        return BandpowerMLP(classes, channels, samples, sampling_rate)
    if name == "eegnet":
        return build_eegnet(classes, channels, samples)
    if name == "shallownet":
        return build_shallow(classes, channels, samples)
    if name == "deepnet":
        return build_deep(classes, channels, samples)
    if name == "tsconv":
        return build_tsconv(classes, channels, samples)
    if name == "conformer":
        return EEGConformer(classes, channels)
    raise ValueError(f"Unknown EEG classifier: {name!r}")
