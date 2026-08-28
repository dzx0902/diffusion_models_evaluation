from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.tora_text_autoencoder import (
    ToraTextAutoencoder, ToraTextAutoencoderConfig, reconstruction_loss,
)


def test_autoencoder_shapes_and_loss() -> None:
    config = ToraTextAutoencoderConfig(input_dim=16, hidden_dim=8, bottleneck_dim=4, dropout=0)
    model = ToraTextAutoencoder(config)
    inputs = torch.randn(2, 5, 16)
    output = model(inputs)
    assert output["latent"].shape == (2, 5, 4)
    assert output["reconstruction"].shape == inputs.shape
    loss, values = reconstruction_loss(output["reconstruction"], inputs)
    assert torch.isfinite(loss)
    assert set(values) == {"mse", "cosine_loss"}
