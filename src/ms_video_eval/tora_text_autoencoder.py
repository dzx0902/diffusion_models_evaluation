"""Nonlinear token-wise semantic bottleneck for Tora T5 states."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ToraTextAutoencoderConfig:
    input_dim: int = 4096
    hidden_dim: int = 1024
    bottleneck_dim: int = 512
    dropout: float = 0.1


class ToraTextAutoencoder(nn.Module):
    def __init__(self, config: ToraTextAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(config.bottleneck_dim),
            nn.Linear(config.bottleneck_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(state)
        return {"latent": latent, "reconstruction": self.decode(latent)}

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(reconstruction, target)
    cosine = 1 - F.cosine_similarity(reconstruction, target, dim=-1).mean()
    return mse + cosine_weight * cosine, {
        "mse": float(mse.detach()), "cosine_loss": float(cosine.detach())
    }
