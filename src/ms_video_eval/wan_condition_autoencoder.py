"""Fixed-slot sequence autoencoder for Wan T5 conditioning states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WanConditionAutoencoderConfig:
    slots: int = 128
    input_dim: int = 4096
    latent_dim: int = 1024
    hidden_dim: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 8
    dropout: float = 0.05


def valid_mask(lengths: torch.Tensor, slots: int) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"Expected lengths [batch], got {tuple(lengths.shape)}")
    if int(lengths.min()) < 1 or int(lengths.max()) > slots:
        raise ValueError(f"Lengths must be within 1..{slots}: {lengths.tolist()}")
    return torch.arange(slots, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class WanConditionAutoencoder(nn.Module):
    """Learn a continuous ``[slots, latent_dim]`` condition space for Wan."""

    def __init__(self, config: WanConditionAutoencoderConfig = WanConditionAutoencoderConfig()) -> None:
        super().__init__()
        if config.hidden_dim % config.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.encoder_position = nn.Parameter(torch.randn(config.slots, config.hidden_dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.encoder_layers,
        )
        self.to_latent = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.latent_dim))
        self.from_latent = nn.Sequential(nn.Linear(config.latent_dim, config.hidden_dim), nn.GELU())
        self.decoder_position = nn.Parameter(torch.randn(config.slots, config.hidden_dim) * 0.02)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.decoder_layers,
        )
        self.output_projection = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.input_dim))

    def _check(self, states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        expected = (states.shape[0], self.config.slots, self.config.input_dim)
        if states.ndim != 3 or tuple(states.shape) != expected:
            raise ValueError(f"Expected states {expected}, got {tuple(states.shape)}")
        return valid_mask(lengths, self.config.slots)

    def encode(self, states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = self._check(states, lengths)
        hidden = self.input_projection(states) + self.encoder_position.unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        latent = self.to_latent(hidden)
        return latent * mask.unsqueeze(-1).to(latent.dtype)

    def decode(self, latent: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        expected = (latent.shape[0], self.config.slots, self.config.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape) != expected:
            raise ValueError(f"Expected latent {expected}, got {tuple(latent.shape)}")
        mask = valid_mask(lengths, self.config.slots)
        hidden = self.from_latent(latent) + self.decoder_position.unsqueeze(0)
        hidden = self.decoder(hidden, src_key_padding_mask=~mask)
        states = self.output_projection(hidden)
        return states * mask.unsqueeze(-1).to(states.dtype)

    def forward(self, states: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(states, lengths)
        return latent, self.decode(latent, lengths)


def autoencoder_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    cosine_weight: float = 0.25,
    pooled_weight: float = 0.25,
    padding_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Measure valid-token reconstruction and force unused slots back to zero."""

    if reconstructed.shape != target.shape:
        raise ValueError(f"Mismatched tensors: {tuple(reconstructed.shape)} vs {tuple(target.shape)}")
    mask = valid_mask(lengths, reconstructed.shape[1])
    valid = mask.unsqueeze(-1).to(reconstructed.dtype)
    padded = (~mask).unsqueeze(-1).to(reconstructed.dtype)
    valid_mse = ((reconstructed - target).square() * valid).sum() / valid.sum().clamp_min(1) / reconstructed.shape[-1]
    padding_mse = (reconstructed.square() * padded).sum() / padded.sum().clamp_min(1) / reconstructed.shape[-1]
    token_cosine = 1 - F.cosine_similarity(reconstructed, target, dim=-1)
    token_cosine_loss = (token_cosine * mask).sum() / mask.sum().clamp_min(1)
    pooled_reconstructed = (reconstructed * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    pooled_target = (target * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    pooled_cosine_loss = 1 - F.cosine_similarity(pooled_reconstructed, pooled_target, dim=-1).mean()
    loss = valid_mse + cosine_weight * token_cosine_loss + pooled_weight * pooled_cosine_loss + padding_weight * padding_mse
    return loss, {
        "loss": float(loss.detach()),
        "valid_mse": float(valid_mse.detach()),
        "padding_mse": float(padding_mse.detach()),
        "token_cosine_loss": float(token_cosine_loss.detach()),
        "pooled_cosine_loss": float(pooled_cosine_loss.detach()),
    }
