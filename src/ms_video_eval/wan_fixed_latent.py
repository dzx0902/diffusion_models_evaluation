"""Fixed-shape latent resampler for Wan native T5 conditioning states.

The module maps a variable-length ``[N, 4096]`` T5 state to a fixed ``[L, d]``
latent and decodes it back to ``[N, 4096]``. Slot prefixes are nested: using
``L=72`` includes exactly the first 68 slots used by ``L=68``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class FixedLatentConfig:
    input_dim: int = 4096
    latent_dim: int = 512
    max_slots: int = 128
    heads: int = 8


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    encoded = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(positions * frequencies)
    encoded[:, 1::2] = torch.cos(positions * frequencies[: encoded[:, 1::2].shape[1]])
    return encoded


class FixedLatentAutoencoder(nn.Module):
    """Query-resampler with a positional decoder for Wan T5 hidden states."""

    def __init__(self, config: FixedLatentConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_proj = nn.Linear(config.input_dim, config.latent_dim)
        self.slot_queries = nn.Parameter(torch.randn(config.max_slots, config.latent_dim) * 0.02)
        self.encode_attn = nn.MultiheadAttention(
            config.latent_dim, config.heads, batch_first=True
        )
        self.encode_norm = nn.LayerNorm(config.latent_dim)
        self.encode_ffn = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 4),
            nn.GELU(),
            nn.Linear(config.latent_dim * 4, config.latent_dim),
        )
        self.decode_base = nn.Parameter(torch.randn(1, 1, config.latent_dim) * 0.02)
        self.decode_attn = nn.MultiheadAttention(
            config.latent_dim, config.heads, batch_first=True
        )
        self.decode_norm = nn.LayerNorm(config.latent_dim)
        self.decode_ffn = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 4),
            nn.GELU(),
            nn.Linear(config.latent_dim * 4, config.latent_dim),
        )
        self.output_proj = nn.Linear(config.latent_dim, config.input_dim)

    def encode(self, hidden: torch.Tensor, slots: int) -> torch.Tensor:
        if not 1 <= slots <= self.config.max_slots:
            raise ValueError(f"slots must be in [1, {self.config.max_slots}], got {slots}")
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected [batch, tokens, {self.config.input_dim}], got {tuple(hidden.shape)}")
        tokens = self.input_proj(self.input_norm(hidden.float()))
        positions = sinusoidal_positions(tokens.shape[1], self.config.latent_dim, tokens.device)
        tokens = tokens + positions.unsqueeze(0).to(tokens.dtype)
        queries = self.slot_queries[:slots].unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.encode_attn(queries, tokens, tokens, need_weights=False)
        latent = self.encode_norm(queries + attended)
        return self.encode_norm(latent + self.encode_ffn(latent))

    def decode(self, latent: torch.Tensor, output_tokens: int) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError(f"Expected [batch, slots, {self.config.latent_dim}], got {tuple(latent.shape)}")
        positions = sinusoidal_positions(output_tokens, self.config.latent_dim, latent.device)
        queries = self.decode_base.expand(latent.shape[0], output_tokens, -1)
        queries = queries + positions.unsqueeze(0).to(latent.dtype)
        attended, _ = self.decode_attn(queries, latent, latent, need_weights=False)
        decoded = self.decode_norm(queries + attended)
        decoded = self.decode_norm(decoded + self.decode_ffn(decoded))
        return self.output_proj(decoded)

    def forward(self, hidden: torch.Tensor, slots: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(hidden, slots)
        return self.decode(latent, hidden.shape[1]), latent


def reconstruction_stats(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    recon = reconstructed.float()
    mse = float((recon - ref).pow(2).mean().detach().cpu().item())
    cosine = torch.nn.functional.cosine_similarity(ref, recon, dim=-1).mean()
    return {"mse": mse, "cosine": float(cosine.detach().cpu().item())}


def save_fixed_latent_checkpoint(
    path: Path,
    model: FixedLatentAutoencoder,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_fixed_latent_checkpoint(path: Path, device: torch.device) -> FixedLatentAutoencoder:
    payload = torch.load(path, map_location=device)
    model = FixedLatentAutoencoder(FixedLatentConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval().requires_grad_(False)
