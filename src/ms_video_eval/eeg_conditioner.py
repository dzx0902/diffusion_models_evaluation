"""EEG encoder that predicts fixed Wan PCA text conditions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EEGConditionerConfig:
    channels: int = 62
    sample_points: int = 600
    hidden_dim: int = 256
    slots: int = 128
    latent_dim: int = 512
    token_count: int = 75
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 8
    dropout: float = 0.15
    min_tokens: int = 63
    max_tokens: int = 94


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(queries, context, context, need_weights=False)
        queries = self.attn_norm(queries + attended)
        return self.ffn_norm(queries + self.ffn(queries))


class EEGConditioner(nn.Module):
    """Temporal-spatial EEG encoder with fixed learned queries.

    Input is per-trial normalized raw EEG ``[batch, 62, T]``. The output is a
    fixed ``[batch, 128, 512]`` PCA condition plus a discrete native-token
    length prediction. Wan receives only the first predicted number of slots.
    """

    def __init__(self, config: EEGConditionerConfig = EEGConditionerConfig()) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(config.channels, config.hidden_dim, kernel_size=51, padding=25, bias=False),
            nn.GroupNorm(16, config.hidden_dim),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=15, padding=7, groups=config.hidden_dim, bias=False),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, config.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.queries = nn.Parameter(torch.randn(config.slots, config.hidden_dim) * 0.02)
        self.decoder = nn.ModuleList(
            CrossAttentionBlock(config.hidden_dim, config.heads, config.dropout)
            for _ in range(config.decoder_layers)
        )
        self.latent_head = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.latent_dim))
        self.length_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.max_tokens - config.min_tokens + 1),
        )

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3 or eeg.shape[1] != self.config.channels:
            raise ValueError(f"Expected [batch, {self.config.channels}, time], got {tuple(eeg.shape)}")
        if eeg.shape[-1] != self.config.sample_points:
            eeg = F.interpolate(eeg, size=self.config.sample_points, mode="linear", align_corners=False)
        tokens = self.stem(eeg)
        tokens = F.adaptive_avg_pool1d(tokens, self.config.token_count).transpose(1, 2)
        tokens = self.encoder(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.decoder:
            queries = block(queries, tokens)
        latent = self.latent_head(queries)
        length_logits = self.length_head(tokens.mean(dim=1))
        return latent, length_logits


def fixed_pca_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    length_logits: torch.Tensor,
    min_tokens: int,
    padding_weight: float = 0.1,
    length_weight: float = 0.2,
    pooled_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked regression on valid slots, zero penalty on padding, and length CE."""
    slots = predicted.shape[1]
    valid = torch.arange(slots, device=predicted.device).unsqueeze(0) < lengths.unsqueeze(1)
    valid3 = valid.unsqueeze(-1).to(predicted.dtype)
    valid_mse = ((predicted - target).pow(2) * valid3).sum() / valid3.sum().clamp_min(1) / predicted.shape[-1]
    padded = (~valid).unsqueeze(-1).to(predicted.dtype)
    padding_mse = (predicted.pow(2) * padded).sum() / padded.sum().clamp_min(1) / predicted.shape[-1]
    pooled_pred = (predicted * valid3).sum(dim=1) / valid3.sum(dim=1).clamp_min(1)
    pooled_target = (target * valid3).sum(dim=1) / valid3.sum(dim=1).clamp_min(1)
    pooled_cosine = 1 - F.cosine_similarity(pooled_pred, pooled_target, dim=-1).mean()
    length_loss = F.cross_entropy(length_logits, lengths - min_tokens)
    loss = valid_mse + padding_weight * padding_mse + length_weight * length_loss + pooled_weight * pooled_cosine
    return loss, {
        "loss": float(loss.detach()),
        "valid_mse": float(valid_mse.detach()),
        "padding_mse": float(padding_mse.detach()),
        "length_ce": float(length_loss.detach()),
        "pooled_cosine_loss": float(pooled_cosine.detach()),
    }
