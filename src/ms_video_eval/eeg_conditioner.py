"""EEG encoder that predicts fixed text-conditioning states."""

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
    architecture: str = "baseline"
    sampling_rate: int = 200


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
    fixed ``[batch, slots, latent_dim]`` condition plus a discrete token-length
    prediction. Fixed-length CLIP targets use a one-class length head.
    """

    def __init__(self, config: EEGConditionerConfig = EEGConditionerConfig()) -> None:
        super().__init__()
        if config.architecture not in {"baseline", "multiscale"}:
            raise ValueError(f"Unknown EEG architecture: {config.architecture}")
        if not 1 <= config.min_tokens <= config.max_tokens <= config.slots:
            raise ValueError(
                f"Token range [{config.min_tokens}, {config.max_tokens}] "
                f"must fit within {config.slots} condition slots"
            )
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
        if config.architecture == "multiscale":
            self.frequency_projection = nn.Sequential(
                nn.LayerNorm(config.channels * 4),
                nn.Linear(config.channels * 4, config.hidden_dim),
                nn.GELU(),
            )
            self.frequency_type = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
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

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.config.channels:
            raise ValueError(f"Expected [batch, {self.config.channels}, time], got {tuple(eeg.shape)}")
        if eeg.shape[-1] != self.config.sample_points:
            eeg = F.interpolate(eeg, size=self.config.sample_points, mode="linear", align_corners=False)
        tokens = self.stem(eeg)
        tokens = F.adaptive_avg_pool1d(tokens, self.config.token_count).transpose(1, 2)
        if self.config.architecture == "multiscale":
            frequency_tokens = self._frequency_tokens(eeg) + self.frequency_type
            tokens = torch.cat([tokens, frequency_tokens], dim=1)
        return self.encoder(tokens)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode(eeg)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.decoder:
            queries = block(queries, tokens)
        latent = self.latent_head(queries)
        length_logits = self.length_head(tokens.mean(dim=1))
        return latent, length_logits

    def _frequency_tokens(self, eeg: torch.Tensor) -> torch.Tensor:
        window = self.config.sampling_rate
        window_count = eeg.shape[-1] // window
        if window_count < 1:
            raise ValueError(
                f"EEG length={eeg.shape[-1]} is shorter than one {window}-sample frequency window"
            )
        segmented = eeg[..., : window_count * window].reshape(
            eeg.shape[0],
            eeg.shape[1],
            window_count,
            window,
        )
        spectrum = torch.fft.rfft(segmented.float(), dim=-1).abs().pow(2)
        frequencies = torch.fft.rfftfreq(window, d=1.0 / self.config.sampling_rate).to(eeg.device)
        bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
        band_power = []
        for low, high in bands:
            mask = (frequencies >= low) & (frequencies < high)
            band_power.append(torch.log1p(spectrum[..., mask].mean(dim=-1)))
        features = torch.stack(band_power, dim=-1)
        features = features.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.frequency_projection(features)


class EEGCategoryProbe(nn.Module):
    """Lightweight category classifier over the shared EEG encoder."""

    def __init__(self, config: EEGConditionerConfig, classes: int) -> None:
        super().__init__()
        if classes < 2:
            raise ValueError("EEG category probe requires at least two classes")
        self.config = config
        self.classes = classes
        self.backbone = EEGConditioner(config)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, classes),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.encode(eeg).mean(dim=1))


def add_condition_offset(
    predicted: torch.Tensor,
    offset: torch.Tensor | None,
) -> torch.Tensor:
    """Reconstruct an absolute condition from a predicted residual."""
    if offset is None:
        return predicted
    if offset.ndim != 2 or tuple(offset.shape) != tuple(predicted.shape[-2:]):
        raise ValueError(
            f"Condition offset shape={tuple(offset.shape)} does not match "
            f"prediction tail={tuple(predicted.shape[-2:])}"
        )
    return predicted + offset.to(device=predicted.device, dtype=predicted.dtype)


def fixed_pca_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    length_logits: torch.Tensor,
    min_tokens: int,
    padding_weight: float = 0.1,
    length_weight: float = 0.2,
    pooled_weight: float = 0.1,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.07,
    positive_mask: torch.Tensor | None = None,
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
    contrastive_loss = predicted.new_zeros(())
    if contrastive_weight > 0:
        pred_norm = F.normalize(pooled_pred, dim=-1)
        target_norm = F.normalize(pooled_target, dim=-1)
        logits = pred_norm @ target_norm.t() / contrastive_temperature
        if positive_mask is None:
            positive_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        else:
            positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        contrastive_loss = -(
            torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
        ).mean()
    loss = (
        valid_mse
        + padding_weight * padding_mse
        + length_weight * length_loss
        + pooled_weight * pooled_cosine
        + contrastive_weight * contrastive_loss
    )
    return loss, {
        "loss": float(loss.detach()),
        "valid_mse": float(valid_mse.detach()),
        "padding_mse": float(padding_mse.detach()),
        "length_ce": float(length_loss.detach()),
        "pooled_cosine_loss": float(pooled_cosine.detach()),
        "contrastive_loss": float(contrastive_loss.detach()),
    }
