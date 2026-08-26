"""Shared EEG encoder and method-specific semantic prediction heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .eeg_conditioner import CrossAttentionBlock, EEGConditioner, EEGConditionerConfig


class SharedEEGEncoder(nn.Module):
    """Feature-only view of the established EEG conditioner backbone.

    The modules are initialized by :class:`EEGConditioner` to guarantee the
    exact same architecture as the continuous alignment baseline, then moved
    into this feature-only module.  Method A/B therefore do not carry unused
    condition queries or latent heads.
    """

    def __init__(self, config: EEGConditionerConfig) -> None:
        super().__init__()
        template = EEGConditioner(config)
        self.config = config
        self.stem = template.stem
        self.encoder = template.encoder
        if config.architecture == "multiscale":
            self.frequency_projection = template.frequency_projection
            self.frequency_type = template.frequency_type

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        if eeg.ndim != 3 or eeg.shape[1] != self.config.channels:
            raise ValueError(
                f"Expected [batch, {self.config.channels}, time], got {tuple(eeg.shape)}"
            )
        if eeg.shape[-1] != self.config.sample_points:
            eeg = F.interpolate(
                eeg, size=self.config.sample_points, mode="linear", align_corners=False
            )
        tokens = self.stem(eeg)
        tokens = F.adaptive_avg_pool1d(tokens, self.config.token_count).transpose(1, 2)
        if self.config.architecture == "multiscale":
            frequency_tokens = self._frequency_tokens(eeg) + self.frequency_type
            tokens = torch.cat([tokens, frequency_tokens], dim=1)
        tokens = self.encoder(tokens)
        return {"tokens": tokens, "pooled": tokens.mean(dim=1)}

    def _frequency_tokens(self, eeg: torch.Tensor) -> torch.Tensor:
        window = self.config.sampling_rate
        window_count = eeg.shape[-1] // window
        if window_count < 1:
            raise ValueError(
                f"EEG length={eeg.shape[-1]} is shorter than one {window}-sample frequency window"
            )
        segmented = eeg[..., : window_count * window].reshape(
            eeg.shape[0], eeg.shape[1], window_count, window
        )
        spectrum = torch.fft.rfft(segmented.float(), dim=-1).abs().pow(2)
        frequencies = torch.fft.rfftfreq(
            window, d=1.0 / self.config.sampling_rate
        ).to(eeg.device)
        band_power = []
        for low, high in ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)):
            mask = (frequencies >= low) & (frequencies < high)
            band_power.append(torch.log1p(spectrum[..., mask].mean(dim=-1)))
        features = torch.stack(band_power, dim=-1)
        features = features.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.frequency_projection(features)


class SemanticSlotModel(nn.Module):
    """Method A/B: independent multi-label heads over one shared EEG feature."""

    def __init__(
        self,
        encoder_config: EEGConditionerConfig,
        slot_classes: Mapping[str, int],
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if not slot_classes or any(value < 1 for value in slot_classes.values()):
            raise ValueError("Each semantic slot must define at least one class")
        self.encoder = SharedEEGEncoder(encoder_config)
        probability = encoder_config.dropout if dropout is None else float(dropout)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(encoder_config.hidden_dim),
            nn.Linear(encoder_config.hidden_dim, encoder_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(probability),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(encoder_config.hidden_dim, classes)
                for name, classes in slot_classes.items()
            }
        )

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        encoded = self.encoder(eeg)
        feature = self.shared_projection(encoded["pooled"])
        return {
            "feature": feature,
            "tokens": encoded["tokens"],
            "logits": {name: head(feature) for name, head in self.heads.items()},
        }


class DirectSemanticAlignmentModel(nn.Module):
    """Method C: shared encoder plus query decoder for a text condition target."""

    def __init__(
        self,
        config: EEGConditionerConfig,
        auxiliary_slot_classes: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = SharedEEGEncoder(config)
        self.queries = nn.Parameter(torch.randn(config.slots, config.hidden_dim) * 0.02)
        self.decoder = nn.ModuleList(
            CrossAttentionBlock(config.hidden_dim, config.heads, config.dropout)
            for _ in range(config.decoder_layers)
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.latent_dim)
        )
        self.length_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.max_tokens - config.min_tokens + 1),
        )
        self.auxiliary_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.hidden_dim, classes)
                for name, classes in (auxiliary_slot_classes or {}).items()
            }
        )

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(eeg)
        queries = self.queries.unsqueeze(0).expand(eeg.shape[0], -1, -1)
        for block in self.decoder:
            queries = block(queries, encoded["tokens"])
        return {
            "feature": encoded["pooled"],
            "latent": self.latent_head(queries),
            "length_logits": self.length_head(encoded["pooled"]),
            "auxiliary_logits": {
                name: head(encoded["pooled"]) for name, head in self.auxiliary_heads.items()
            },
        }


def semantic_slot_loss(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor] | None = None,
    slot_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked BCE for independently supervised multi-label semantic slots."""

    if set(logits) != set(targets):
        raise ValueError(f"Logit/target slots differ: {set(logits)} != {set(targets)}")
    slot_weights = slot_weights or {}
    losses: dict[str, torch.Tensor] = {}
    total = next(iter(logits.values())).new_zeros(())
    active_weight = 0.0
    for name, prediction in logits.items():
        target = targets[name].to(device=prediction.device, dtype=prediction.dtype)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Slot {name!r} prediction shape {tuple(prediction.shape)} "
                f"!= target {tuple(target.shape)}"
            )
        per_example = F.binary_cross_entropy_with_logits(
            prediction, target, reduction="none"
        ).mean(dim=-1)
        mask = None if target_masks is None else target_masks.get(name)
        if mask is not None:
            mask = mask.to(device=prediction.device, dtype=prediction.dtype)
            value = (per_example * mask).sum() / mask.sum().clamp_min(1)
        else:
            value = per_example.mean()
        weight = float(slot_weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("Semantic slot weights must be non-negative")
        losses[name] = value
        total = total + weight * value
        active_weight += weight
    total = total / max(active_weight, 1e-12)
    return total, {name: float(value.detach()) for name, value in losses.items()}


def cross_session_consistency_loss(
    features: torch.Tensor,
    video_ids: Sequence[str],
) -> torch.Tensor:
    """Mean cosine distance across distinct trials of the same video."""

    if features.ndim != 2 or features.shape[0] != len(video_ids):
        raise ValueError("features/video_ids shape mismatch")
    normalized = F.normalize(features, dim=-1)
    similarities = normalized @ normalized.t()
    mask = torch.tensor(
        [
            [left == right and i != j for j, right in enumerate(video_ids)]
            for i, left in enumerate(video_ids)
        ],
        device=features.device,
        dtype=torch.bool,
    )
    if not mask.any():
        return features.new_zeros(())
    return (1.0 - similarities[mask]).mean()


def semantic_soft_contrastive_loss(
    eeg_features: torch.Tensor,
    text_features: torch.Tensor,
    target_similarity: torch.Tensor | None = None,
    temperature: float = 0.07,
    semantic_temperature: float = 0.07,
) -> torch.Tensor:
    """Hard InfoNCE or soft-target EEG-to-text contrastive alignment."""

    if temperature <= 0 or semantic_temperature <= 0:
        raise ValueError("Contrastive temperatures must be positive")
    if eeg_features.shape != text_features.shape or eeg_features.ndim != 2:
        raise ValueError("EEG/text features must have matching [batch, dim] shapes")
    logits = F.normalize(eeg_features, dim=-1) @ F.normalize(text_features, dim=-1).t()
    logits = logits / temperature
    if target_similarity is None:
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)
    if target_similarity.shape != logits.shape:
        raise ValueError("target_similarity must be [batch, batch]")
    target = F.softmax(target_similarity.to(logits.device) / semantic_temperature, dim=-1)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def full_text_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Full text states must have matching [batch, tokens, hidden] shapes")
    if mse_weight < 0 or cosine_weight < 0:
        raise ValueError("Alignment weights must be non-negative")
    mse = F.mse_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
    loss = mse_weight * mse + cosine_weight * cosine
    return loss, {"mse": float(mse.detach()), "cosine_loss": float(cosine.detach())}


def multi_positive_contrastive_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric multi-positive InfoNCE over pooled semantic states."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("Contrastive features must have matching [batch, dim] shapes")
    if positive_mask.shape != (predicted.shape[0], predicted.shape[0]):
        raise ValueError("positive_mask must be [batch, batch]")
    if not positive_mask.bool().any(dim=1).all():
        raise ValueError("Every contrastive row needs at least one positive")
    logits = F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).t()
    logits = logits / temperature
    mask = positive_mask.to(device=logits.device, dtype=torch.bool)

    def direction(values: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        numerator = torch.logsumexp(values.masked_fill(~positives, float("-inf")), dim=1)
        denominator = torch.logsumexp(values, dim=1)
        return -(numerator - denominator).mean()

    return 0.5 * (direction(logits, mask) + direction(logits.t(), mask.t()))
