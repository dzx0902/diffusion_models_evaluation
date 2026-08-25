"""Low-dimensional EEG encoder for centered text-condition retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EEGPooledRetrieverConfig:
    channels: int = 62
    sample_points: int = 800
    sampling_rate: int = 200
    hidden_dim: int = 256
    target_dim: int = 768
    token_count: int = 75
    encoder_layers: int = 2
    heads: int = 8
    dropout: float = 0.15
    architecture: str = "multiscale"


class EEGPooledRetriever(nn.Module):
    """Encode one EEG trial into a centered pooled text representation."""

    def __init__(self, config: EEGPooledRetrieverConfig) -> None:
        super().__init__()
        if config.architecture not in {"baseline", "multiscale"}:
            raise ValueError(f"Unknown EEG architecture: {config.architecture}")
        if config.hidden_dim % config.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if config.hidden_dim % 16:
            raise ValueError("hidden_dim must be divisible by 16 for GroupNorm")
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.channels,
                config.hidden_dim,
                kernel_size=51,
                padding=25,
                bias=False,
            ),
            nn.GroupNorm(16, config.hidden_dim),
            nn.GELU(),
            nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=15,
                padding=7,
                groups=config.hidden_dim,
                bias=False,
            ),
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
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.target_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.config.channels:
            raise ValueError(
                f"Expected [batch, {self.config.channels}, time], got {tuple(eeg.shape)}"
            )
        if eeg.shape[-1] != self.config.sample_points:
            eeg = F.interpolate(
                eeg,
                size=self.config.sample_points,
                mode="linear",
                align_corners=False,
            )
        tokens = self.stem(eeg)
        tokens = F.adaptive_avg_pool1d(tokens, self.config.token_count).transpose(1, 2)
        if self.config.architecture == "multiscale":
            tokens = torch.cat([tokens, self._frequency_tokens(eeg) + self.frequency_type], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))

    def _frequency_tokens(self, eeg: torch.Tensor) -> torch.Tensor:
        window = self.config.sampling_rate
        window_count = eeg.shape[-1] // window
        if window_count < 1:
            raise ValueError(
                f"EEG length={eeg.shape[-1]} is shorter than one {window}-sample window"
            )
        segmented = eeg[..., : window_count * window].reshape(
            eeg.shape[0],
            eeg.shape[1],
            window_count,
            window,
        )
        spectrum = torch.fft.rfft(segmented.float(), dim=-1).abs().pow(2)
        frequencies = torch.fft.rfftfreq(
            window,
            d=1.0 / self.config.sampling_rate,
        ).to(eeg.device)
        bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
        band_power = []
        for low, high in bands:
            mask = (frequencies >= low) & (frequencies < high)
            band_power.append(torch.log1p(spectrum[..., mask].mean(dim=-1)))
        features = torch.stack(band_power, dim=-1)
        features = features.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.frequency_projection(features)


def positive_mask(labels: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in labels] for left in labels],
        dtype=torch.bool,
        device=device,
    )


def multi_positive_contrastive_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = predicted @ target.t() / temperature
    positives = positives.to(device=logits.device, dtype=torch.bool)
    positive_logits = logits.masked_fill(~positives, float("-inf"))
    forward = -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    reverse_logits = logits.t()
    reverse_positive_logits = reverse_logits.masked_fill(~positives.t(), float("-inf"))
    reverse = -(
        torch.logsumexp(reverse_positive_logits, dim=1)
        - torch.logsumexp(reverse_logits, dim=1)
    ).mean()
    return 0.5 * (forward + reverse)


def full_bank_contrastive_loss(
    predicted: torch.Tensor,
    candidates: torch.Tensor,
    true_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Classify each prediction against every unique prompt in a caption bank."""
    if candidates.ndim != 2 or candidates.shape[1] != predicted.shape[1]:
        raise ValueError(
            "Candidate bank must have shape [prompts, target_dim], got "
            f"{tuple(candidates.shape)} for predictions {tuple(predicted.shape)}"
        )
    if true_indices.shape != (predicted.shape[0],):
        raise ValueError(
            f"Expected {predicted.shape[0]} true indices, got {tuple(true_indices.shape)}"
        )
    logits = (
        F.normalize(predicted, dim=-1)
        @ F.normalize(candidates, dim=-1).t()
        / temperature
    )
    return F.cross_entropy(logits, true_indices.to(logits.device, dtype=torch.long))


def variance_loss(
    predicted: torch.Tensor,
    target_std: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    if predicted.shape[0] < 2:
        return predicted.new_zeros(())
    standard_deviation = torch.sqrt(predicted.var(dim=0, unbiased=False) + 1e-4)
    expected = torch.as_tensor(
        target_std,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    return F.relu(expected - standard_deviation).mean()


def covariance_loss(predicted: torch.Tensor) -> torch.Tensor:
    if predicted.shape[0] < 2:
        return predicted.new_zeros(())
    centered = predicted - predicted.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / (predicted.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    return off_diagonal.square().sum() / predicted.shape[1]


def pooled_retrieval_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    positives: torch.Tensor,
    *,
    temperature: float = 0.07,
    mse_weight: float = 0.1,
    cosine_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    variance_weight: float = 0.05,
    covariance_weight: float = 0.005,
    contrastive_candidates: torch.Tensor | None = None,
    contrastive_true_indices: torch.Tensor | None = None,
    variance_target_std: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(predicted, target)
    cosine = 1 - F.cosine_similarity(predicted, target, dim=-1).mean()
    if contrastive_candidates is None:
        contrastive = multi_positive_contrastive_loss(
            predicted,
            target,
            positives,
            temperature,
        )
    else:
        if contrastive_true_indices is None:
            raise ValueError("Full-bank contrastive loss requires true prompt indices")
        contrastive = full_bank_contrastive_loss(
            predicted,
            contrastive_candidates,
            contrastive_true_indices,
            temperature,
        )
    variance = variance_loss(predicted, variance_target_std)
    covariance = covariance_loss(predicted)
    loss = (
        mse_weight * mse
        + cosine_weight * cosine
        + contrastive_weight * contrastive
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return loss, {
        "loss": float(loss.detach()),
        "mse": float(mse.detach()),
        "cosine_loss": float(cosine.detach()),
        "contrastive_loss": float(contrastive.detach()),
        "variance_loss": float(variance.detach()),
        "covariance_loss": float(covariance.detach()),
    }


def retrieval_ranks(
    predicted: torch.Tensor,
    candidates: torch.Tensor,
    true_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    similarities = F.normalize(predicted, dim=-1) @ F.normalize(candidates, dim=-1).t()
    true_similarities = similarities.gather(1, true_indices[:, None])
    greater = (similarities > true_similarities).sum(dim=1).float()
    ties = torch.isclose(similarities, true_similarities, atol=1e-7, rtol=1e-6).sum(dim=1)
    ranks = 1.0 + greater + 0.5 * (ties.float() - 1.0)
    return ranks, similarities


def grouped_retrieval_metrics(
    predicted: torch.Tensor,
    candidates: torch.Tensor,
    true_indices: torch.Tensor,
    group_ids: list[str],
) -> dict[str, float]:
    """Average repeated observations before evaluating prompt retrieval."""
    if len(group_ids) != predicted.shape[0]:
        raise ValueError("group_ids must match the prediction batch")
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    grouped_predictions = []
    grouped_indices = []
    for indices in groups.values():
        labels = true_indices[indices]
        if not torch.equal(labels, labels[:1].expand_as(labels)):
            raise ValueError("All observations in a group must share one target")
        grouped_predictions.append(predicted[indices].mean(dim=0))
        grouped_indices.append(labels[0])
    averaged = torch.stack(grouped_predictions)
    labels = torch.stack(grouped_indices)
    ranks, _ = retrieval_ranks(averaged, candidates, labels)
    exact = candidates[labels]
    cosine = F.cosine_similarity(averaged, exact, dim=-1)
    energy = averaged.square().mean(dim=-1) / exact.square().mean(dim=-1).clamp_min(1e-12)
    return {
        "count": len(groups),
        "recall_at_1": float((ranks <= 1).float().mean().item()),
        "recall_at_5": float((ranks <= 5).float().mean().item()),
        "mrr": float((1.0 / ranks).mean().item()),
        "mean_rank": float(ranks.mean().item()),
        "mean_cosine": float(cosine.mean().item()),
        "mean_energy_ratio": float(energy.mean().item()),
    }
