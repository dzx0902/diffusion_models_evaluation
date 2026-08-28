"""Config-driven training tricks shared by EEG semantic ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EEGAugmentationConfig:
    gaussian_noise_std: float = 0.0
    temporal_mask_probability: float = 0.0
    temporal_mask_fraction: float = 0.0
    channel_dropout_probability: float = 0.0
    amplitude_scale_min: float = 1.0
    amplitude_scale_max: float = 1.0
    temporal_shift_samples: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "EEGAugmentationConfig":
        values = values or {}
        config = cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})
        if config.gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std must be non-negative")
        for name in ("temporal_mask_probability", "temporal_mask_fraction", "channel_dropout_probability"):
            if not 0 <= getattr(config, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if config.amplitude_scale_min <= 0 or config.amplitude_scale_max < config.amplitude_scale_min:
            raise ValueError("Invalid amplitude scaling range")
        if config.temporal_shift_samples < 0:
            raise ValueError("temporal_shift_samples must be non-negative")
        return config


def augment_eeg(eeg: torch.Tensor, config: EEGAugmentationConfig) -> torch.Tensor:
    """Apply weak, per-example EEG augmentation without changing semantic labels."""

    if eeg.ndim != 3:
        raise ValueError("EEG augmentation expects [batch, channels, time]")
    result = eeg
    batch, channels, samples = result.shape
    if config.amplitude_scale_min != 1.0 or config.amplitude_scale_max != 1.0:
        scale = torch.empty(batch, 1, 1, device=eeg.device).uniform_(
            config.amplitude_scale_min, config.amplitude_scale_max
        )
        result = result * scale
    if config.gaussian_noise_std:
        result = result + torch.randn_like(result) * config.gaussian_noise_std
    if config.channel_dropout_probability:
        keep = torch.rand(batch, channels, 1, device=eeg.device) >= config.channel_dropout_probability
        result = result * keep.to(result.dtype)
    if config.temporal_mask_probability and config.temporal_mask_fraction:
        width = min(samples, max(1, round(samples * config.temporal_mask_fraction)))
        apply = torch.rand(batch, device=eeg.device) < config.temporal_mask_probability
        starts = torch.randint(0, samples - width + 1, (batch,), device=eeg.device)
        time = torch.arange(samples, device=eeg.device).unsqueeze(0)
        mask = apply.unsqueeze(1) & (time >= starts.unsqueeze(1)) & (time < (starts + width).unsqueeze(1))
        result = result.masked_fill(mask.unsqueeze(1), 0.0)
    if config.temporal_shift_samples:
        shifts = torch.randint(
            -config.temporal_shift_samples,
            config.temporal_shift_samples + 1,
            (batch,),
            device=eeg.device,
        )
        result = torch.stack([torch.roll(item, int(shift), dims=-1) for item, shift in zip(result, shifts)])
    return result


def curriculum_multiplier(
    epoch: int,
    epochs: int,
    schedule: Mapping[str, Any] | None,
    component: str,
) -> float:
    """Piecewise-linear multiplier between configured start/end values."""

    if not schedule or not bool(schedule.get("enabled", False)):
        return 1.0
    values = schedule.get(component, {})
    start = float(values.get("start", 1.0))
    end = float(values.get("end", start))
    if start < 0 or end < 0:
        raise ValueError("Curriculum multipliers must be non-negative")
    progress = 1.0 if epochs <= 1 else (epoch - 1) / (epochs - 1)
    return start + (end - start) * min(1.0, max(0.0, progress))


def semantic_similarity_matrix(
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    slot_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Weighted Jaccard similarity between structured multi-label targets."""

    if not targets:
        raise ValueError("At least one semantic target slot is required")
    weights = slot_weights or {}
    first = next(iter(targets.values()))
    result = first.new_zeros((first.shape[0], first.shape[0]))
    normalizer = first.new_zeros((first.shape[0], first.shape[0]))
    for slot, values in targets.items():
        values = values.float()
        intersection = torch.minimum(values[:, None, :], values[None, :, :]).sum(dim=-1)
        union = torch.maximum(values[:, None, :], values[None, :, :]).sum(dim=-1).clamp_min(1)
        valid = masks[slot].float()
        pair_valid = valid[:, None] * valid[None, :]
        weight = float(weights.get(slot, 1.0))
        result = result + weight * pair_valid * (intersection / union)
        normalizer = normalizer + weight * pair_valid
    result = result / normalizer.clamp_min(1e-12)
    return result.fill_diagonal_(1.0)


def weighted_positive_contrastive_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    positive_weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric contrastive cross entropy with normalized weighted positives."""

    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Contrastive inputs must have matching [batch, dim] shapes")
    if positive_weights.shape != (left.shape[0], left.shape[0]):
        raise ValueError("positive_weights must be [batch, batch]")
    if temperature <= 0 or (positive_weights < 0).any():
        raise ValueError("Invalid contrastive temperature or positive weight")
    if not (positive_weights.sum(dim=1) > 0).all():
        raise ValueError("Every row must contain a positive")
    logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).t() / temperature

    def direction(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        target = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return -(target * F.log_softmax(values, dim=1)).sum(dim=1).mean()

    weights = positive_weights.to(device=logits.device, dtype=logits.dtype)
    return 0.5 * (direction(logits, weights) + direction(logits.t(), weights.t()))


def classification_prototype_loss(
    features: torch.Tensor,
    heads: nn.ModuleDict,
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    slot_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Use classifier rows as trainable semantic class prototypes."""

    weights = slot_weights or {}
    total = features.new_zeros(())
    normalizer = 0.0
    normalized_features = F.normalize(features, dim=-1)
    for slot, head in heads.items():
        if not isinstance(head, nn.Linear):
            raise TypeError("Prototype regularization requires linear semantic heads")
        labels = targets[slot].to(features.device)
        valid = masks[slot].to(features.device).bool()
        if not valid.any():
            continue
        class_centers = F.normalize(head.weight, dim=-1)
        target_center = labels @ class_centers
        target_center = target_center / labels.sum(dim=-1, keepdim=True).clamp_min(1)
        value = (1 - (normalized_features * F.normalize(target_center, dim=-1)).sum(dim=-1))[valid].mean()
        weight = float(weights.get(slot, 1.0))
        total = total + weight * value
        normalizer += weight
    return total / max(normalizer, 1e-12)


def fixed_prototype_loss(
    features: torch.Tensor,
    prototypes: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    slot_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Cosine loss to fixed train-only target-space semantic prototypes."""

    weights = slot_weights or {}
    total = features.new_zeros(())
    normalizer = 0.0
    for slot, centers in prototypes.items():
        labels = targets[slot].to(features.device)
        valid = masks[slot].to(features.device).bool()
        centers = centers.to(features.device)
        available = centers.norm(dim=-1) > 0
        labels = labels * available.to(labels.dtype)
        valid = valid & (labels.sum(dim=-1) > 0)
        if not valid.any():
            continue
        target_center = labels @ centers
        target_center = target_center / labels.sum(dim=-1, keepdim=True).clamp_min(1)
        value = 1 - F.cosine_similarity(features[valid], target_center[valid], dim=-1).mean()
        weight = float(weights.get(slot, 1.0))
        total = total + weight * value
        normalizer += weight
    return total / max(normalizer, 1e-12)


def video_positive_weights(
    video_ids: Sequence[str],
    semantic_similarity: torch.Tensor,
    same_video: float = 1.0,
    semantic_scale: float = 0.8,
) -> torch.Tensor:
    """Combine exact-video positives with graded structured-semantic positives."""

    if same_video <= 0 or semantic_scale < 0:
        raise ValueError("Positive weights must be non-negative and same_video must be positive")
    weights = semantic_similarity * semantic_scale
    exact = torch.tensor(
        [[left == right for right in video_ids] for left in video_ids],
        device=semantic_similarity.device,
        dtype=torch.bool,
    )
    weights = weights.masked_fill(exact, same_video)
    return weights.fill_diagonal_(same_video)


def hierarchical_positive_weights(
    video_ids: Sequence[str],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Build same-video/full/action+object/action positive tiers."""

    values = {"same_video": 1.0, "same_full_semantics": 0.8,
              "same_action_object": 0.6, "same_action": 0.4, **(weights or {})}
    if any(float(value) < 0 for value in values.values()):
        raise ValueError("Hierarchical positive weights must be non-negative")
    first = next(iter(targets.values()))
    batch = first.shape[0]
    result = first.new_zeros((batch, batch))

    def overlap(slot: str) -> torch.Tensor:
        if slot not in targets:
            return torch.zeros(batch, batch, dtype=torch.bool, device=first.device)
        target = targets[slot].to(first.device).bool()
        valid = masks[slot].to(first.device).bool()
        return (target.float() @ target.float().t() > 0) & valid[:, None] & valid[None, :]

    action_slot = "fine_action" if "fine_action" in targets else "coarse_action"
    action = overlap(action_slot)
    objects = overlap("object")
    result = torch.where(action, result.new_tensor(float(values["same_action"])), result)
    result = torch.where(
        action & objects, result.new_tensor(float(values["same_action_object"])), result
    )
    full = torch.ones(batch, batch, dtype=torch.bool, device=first.device)
    for slot, target in targets.items():
        target = target.to(first.device)
        valid = masks[slot].to(first.device).bool()
        full &= (target[:, None, :] == target[None, :, :]).all(dim=-1) & valid[:, None] & valid[None, :]
    result = torch.where(full, result.new_tensor(float(values["same_full_semantics"])), result)
    same_video_mask = torch.tensor(
        [[left == right for right in video_ids] for left in video_ids],
        device=first.device, dtype=torch.bool,
    )
    result = torch.where(
        same_video_mask, result.new_tensor(float(values["same_video"])), result
    )
    return result.fill_diagonal_(float(values["same_video"]))
