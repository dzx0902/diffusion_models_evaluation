from __future__ import annotations

import torch
from torch import nn
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_tricks import (
    EEGAugmentationConfig,
    augment_eeg,
    classification_prototype_loss,
    curriculum_multiplier,
    hierarchical_positive_weights,
    semantic_similarity_matrix,
    video_positive_weights,
    weighted_positive_contrastive_loss,
)


def test_augmentation_preserves_shape_and_can_be_disabled() -> None:
    eeg = torch.randn(4, 62, 800)
    assert torch.equal(augment_eeg(eeg, EEGAugmentationConfig()), eeg)
    config = EEGAugmentationConfig(
        gaussian_noise_std=0.01,
        temporal_mask_probability=1.0,
        temporal_mask_fraction=0.1,
        channel_dropout_probability=0.1,
        amplitude_scale_min=0.9,
        amplitude_scale_max=1.1,
        temporal_shift_samples=5,
    )
    assert augment_eeg(eeg, config).shape == eeg.shape


def test_curriculum_interpolates_endpoints() -> None:
    schedule = {"enabled": True, "classification": {"start": 2.0, "end": 0.5}}
    assert curriculum_multiplier(1, 5, schedule, "classification") == 2.0
    assert curriculum_multiplier(5, 5, schedule, "classification") == 0.5


def test_semantic_similarity_and_video_weights() -> None:
    targets = {"action": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])}
    masks = {"action": torch.ones(3)}
    similarity = semantic_similarity_matrix(targets, masks)
    assert similarity[0, 1] == 1
    assert similarity[0, 2] == 0
    weights = video_positive_weights(["a", "a", "b"], similarity)
    assert weights[0, 1] == 1
    assert weights[0, 2] == 0


def test_weighted_contrastive_and_prototype_losses_are_finite() -> None:
    left = torch.randn(3, 8)
    right = torch.randn(3, 8)
    weights = torch.eye(3)
    assert torch.isfinite(weighted_positive_contrastive_loss(left, right, weights))
    heads = nn.ModuleDict({"action": nn.Linear(8, 2)})
    targets = {"action": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])}
    masks = {"action": torch.ones(3)}
    loss = classification_prototype_loss(left, heads, targets, masks)
    assert torch.isfinite(loss)


def test_hierarchical_positive_tiers() -> None:
    targets = {
        "coarse_action": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        "object": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    }
    masks = {key: torch.ones(3) for key in targets}
    weights = hierarchical_positive_weights(["a", "b", "c"], targets, masks)
    assert weights[0, 1] == 0.8
    assert weights[0, 2] == 0.4
