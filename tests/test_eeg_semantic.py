from __future__ import annotations

import unittest

import torch

from src.ms_video_eval.eeg_conditioner import EEGConditionerConfig
from src.ms_video_eval.eeg_semantic import (
    DirectSemanticAlignmentModel,
    SemanticSlotModel,
    cross_session_consistency_loss,
    full_text_alignment_loss,
    multi_positive_contrastive_loss,
    semantic_slot_loss,
    semantic_soft_contrastive_loss,
)


def tiny_config() -> EEGConditionerConfig:
    return EEGConditionerConfig(
        channels=4,
        sample_points=64,
        hidden_dim=32,
        slots=7,
        latent_dim=8,
        token_count=8,
        encoder_layers=1,
        decoder_layers=1,
        heads=8,
        min_tokens=5,
        max_tokens=7,
    )


class EEGSemanticTest(unittest.TestCase):
    def test_slot_model_shapes_and_loss(self) -> None:
        model = SemanticSlotModel(tiny_config(), {"subject": 3, "object": 4})
        result = model(torch.randn(2, 4, 64))
        self.assertEqual(tuple(result["feature"].shape), (2, 32))
        self.assertEqual(tuple(result["logits"]["subject"].shape), (2, 3))
        targets = {"subject": torch.zeros(2, 3), "object": torch.zeros(2, 4)}
        loss, values = semantic_slot_loss(result["logits"], targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(values), {"subject", "object"})

    def test_direct_model_shape(self) -> None:
        model = DirectSemanticAlignmentModel(tiny_config(), {"subject": 3}).eval()
        with torch.inference_mode():
            result = model(torch.randn(2, 4, 64))
        self.assertEqual(tuple(result["latent"].shape), (2, 7, 8))
        self.assertEqual(tuple(result["length_logits"].shape), (2, 3))
        self.assertEqual(tuple(result["auxiliary_logits"]["subject"].shape), (2, 3))

    def test_full_alignment_and_multi_positive_losses(self) -> None:
        predicted = torch.randn(3, 7, 8)
        target = torch.randn(3, 7, 8)
        alignment, values = full_text_alignment_loss(predicted, target)
        self.assertTrue(torch.isfinite(alignment))
        self.assertEqual(set(values), {"mse", "cosine_loss"})
        mask = torch.tensor(
            [[True, True, False], [True, True, False], [False, False, True]]
        )
        contrastive = multi_positive_contrastive_loss(
            predicted.mean(1), target.mean(1), mask
        )
        self.assertTrue(torch.isfinite(contrastive))

    def test_cross_session_consistency_uses_same_video_only(self) -> None:
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        loss = cross_session_consistency_loss(features, ["a", "a", "b"])
        self.assertAlmostEqual(float(loss), 0.0)
        no_pairs = cross_session_consistency_loss(features, ["a", "b", "c"])
        self.assertEqual(float(no_pairs), 0.0)

    def test_soft_contrastive_is_finite(self) -> None:
        eeg = torch.randn(4, 8)
        text = torch.randn(4, 8)
        similarity = torch.eye(4)
        loss = semantic_soft_contrastive_loss(eeg, text, similarity)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
