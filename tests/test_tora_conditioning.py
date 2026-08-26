from __future__ import annotations

import unittest

import torch

from src.ms_video_eval.tora_conditioning import (
    ToraPCAProjector,
    ToraTextCondition,
    inject_tora_crossattn,
)


class ToraConditioningTest(unittest.TestCase):
    def test_pca_projector_round_trip_shape(self) -> None:
        components = torch.eye(4)[:2]
        projector = ToraPCAProjector(torch.zeros(4), components)
        state = torch.randn(3, 4)
        latent = projector.encode(state)
        restored = projector.decode(latent)
        self.assertEqual(tuple(latent.shape), (3, 2))
        self.assertEqual(tuple(restored.shape), (3, 4))

    def test_condition_validation_accepts_dynamic_test_dimension(self) -> None:
        condition = ToraTextCondition("01-001", "caption", torch.randn(226, 8))
        condition.validate(hidden_dim=8)

    def test_injection_changes_only_cross_attention(self) -> None:
        native = {"crossattn": torch.zeros(2, 226, 8), "vector": torch.ones(2, 4)}
        hidden = torch.ones(226, 8)
        injected = inject_tora_crossattn(native, hidden)
        self.assertTrue(torch.equal(injected["crossattn"], torch.ones(2, 226, 8)))
        self.assertIs(injected["vector"], native["vector"])
        self.assertTrue(torch.equal(native["crossattn"], torch.zeros(2, 226, 8)))

    def test_injection_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            inject_tora_crossattn({"crossattn": torch.zeros(1, 226, 8)}, torch.zeros(225, 8))


if __name__ == "__main__":
    unittest.main()
