from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_eeg_wan_conditioner import VideoGroupedBatchSampler, history_early_stop_state
from src.ms_video_eval.eeg_conditioner import (
    EEGConditioner,
    EEGConditionerConfig,
    add_condition_offset,
    fixed_pca_loss,
)


class EEGConditionerV2Test(unittest.TestCase):
    def test_condition_offset_reconstructs_absolute_target(self) -> None:
        residual = torch.randn(2, 7, 8)
        mean = torch.randn(7, 8)

        reconstructed = add_condition_offset(residual, mean)

        torch.testing.assert_close(reconstructed, residual + mean)

    def test_condition_offset_rejects_incompatible_shape(self) -> None:
        with self.assertRaises(ValueError):
            add_condition_offset(torch.randn(2, 7, 8), torch.randn(6, 8))

    def test_multiscale_condition_shape(self) -> None:
        config = EEGConditionerConfig(
            channels=4,
            sample_points=800,
            hidden_dim=32,
            slots=7,
            latent_dim=8,
            token_count=8,
            encoder_layers=1,
            decoder_layers=1,
            heads=8,
            min_tokens=7,
            max_tokens=7,
            architecture="multiscale",
            sampling_rate=200,
        )
        model = EEGConditioner(config).eval()

        with torch.inference_mode():
            latent, logits = model(torch.randn(2, 4, 800))

        self.assertEqual(tuple(latent.shape), (2, 7, 8))
        self.assertEqual(tuple(logits.shape), (2, 1))

    def test_fixed_cogvideox_token_range_is_supported(self) -> None:
        config = EEGConditionerConfig(
            hidden_dim=32,
            slots=226,
            latent_dim=16,
            encoder_layers=1,
            decoder_layers=1,
            heads=8,
            min_tokens=226,
            max_tokens=226,
        )

        model = EEGConditioner(config)

        self.assertEqual(model.length_head[-1].out_features, 1)

    def test_token_range_cannot_exceed_condition_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "must fit within 128 condition slots"):
            EEGConditioner(EEGConditionerConfig(slots=128, min_tokens=1, max_tokens=226))

    def test_multi_positive_contrastive_loss_is_finite(self) -> None:
        predicted = torch.randn(3, 7, 8)
        target = torch.randn(3, 7, 8)
        lengths = torch.full((3,), 7, dtype=torch.long)
        logits = torch.zeros(3, 1)
        positive_mask = torch.tensor(
            [[True, True, False], [True, True, False], [False, False, True]]
        )

        loss, values = fixed_pca_loss(
            predicted,
            target,
            lengths,
            logits,
            min_tokens=7,
            contrastive_weight=0.2,
            positive_mask=positive_mask,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(values["contrastive_loss"], 0.0)

    def test_grouped_sampler_keeps_sessions_together(self) -> None:
        rows = [
            {"video_id": video_id, "session": session}
            for video_id in ("01-001", "01-002", "01-003")
            for session in ("session1", "session2", "session3")
        ]
        sampler = VideoGroupedBatchSampler(rows, batch_size=6, shuffle=False)

        batches = list(sampler)

        self.assertEqual([len(batch) for batch in batches], [6, 3])
        for batch in batches:
            counts: dict[str, int] = {}
            for index in batch:
                counts[rows[index]["video_id"]] = counts.get(rows[index]["video_id"], 0) + 1
            self.assertTrue(all(count == 3 for count in counts.values()))

    def test_history_early_stopping_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            rows = [1.0, 0.8, 0.79, 0.795]
            path.write_text(
                "".join(json.dumps({"valid": {"loss": value}}) + "\n" for value in rows),
                encoding="utf-8",
            )

            best, stale = history_early_stop_state(path, "loss", min_delta=0.005)

        self.assertAlmostEqual(best, 0.79)
        self.assertEqual(stale, 1)


if __name__ == "__main__":
    unittest.main()
