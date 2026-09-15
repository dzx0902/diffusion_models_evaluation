from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_compact_tora_alignment import alignment_losses


def test_pooled_stage_ignores_zero_mean_token_residual() -> None:
    latent = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]]
    )
    output = {
        "latent": latent,
        "features": torch.zeros(1, 3, 8),
        "session_object_logits": torch.zeros(1, 3, 6),
        "fused_object_logits": torch.zeros(1, 6),
        "session_pair_logits": torch.zeros(1, 3, 8),
        "fused_pair_logits": torch.zeros(1, 8),
    }
    target = torch.zeros_like(latent)
    weights = {
        "mse": 1.0,
        "cosine": 0.0,
        "contrastive": 0.0,
        "auxiliary_classification": 0.0,
        "session_consistency": 0.0,
        "prototype": 0.0,
    }
    common = (
        output,
        target,
        torch.tensor([0]),
        torch.zeros(1, 6),
        torch.ones(6),
        None,
        torch.zeros(8, 4),
        weights,
        {"mode": "hard_multi_positive", "temperature": 0.07},
        {"alignment": 1.0, "classification": 1.0},
    )
    token_loss, _, _ = alignment_losses(*common, alignment_level="token")
    pooled_loss, _, _ = alignment_losses(*common, alignment_level="pooled")
    assert token_loss > 0
    torch.testing.assert_close(pooled_loss, torch.zeros_like(pooled_loss))
