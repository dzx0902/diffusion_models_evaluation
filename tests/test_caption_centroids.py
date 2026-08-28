from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_tora_caption_centroids import semantic_centroid


def test_raw_centroid_is_arithmetic_mean() -> None:
    states = [torch.ones(2, 3), torch.full((2, 3), 3.0)]
    assert torch.equal(semantic_centroid(states, "raw_mean"), torch.full((2, 3), 2.0))


def test_unit_rescaled_centroid_keeps_shape_and_scale() -> None:
    states = [torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 4.0]])]
    centroid = semantic_centroid(states, "unit_rescaled")
    assert centroid.shape == (1, 2)
    assert torch.allclose(centroid.norm(dim=-1), torch.tensor([3.0]))
