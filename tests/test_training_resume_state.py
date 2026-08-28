from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_eeg_semantic import atomic_torch_save, capture_rng_state, restore_rng_state


def test_rng_state_restores_training_and_loader_streams() -> None:
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(13)
    state = capture_rng_state(generator)
    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    restore_rng_state(state, generator)
    actual = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3])


def test_atomic_torch_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    atomic_torch_save({"epoch": 3}, path)
    assert torch.load(path, weights_only=True) == {"epoch": 3}
    assert not path.with_suffix(".pt.tmp").exists()
