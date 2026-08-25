from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ms_video_eval.eeg_pooled_retriever import (
    EEGPooledRetriever,
    EEGPooledRetrieverConfig,
    full_bank_contrastive_loss,
    grouped_retrieval_metrics,
    pooled_retrieval_loss,
    positive_mask,
    retrieval_ranks,
)
from build_eeg_pooled_retrieval_suite import select_trials


def test_pooled_retriever_output_shape() -> None:
    config = EEGPooledRetrieverConfig(
        channels=4,
        sample_points=200,
        sampling_rate=100,
        hidden_dim=32,
        target_dim=16,
        token_count=8,
        encoder_layers=1,
        heads=8,
        architecture="multiscale",
    )
    model = EEGPooledRetriever(config).eval()

    with torch.inference_mode():
        output = model(torch.randn(3, 4, 200))

    assert tuple(output.shape) == (3, 16)


def test_multi_positive_loss_is_finite() -> None:
    predicted = torch.randn(4, 16, requires_grad=True)
    target = torch.randn(4, 16)
    labels = ["same", "same", "left", "right"]

    loss, metrics = pooled_retrieval_loss(
        predicted,
        target,
        positive_mask(labels, predicted.device),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(predicted.grad).all()
    assert metrics["contrastive_loss"] >= 0


def test_full_bank_loss_prefers_correct_candidates() -> None:
    candidates = torch.eye(4)
    true_indices = torch.tensor([2, 0])
    correct = torch.stack([candidates[2], candidates[0]]).requires_grad_()
    wrong = torch.stack([candidates[1], candidates[3]])

    correct_loss = full_bank_contrastive_loss(
        correct, candidates, true_indices, temperature=0.1
    )
    wrong_loss = full_bank_contrastive_loss(
        wrong, candidates, true_indices, temperature=0.1
    )
    correct_loss.backward()

    assert correct_loss < wrong_loss
    assert torch.isfinite(correct.grad).all()


def test_pooled_loss_accepts_full_bank() -> None:
    candidates = torch.eye(4)
    predicted = torch.randn(3, 4, requires_grad=True)
    target = candidates[torch.tensor([0, 1, 2])]
    indices = torch.tensor([0, 1, 2])

    loss, metrics = pooled_retrieval_loss(
        predicted,
        target,
        torch.eye(3, dtype=torch.bool),
        contrastive_candidates=candidates,
        contrastive_true_indices=indices,
        variance_target_std=candidates.std(dim=0, unbiased=False),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(predicted.grad).all()
    assert metrics["contrastive_loss"] >= 0


def test_retrieval_ranks_report_exact_matches() -> None:
    candidates = torch.eye(3)
    predicted = torch.stack([candidates[2], candidates[0]])
    true_indices = torch.tensor([2, 0])

    ranks, similarities = retrieval_ranks(predicted, candidates, true_indices)

    torch.testing.assert_close(ranks, torch.ones(2))
    assert tuple(similarities.shape) == (2, 3)


def test_retrieval_ranks_penalize_ties() -> None:
    predicted = torch.zeros(1, 3)
    candidates = torch.eye(3)

    ranks, _ = retrieval_ranks(predicted, candidates, torch.tensor([0]))

    torch.testing.assert_close(ranks, torch.tensor([2.0]))


def test_grouped_retrieval_averages_repeated_observations() -> None:
    candidates = torch.eye(3)
    predicted = torch.tensor(
        [
            [0.6, 0.8, 0.0],
            [0.6, -0.8, 0.0],
            [0.0, 0.6, 0.8],
            [0.0, 0.6, -0.8],
        ]
    )
    metrics = grouped_retrieval_metrics(
        predicted,
        candidates,
        torch.tensor([0, 0, 1, 1]),
        ["video-a", "video-a", "video-b", "video-b"],
    )

    assert metrics["count"] == 2
    assert metrics["recall_at_1"] == 1.0
    assert metrics["mean_rank"] == 1.0


def test_retrieval_suite_selects_best_unique_video() -> None:
    base = {
        "reciprocal_rank": "1.0",
        "cosine": "0.5",
        "standardized_mse": "1.0",
        "energy_ratio": "0.8",
        "nearest_prompt": "A person kicks a ball.",
        "nearest_video_id": "01-001",
        "nearest_cosine": "0.6",
        "retrieval_margin": "0.1",
        "prompt": "A person kicks a ball.",
    }
    metrics = [
        {**base, "video_id": "01-001", "session": "session1", "trial_index": "0", "rank": "1"},
        {**base, "video_id": "01-001", "session": "session2", "trial_index": "0", "rank": "2"},
        {**base, "video_id": "02-001", "session": "session1", "trial_index": "1", "rank": "3"},
    ]
    targets = {
        "01-001": {"latent_path": "/tmp/01.pt"},
        "02-001": {"latent_path": "/tmp/02.pt"},
    }

    selected = select_trials(metrics, targets, top_k=2)

    assert [row["video_id"] for row in selected] == ["01-001", "02-001"]
    assert selected[0]["session"] == "session1"
    assert selected[0]["nearest_exact_latent_path"] == "/tmp/01.pt"
