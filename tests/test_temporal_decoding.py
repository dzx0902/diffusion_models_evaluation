from __future__ import annotations

import torch

from src.ms_video_eval.eeg2caption_adapter import CATEGORY_NAMES, category_object_matrix
from src.ms_video_eval.temporal_decoding import decoder_metrics, temporal_decoder_predictions


ALLOWED = CATEGORY_NAMES[:6]


def test_temporal_decoders_accumulate_windows_and_preserve_valid_pairs() -> None:
    category_logits = torch.full((2, 3, 4, 6), -4.0)
    category_logits[0, ..., 0] = 4.0
    category_logits[1, ..., 5] = 4.0
    object_logits = torch.full((2, 3, 4, 6), -5.0)
    # 01 = person + ball; 06 = person + bird.
    object_logits[0, ..., 0] = 5.0
    object_logits[0, ..., 3] = 5.0
    object_logits[1, ..., 0] = 5.0
    object_logits[1, ..., 5] = 5.0
    predictions = temporal_decoder_predictions(
        category_logits, object_logits, ALLOWED, hybrid_alpha=0.5
    )
    for decoder in (
        "mean_logit", "mean_probability", "majority_vote",
        "object_top2", "valid_pair_object", "hybrid",
    ):
        assert predictions[decoder]["category"].tolist() == [0, 5]
    targets = torch.tensor([0, 5])
    objects = category_object_matrix()[targets]
    metrics = decoder_metrics(predictions, targets, objects)
    assert all(value["category_accuracy"] == 1.0 for value in metrics.values())
    assert metrics["object_top2"]["invalid_pair_rate"] == 0.0


def test_unconstrained_top2_reports_invalid_pair_but_constrained_decoder_does_not() -> None:
    category_logits = torch.zeros(1, 3, 2, 6)
    object_logits = torch.full((1, 3, 2, 6), -5.0)
    # person + dog is not a valid pair in categories 01--06.
    object_logits[..., 0] = 5.0
    object_logits[..., 1] = 4.0
    predictions = temporal_decoder_predictions(category_logits, object_logits, ALLOWED)
    assert predictions["object_top2"]["category"].item() == -1
    assert predictions["valid_pair_object"]["category"].item() >= 0
    targets = torch.tensor([0])
    objects = category_object_matrix()[targets]
    metrics = decoder_metrics(predictions, targets, objects)
    assert metrics["object_top2"]["invalid_pair_rate"] == 1.0
    assert metrics["valid_pair_object"]["invalid_pair_rate"] == 0.0


def test_temporal_decoder_rejects_invalid_alpha() -> None:
    logits = torch.zeros(1, 3, 1, 6)
    try:
        temporal_decoder_predictions(logits, logits, ALLOWED, hybrid_alpha=1.1)
    except ValueError as error:
        assert "hybrid_alpha" in str(error)
    else:
        raise AssertionError("Expected invalid alpha to fail")
