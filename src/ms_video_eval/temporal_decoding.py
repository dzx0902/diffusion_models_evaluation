"""Leakage-safe post-hoc decoders for temporal EEG2Caption predictions."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .eeg2caption_adapter import CATEGORY_NAMES, category_object_matrix


def _category_matrix(allowed_categories: Sequence[str]) -> torch.Tensor:
    indices = torch.tensor(
        [CATEGORY_NAMES.index(name) for name in allowed_categories], dtype=torch.long
    )
    return category_object_matrix()[indices]


def temporal_decoder_predictions(
    segment_category_logits: torch.Tensor,
    segment_object_logits: torch.Tensor,
    allowed_categories: Sequence[str],
    *,
    hybrid_alpha: float = 0.5,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return category and object-set predictions for each decoding rule."""

    if segment_category_logits.ndim != 4 or segment_object_logits.ndim != 4:
        raise ValueError("Expected [video,session,segment,class] logits")
    if segment_category_logits.shape[:3] != segment_object_logits.shape[:3]:
        raise ValueError("Category and object logits must share video/session/segment axes")
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("hybrid_alpha must lie in [0,1]")
    matrix = _category_matrix(allowed_categories).to(segment_category_logits.device)
    if segment_category_logits.shape[-1] != len(allowed_categories):
        raise ValueError("Category logit dimension does not match allowed_categories")
    if segment_object_logits.shape[-1] != matrix.shape[-1]:
        raise ValueError("Object logit dimension does not match ontology")

    category_log_probability = F.log_softmax(segment_category_logits, dim=-1).mean(dim=(1, 2))
    category_probability = segment_category_logits.softmax(dim=-1).mean(dim=(1, 2))
    mean_logit = segment_category_logits.mean(dim=(1, 2))
    votes = F.one_hot(
        segment_category_logits.argmax(dim=-1), num_classes=len(allowed_categories)
    ).float().mean(dim=(1, 2))
    # A tiny probability tie-break keeps majority vote deterministic.
    majority_score = votes + category_probability * 1e-6

    object_probability = segment_object_logits.sigmoid().mean(dim=(1, 2))
    top2_indices = object_probability.topk(2, dim=-1).indices
    top2_objects = torch.zeros_like(object_probability)
    top2_objects.scatter_(1, top2_indices, 1.0)
    top2_category = torch.full(
        (len(top2_objects),), -1, dtype=torch.long, device=top2_objects.device
    )
    for category, pair in enumerate(matrix):
        matched = top2_objects.eq(pair).all(dim=-1)
        top2_category[matched] = category

    log_positive = F.logsigmoid(segment_object_logits)
    log_negative = F.logsigmoid(-segment_object_logits)
    bernoulli_pair = (
        log_positive[..., None, :] * matrix[None, None, None, :, :]
        + log_negative[..., None, :] * (1.0 - matrix[None, None, None, :, :])
    ).sum(dim=-1).mean(dim=(1, 2))
    valid_pair_category = bernoulli_pair.argmax(dim=-1)
    hybrid_score = (
        (1.0 - hybrid_alpha) * category_log_probability
        + hybrid_alpha * bernoulli_pair / matrix.shape[-1]
    )

    def category_result(score: torch.Tensor) -> dict[str, torch.Tensor]:
        category = score.argmax(dim=-1)
        return {"category": category, "objects": matrix[category], "score": score}

    return {
        "mean_logit": category_result(mean_logit),
        "mean_probability": category_result(category_probability),
        "majority_vote": category_result(majority_score),
        "object_top2": {
            "category": top2_category, "objects": top2_objects,
            "score": object_probability,
        },
        "valid_pair_object": {
            "category": valid_pair_category,
            "objects": matrix[valid_pair_category], "score": bernoulli_pair,
        },
        "hybrid": category_result(hybrid_score),
    }


def decoder_metrics(
    predictions: dict[str, dict[str, torch.Tensor]],
    category_targets: torch.Tensor,
    object_targets: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, values in predictions.items():
        category = values["category"].cpu()
        objects = values["objects"].cpu()
        valid = category.ge(0)
        correct = category.eq(category_targets.cpu())
        object_correct = objects.eq(object_targets.cpu()).all(dim=-1)
        result[name] = {
            "video_count": len(category),
            "category_accuracy": float(correct.float().mean()),
            "object_exact": float(object_correct.float().mean()),
            "invalid_pair_rate": float((~valid).float().mean()),
            "unique_predicted_categories": int(category[valid].unique().numel()),
            "category_correct": correct.tolist(),
            "object_correct": object_correct.tolist(),
        }
    return result
