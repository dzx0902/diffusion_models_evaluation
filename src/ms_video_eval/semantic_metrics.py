"""Metrics for independently decoded semantic slots."""

from __future__ import annotations

from typing import Mapping

import torch


def search_slot_thresholds(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    candidates: list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Select each slot threshold by validation macro F1 only."""

    if not candidates or any(not 0.0 <= value <= 1.0 for value in candidates):
        raise ValueError("Threshold candidates must be non-empty values in [0, 1]")
    selected: dict[str, float] = {}
    for slot in logits:
        best_threshold = float(candidates[0])
        best_score = float("-inf")
        for threshold in candidates:
            values = multilabel_slot_metrics(
                {slot: logits[slot]},
                {slot: targets[slot]},
                {slot: target_masks[slot]},
                {slot: float(threshold)},
            )
            score = float(values[slot]["macro_f1"])
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        selected[slot] = best_threshold
    return selected


def multilabel_slot_metrics(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    thresholds = thresholds or {}
    result: dict[str, object] = {}
    macro_values = []
    micro_values = []
    for slot, prediction in logits.items():
        target = targets[slot].bool()
        mask = target_masks[slot].bool()
        if prediction.shape != target.shape or mask.shape != (prediction.shape[0],):
            raise ValueError(f"Invalid metric shapes for slot {slot!r}")
        if not mask.any():
            result[slot] = {"examples": 0, "macro_f1": 0.0, "micro_f1": 0.0, "exact": 0.0}
            continue
        pred = prediction.sigmoid().ge(float(thresholds.get(slot, 0.5)))[mask]
        truth = target[mask]
        tp = (pred & truth).sum(dim=0).float()
        fp = (pred & ~truth).sum(dim=0).float()
        fn = (~pred & truth).sum(dim=0).float()
        class_f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        supported = truth.sum(dim=0) > 0
        macro = class_f1[supported].mean().item() if supported.any() else 0.0
        micro = (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1)).item()
        exact = pred.eq(truth).all(dim=-1).float().mean().item()
        coverage = pred.any(dim=-1).float().mean().item()
        predicted_cardinality = pred.sum(dim=-1).float().mean().item()
        target_cardinality = truth.sum(dim=-1).float().mean().item()
        result[slot] = {
            "examples": int(mask.sum()),
            "macro_f1": macro,
            "micro_f1": micro,
            "exact": exact,
            "coverage": coverage,
            "predicted_cardinality": predicted_cardinality,
            "target_cardinality": target_cardinality,
        }
        macro_values.append(macro)
        micro_values.append(micro)
    result["aggregate"] = {
        "macro_f1": sum(macro_values) / max(1, len(macro_values)),
        "micro_f1": sum(micro_values) / max(1, len(micro_values)),
    }
    return result
