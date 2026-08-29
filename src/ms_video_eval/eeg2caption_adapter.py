"""Adapt the released EEG2Caption Compact pipeline to this repository protocol.

The Compact encoder itself is imported from ``EEG2Caption/src/common.py``.  This
module only adapts manifests, the eight-category ontology, structured heads,
and leakage-safe fold definitions used by the ablation framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from EEG2Caption.src.common import CompactEEGClassifier, ThreeSessionDataset

from .eeg_conditioner import CrossAttentionBlock
from .semantic_data import (
    SemanticVocabulary,
    load_semantic_record_map,
    load_trial_rows,
    load_video_partitions,
)
from .semantic_schema import CORE_ENTITIES_BY_CATEGORY, SemanticRecord


OBJECT_NAMES = ("person", "dog", "car", "ball", "flower", "bird")
CATEGORY_NAMES = tuple(sorted(CORE_ENTITIES_BY_CATEGORY))


@dataclass(frozen=True)
class EEG2CaptionFold:
    eeg: torch.Tensor
    video_ids: tuple[str, ...]
    object_labels: torch.Tensor
    category_targets: torch.Tensor
    cardinalities: torch.Tensor
    split_indices: dict[str, list[int]]
    records: dict[str, SemanticRecord]
    data_metadata: dict[str, Any]


def category_object_matrix() -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(OBJECT_NAMES)}
    matrix = torch.zeros(len(CATEGORY_NAMES), len(OBJECT_NAMES))
    for category_index, category in enumerate(CATEGORY_NAMES):
        for entity in CORE_ENTITIES_BY_CATEGORY[category]:
            matrix[category_index, lookup[entity]] = 1.0
    return matrix


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_eeg2caption_fold(
    project_root: Path,
    trials_path: Path,
    semantic_labels_path: Path,
    split_plan_path: Path,
    fold: str,
    sessions: Sequence[str],
    sample_points: int = 800,
) -> EEG2CaptionFold:
    """Load aligned videos as [video, session, channel, time], once per NPZ."""

    if len(sessions) != 3 or len(set(sessions)) != 3:
        raise ValueError("EEG2Caption requires exactly three distinct sessions")
    rows = load_trial_rows(trials_path)
    partitions = load_video_partitions(split_plan_path, fold)
    video_ids = tuple(sorted(set().union(*partitions.values())))
    if len(video_ids) != sum(map(len, partitions.values())):
        raise RuntimeError("Video leakage in EEG2Caption fold")
    records = load_semantic_record_map(semantic_labels_path)
    missing_records = set(video_ids) - set(records)
    if missing_records:
        raise KeyError(f"Missing semantic records: {sorted(missing_records)[:5]}")

    selected_rows: dict[tuple[str, str], dict[str, str]] = {}
    allowed = set(video_ids)
    wanted_sessions = set(sessions)
    for row in rows:
        key = (row["video_id"], row["session"])
        if row["video_id"] not in allowed or row["session"] not in wanted_sessions:
            continue
        if key in selected_rows:
            raise ValueError(f"Duplicate EEG trial for {key}")
        selected_rows[key] = row
    expected = {(video_id, session) for video_id in video_ids for session in sessions}
    missing = expected - set(selected_rows)
    if missing:
        raise ValueError(f"Missing EEG trials such as {sorted(missing)[:5]}")

    arrays: dict[Path, np.ndarray] = {}
    for row in selected_rows.values():
        path = _resolve(project_root, row["npz_path"])
        if path not in arrays:
            with np.load(path, allow_pickle=False) as package:
                if "eeg" not in package:
                    raise KeyError(f"Missing eeg array in {path}")
                arrays[path] = np.asarray(package["eeg"])

    videos = []
    for video_id in video_ids:
        session_values = []
        for session in sessions:
            row = selected_rows[(video_id, session)]
            if int(row["length_samples"]) < sample_points:
                raise ValueError(f"{video_id}/{session} has fewer than {sample_points} samples")
            path = _resolve(project_root, row["npz_path"])
            signal = np.asarray(
                arrays[path][int(row["trial_index"]), :, :sample_points], dtype=np.float32
            )
            if signal.shape != (62, sample_points):
                raise ValueError(f"Unexpected EEG shape {signal.shape} for {video_id}/{session}")
            session_values.append(torch.from_numpy(signal.copy()))
        videos.append(torch.stack(session_values))
    eeg = torch.stack(videos)

    matrix = category_object_matrix()
    category_lookup = {name: index for index, name in enumerate(CATEGORY_NAMES)}
    category_targets = torch.tensor(
        [category_lookup[video_id.split("-", 1)[0]] for video_id in video_ids],
        dtype=torch.long,
    )
    object_labels = matrix[category_targets]
    cardinalities = object_labels.sum(dim=1).long()
    id_lookup = {video_id: index for index, video_id in enumerate(video_ids)}
    split_indices = {
        name: [id_lookup[video_id] for video_id in sorted(values)]
        for name, values in partitions.items()
    }
    return EEG2CaptionFold(
        eeg=eeg,
        video_ids=video_ids,
        object_labels=object_labels,
        category_targets=category_targets,
        cardinalities=cardinalities,
        split_indices=split_indices,
        records={video_id: records[video_id] for video_id in video_ids},
        data_metadata={
            "trials": str(trials_path.resolve()),
            "sessions": list(sessions),
            "sample_points": sample_points,
            "video_count": len(video_ids),
            "npz_paths": [str(path.resolve()) for path in sorted(arrays)],
        },
    )


def normalization_stats(
    eeg: torch.Tensor, train_indices: Sequence[int], eeg_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    means, standard_deviations = [], []
    for session_index in range(eeg.shape[1]):
        values = eeg[list(train_indices), session_index] * eeg_scale
        means.append(values.mean(dim=(0, 2)))
        standard_deviations.append(values.std(dim=(0, 2)).clamp_min(1e-6))
    return torch.stack(means), torch.stack(standard_deviations)


class AdaptedThreeSessionDataset(ThreeSessionDataset):
    def __init__(
        self,
        *args: Any,
        semantic_targets: Mapping[str, torch.Tensor] | None = None,
        semantic_masks: Mapping[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_targets = dict(semantic_targets or {})
        self.semantic_masks = dict(semantic_masks or {})

    def __getitem__(self, item: int) -> dict[str, Any]:
        result = super().__getitem__(item)
        index = int(result["global_index"])
        result["semantic_targets"] = {
            key: value[index] for key, value in self.semantic_targets.items()
        }
        result["semantic_masks"] = {
            key: value[index] for key, value in self.semantic_masks.items()
        }
        return result


class CompactStructuredClassifier(CompactEEGClassifier):
    """The released Compact model plus optional structured semantic heads."""

    def __init__(self, *args: Any, semantic_classes: Mapping[str, int] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        feature_dim = int(kwargs.get("feature_dim", 128))
        self.semantic_heads = nn.ModuleDict({
            name: nn.Linear(feature_dim, count)
            for name, count in (semantic_classes or {}).items()
        })

    def forward(self, eeg: torch.Tensor) -> dict[str, Any]:
        output = super().forward(eeg)
        features = output["features"]
        session_logits = {
            name: head(features) for name, head in self.semantic_heads.items()
        }
        output["session_semantic_logits"] = session_logits
        output["fused_semantic_logits"] = {
            name: values.mean(dim=1) for name, values in session_logits.items()
        }
        return output


class CompactToraAlignmentModel(CompactEEGClassifier):
    """The same Compact encoder followed by a Tora-condition query decoder."""

    def __init__(
        self,
        *args: Any,
        condition_slots: int,
        condition_dim: int,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        feature_dim = int(kwargs.get("feature_dim", 128))
        if condition_slots < 1 or condition_dim < 1:
            raise ValueError("Condition shape must be positive")
        if feature_dim % decoder_heads:
            raise ValueError("feature_dim must be divisible by decoder_heads")
        self.condition_slots = int(condition_slots)
        self.condition_dim = int(condition_dim)
        self.queries = nn.Parameter(torch.randn(condition_slots, feature_dim) * 0.02)
        self.condition_decoder = nn.ModuleList(
            CrossAttentionBlock(feature_dim, decoder_heads, float(kwargs.get("dropout", 0.35)))
            for _ in range(decoder_layers)
        )
        self.condition_head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, condition_dim)
        )

    def forward(self, eeg: torch.Tensor) -> dict[str, Any]:
        output = super().forward(eeg)
        context = output["features"]
        queries = self.queries.unsqueeze(0).expand(eeg.shape[0], -1, -1)
        for block in self.condition_decoder:
            queries = block(queries, context)
        output["latent"] = self.condition_head(queries)
        return output


def build_semantic_targets(
    fold: EEG2CaptionFold,
    vocabulary: SemanticVocabulary,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    targets: dict[str, list[torch.Tensor]] = {slot: [] for slot in vocabulary.values}
    masks: dict[str, list[torch.Tensor]] = {slot: [] for slot in vocabulary.values}
    for video_id in fold.video_ids:
        encoded, encoded_masks = vocabulary.encode(fold.records[video_id])
        for slot in targets:
            targets[slot].append(encoded[slot])
            masks[slot].append(encoded_masks[slot])
    return (
        {slot: torch.stack(values) for slot, values in targets.items()},
        {slot: torch.stack(values) for slot, values in masks.items()},
    )


def predicted_object_sets(
    object_logits: torch.Tensor,
    category_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K objects where K comes from the predicted category, never GT labels."""

    category_predictions = category_logits.argmax(dim=-1).cpu()
    cardinalities = category_object_matrix()[category_predictions].sum(dim=1).long()
    probabilities = object_logits.sigmoid().cpu()
    predictions = torch.zeros_like(probabilities)
    for index, cardinality in enumerate(cardinalities.tolist()):
        predictions[index, probabilities[index].topk(cardinality).indices] = 1.0
    return predictions, category_predictions


def natural_object_caption(objects: Sequence[str]) -> str:
    articles = {name: f"a {name}" for name in OBJECT_NAMES}
    rendered = [articles.get(value, value) for value in objects]
    if len(rendered) == 1:
        joined = rendered[0]
    elif len(rendered) == 2:
        joined = f"{rendered[0]} and {rendered[1]}"
    else:
        joined = ", ".join(rendered[:-1]) + f", and {rendered[-1]}"
    return f"A realistic video showing {joined} together."
