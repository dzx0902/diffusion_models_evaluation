"""Leakage-safe trial selection and semantic targets for EEG experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from .eeg_protocol import filter_trial_duration
from .semantic_schema import SemanticRecord, load_semantic_records, normalize_video_id


SLOT_FIELDS = {
    "subject": "subjects",
    "object": "objects",
    "count": "subject_count",
    "coarse_action": "coarse_actions",
    "fine_action": "fine_actions",
    "relation": "relations",
}


def load_video_partitions(plan_path: Path, experiment: str) -> dict[str, set[str]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = next(
        (item for item in plan.get("experiments", []) if item.get("name") == experiment),
        None,
    )
    if selected is None:
        raise KeyError(f"Unknown experiment {experiment!r} in {plan_path}")
    result = {
        name: {normalize_video_id(value) for value in selected[f"{name}_video_ids"]}
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = result[left] & result[right]
        if overlap:
            raise ValueError(f"Video leakage between {left}/{right}: {sorted(overlap)[:5]}")
    return result


def load_trial_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_id", "session", "npz_path", "trial_index", "length_samples"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Trial CSV must contain {sorted(required)}")
    for row in rows:
        row["video_id"] = normalize_video_id(row["video_id"])
        if row["session"] == "session_average":
            raise ValueError("session_average is forbidden in EEG semantic experiments")
    return rows


def select_partition_trials(
    rows: Sequence[dict[str, str]],
    video_ids: set[str],
    sessions: Sequence[str] = ("session1", "session2", "session3"),
    duration_sec: float | None = None,
) -> list[dict[str, str]]:
    allowed_sessions = set(sessions)
    selected = [
        row for row in rows
        if row["video_id"] in video_ids and row["session"] in allowed_sessions
    ]
    selected = filter_trial_duration(selected, duration_sec)
    unknown_ids = {row["video_id"] for row in selected} - video_ids
    if unknown_ids:
        raise RuntimeError(f"Partition selection escaped requested video IDs: {unknown_ids}")
    if not selected:
        raise ValueError("Partition/session/duration selection produced no EEG trials")
    return selected


def _slot_values(record: SemanticRecord, slot: str) -> tuple[str, ...]:
    if slot not in SLOT_FIELDS:
        raise KeyError(f"Unknown semantic slot {slot!r}")
    value = getattr(record, SLOT_FIELDS[slot])
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class SemanticVocabulary:
    values: dict[str, tuple[str, ...]]
    min_frequency: dict[str, int]
    unknown_token: str = "__unknown__"

    @classmethod
    def fit(
        cls,
        records: Iterable[SemanticRecord],
        slots: Sequence[str],
        min_frequency: Mapping[str, int] | None = None,
    ) -> "SemanticVocabulary":
        records = list(records)
        minimum = {slot: int((min_frequency or {}).get(slot, 1)) for slot in slots}
        if any(value < 1 for value in minimum.values()):
            raise ValueError("Vocabulary minimum frequencies must be >= 1")
        values: dict[str, tuple[str, ...]] = {}
        for slot in slots:
            counts = Counter(value for record in records for value in _slot_values(record, slot))
            kept = sorted(value for value, count in counts.items() if count >= minimum[slot])
            values[slot] = tuple([*kept, cls.unknown_token])
        return cls(values=values, min_frequency=minimum)

    def encode(self, record: SemanticRecord) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for slot, vocabulary in self.values.items():
            target = torch.zeros(len(vocabulary), dtype=torch.float32)
            raw_values = _slot_values(record, slot)
            lookup = {value: index for index, value in enumerate(vocabulary)}
            for value in raw_values:
                target[lookup.get(value, lookup[self.unknown_token])] = 1.0
            targets[slot] = target
            masks[slot] = torch.tensor(bool(raw_values), dtype=torch.float32)
        return targets, masks

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "values": self.values,
            "min_frequency": self.min_frequency,
            "unknown_token": self.unknown_token,
        }


class EEGSemanticDataset(Dataset):
    """One dataset item is one raw session trial, never a session average."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        semantic_records: Mapping[str, SemanticRecord],
        vocabulary: SemanticVocabulary,
        project_root: Path,
        sample_points: int = 800,
        eeg_array_cache: MutableMapping[Path, np.ndarray] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.semantic_records = dict(semantic_records)
        self.vocabulary = vocabulary
        self.project_root = project_root
        if sample_points < 1:
            raise ValueError("sample_points must be positive")
        self.sample_points = int(sample_points)
        # Caching the NpzFile handle is not enough: indexing a compressed
        # member inflates the complete EEG array on every __getitem__ call.
        # Cache the inflated array itself and allow train/validation sharing.
        self.eeg_array_cache = eeg_array_cache if eeg_array_cache is not None else {}
        missing = {row["video_id"] for row in self.rows} - set(self.semantic_records)
        if missing:
            raise KeyError(f"Semantic labels missing video IDs: {sorted(missing)[:5]}")
        # Preload before DataLoader workers start. Forked workers can then share
        # these read-only pages instead of each inflating the archives.
        for path in sorted({self._resolve_path(row["npz_path"]) for row in self.rows}):
            self._eeg_array(path)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _eeg_array(self, path: Path) -> np.ndarray:
        if path not in self.eeg_array_cache:
            with np.load(path, allow_pickle=False) as package:
                if "eeg" not in package:
                    raise KeyError(f"Missing 'eeg' array in {path}")
                self.eeg_array_cache[path] = np.asarray(package["eeg"])
        return self.eeg_array_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self._resolve_path(row["npz_path"])
        eeg_array = self._eeg_array(path)
        length = int(row["length_samples"])
        signal = np.asarray(
            eeg_array[int(row["trial_index"]), :, : min(length, self.sample_points)],
            dtype=np.float32,
        )
        if signal.shape[-1] < self.sample_points:
            signal = np.pad(signal, ((0, 0), (0, self.sample_points - signal.shape[-1])))
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-6
        )
        record = self.semantic_records[row["video_id"]]
        targets, masks = self.vocabulary.encode(record)
        return {
            "eeg": torch.from_numpy(signal),
            "targets": targets,
            "target_masks": masks,
            "video_id": row["video_id"],
            "session": row["session"],
            "caption": record.caption,
            "length_samples": length,
        }


class VideoGroupedBatchSampler(Sampler[list[int]]):
    """Keep all selected sessions of one video in a common mini-batch."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["video_id"], []).append(index)
        if batch_size < 1 or any(len(group) > batch_size for group in groups.values()):
            raise ValueError("batch_size must fit every complete video/session group")
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self.shuffle:
            order = torch.randperm(len(self.groups), generator=self.generator).tolist()
            groups = [self.groups[index] for index in order]
        else:
            groups = self.groups
        batch: list[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        count = 0
        size = 0
        for group in self.groups:
            if size and size + len(group) > self.batch_size:
                count += 1
                size = 0
            size += len(group)
        return count + int(size > 0)


def semantic_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["eeg"].shape[-1] for item in batch)
    eeg = torch.zeros(len(batch), batch[0]["eeg"].shape[0], max_length)
    for index, item in enumerate(batch):
        eeg[index, :, : item["eeg"].shape[-1]] = item["eeg"]
    slots = batch[0]["targets"]
    return {
        "eeg": eeg,
        "targets": {slot: torch.stack([item["targets"][slot] for item in batch]) for slot in slots},
        "target_masks": {
            slot: torch.stack([item["target_masks"][slot] for item in batch]) for slot in slots
        },
        "video_id": [item["video_id"] for item in batch],
        "session": [item["session"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "length_samples": torch.tensor([item["length_samples"] for item in batch]),
    }


def load_semantic_record_map(path: Path) -> dict[str, SemanticRecord]:
    return {record.video_id: record for record in load_semantic_records(path)}
