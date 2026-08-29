"""Shared data, Compact EEG model, splitting, and evaluation utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset


SESSIONS = ("session1", "session2", "session3")
OBJECT_NAMES = ("person", "dog", "car", "ball", "flower", "bird")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, map_location: Union[str, torch.device] = "cpu") -> Any:
    """Load packages on both older and newer PyTorch releases."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def normalize_video_id(value: object) -> str:
    return Path(str(value).strip()).stem.replace("-", "_")


def resolve_subject(subjects_root: Path, subject: str) -> Path:
    direct = subjects_root / subject
    if direct.is_dir():
        return direct
    mapping_path = subjects_root / "subject.txt"
    if mapping_path.is_file():
        mapping: Dict[str, str] = {}
        for line in mapping_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(",")
            if separator and key.strip() and value.strip():
                mapping[key.strip().lower()] = value.strip()
        mapped = mapping.get(subject.lower())
        if mapped and (subjects_root / mapped).is_dir():
            return subjects_root / mapped
    available = sorted(path.name for path in subjects_root.iterdir() if path.is_dir())
    raise FileNotFoundError(
        "Cannot resolve subject {!r}. Available examples: {}".format(
            subject, available[:8]
        )
    )


def load_label_package(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "Label package not found: {}. Run scripts/prepare_labels.sh first.".format(path)
        )
    package = torch_load(path)
    required = (
        "ids", "labels", "pair_indices", "cardinalities", "object_names",
        "pair_prefixes", "pair_names", "pair_objects",
    )
    missing = [key for key in required if key not in package]
    if missing:
        raise KeyError("Label package is missing {}.".format(missing))
    count = len(package["ids"])
    if count != 468 or package["labels"].shape != (468, 6):
        raise ValueError(
            "Expected the 468-video two-object package [468,6], got {} and {}.".format(
                count, tuple(package["labels"].shape)
            )
        )
    if list(package["object_names"]) != list(OBJECT_NAMES):
        raise ValueError("Unexpected object order: {}".format(package["object_names"]))
    if not bool((package["cardinalities"] == 2).all()):
        raise ValueError("This pipeline only supports videos containing two objects.")
    return package


def load_three_sessions(
    subject_dir: Path,
    expected_ids: Sequence[str],
    num_samples: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load EEG aligned as [video, session, 62 channels, time]."""
    arrays: List[torch.Tensor] = []
    paths: List[str] = []
    reference_channels: List[str] = []
    for session in SESSIONS:
        path = subject_dir / session / "EEG" / "eeg_data.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as package:
            ids = [normalize_video_id(value) for value in package["filename"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate EEG video IDs in {}.".format(path))
            lookup = {video_id: index for index, video_id in enumerate(ids)}
            missing = [video_id for video_id in expected_ids if video_id not in lookup]
            if missing:
                raise ValueError("{} is missing IDs such as {}.".format(path, missing[:3]))
            indices = np.asarray([lookup[video_id] for video_id in expected_ids])
            eeg = package["eeg"]
            if eeg.ndim != 3 or eeg.shape[1] != 62 or eeg.shape[2] < num_samples:
                raise ValueError(
                    "Expected EEG [N,62,>={}], got {} in {}.".format(
                        num_samples, eeg.shape, path
                    )
                )
            if "mask" in package:
                valid = np.asarray(package["mask"])[indices, :num_samples]
                if not bool(valid.all()):
                    raise ValueError(
                        "The first {} EEG samples contain padding in {}.".format(
                            num_samples, path
                        )
                    )
            selected = np.asarray(eeg[indices, :, :num_samples], dtype=np.float32)
            arrays.append(torch.from_numpy(selected.copy()))
            sfreq = float(np.asarray(package["sfreq"]).reshape(-1)[0])
            if not np.isclose(sfreq, 200.0):
                raise ValueError("Expected 200 Hz, got {} in {}.".format(sfreq, path))
            if "channels" in package:
                channels = [str(value) for value in package["channels"]]
                if reference_channels and channels != reference_channels:
                    raise ValueError("EEG channel order differs across sessions.")
                reference_channels = channels
        paths.append(str(path.resolve()))
    return torch.stack(arrays, dim=1), {
        "sessions": list(SESSIONS),
        "session_paths": paths,
        "sfreq": 200.0,
        "channels": reference_channels,
        "num_samples": num_samples,
    }


def stratified_video_split(
    pair_indices: torch.Tensor,
    val_per_class: int,
    test_per_class: int,
    seed: int,
) -> Dict[str, List[int]]:
    """Split unique videos by six object-pair classes, shared by all sessions."""
    generator = np.random.default_rng(seed)
    result: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    values = pair_indices.cpu().numpy()
    for class_index in sorted(np.unique(values).tolist()):
        indices = np.flatnonzero(values == class_index)
        generator.shuffle(indices)
        if val_per_class + test_per_class >= len(indices):
            raise ValueError("Validation + test consume class {}.".format(class_index))
        result["test"].extend(indices[:test_per_class].tolist())
        result["val"].extend(
            indices[test_per_class:test_per_class + val_per_class].tolist()
        )
        result["train"].extend(indices[test_per_class + val_per_class:].tolist())
    return {name: sorted(indices) for name, indices in result.items()}


def normalization_stats(
    eeg: torch.Tensor,
    train_indices: Sequence[int],
    eeg_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    means, standard_deviations = [], []
    for session_index in range(eeg.shape[1]):
        values = eeg[train_indices, session_index] * eeg_scale
        means.append(values.mean(dim=(0, 2)))
        standard_deviations.append(values.std(dim=(0, 2)).clamp_min(1e-6))
    return torch.stack(means), torch.stack(standard_deviations)


class ThreeSessionDataset(Dataset):
    def __init__(
        self,
        eeg: torch.Tensor,
        labels: torch.Tensor,
        pair_indices: torch.Tensor,
        cardinalities: torch.Tensor,
        ids: Sequence[str],
        indices: Sequence[int],
        mean: torch.Tensor,
        std: torch.Tensor,
        eeg_scale: float,
        training: bool = False,
        noise_std: float = 0.0,
        time_mask_samples: int = 0,
    ) -> None:
        self.eeg = eeg
        self.labels = labels.float()
        self.pair_indices = pair_indices.long()
        self.cardinalities = cardinalities.long()
        self.ids = list(ids)
        self.indices = list(indices)
        self.mean = mean[:, :, None]
        self.std = std[:, :, None]
        self.eeg_scale = float(eeg_scale)
        self.training = training
        self.noise_std = float(noise_std)
        self.time_mask_samples = int(time_mask_samples)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        global_index = self.indices[item]
        signal = (self.eeg[global_index] * self.eeg_scale - self.mean) / self.std
        signal = signal.clone()
        if self.training and self.noise_std > 0:
            signal.add_(torch.randn_like(signal) * self.noise_std)
        if self.training and self.time_mask_samples > 0:
            width = min(self.time_mask_samples, signal.shape[-1])
            start = torch.randint(0, signal.shape[-1] - width + 1, (1,)).item()
            signal[..., start:start + width] = 0
        return {
            "eeg": signal,
            "label": self.labels[global_index],
            "pair_index": self.pair_indices[global_index],
            "cardinality": self.cardinalities[global_index],
            "video_id": self.ids[global_index],
            "global_index": global_index,
        }


class CompactEEGClassifier(nn.Module):
    """EEGNet-style shared encoder with prediction-level three-session fusion."""

    def __init__(
        self,
        num_channels: int = 62,
        num_objects: int = 6,
        num_pairs: int = 6,
        temporal_filters: int = 16,
        spatial_multiplier: int = 2,
        feature_dim: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        spatial_filters = temporal_filters * spatial_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                (num_channels, 1),
                stride=(num_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                spatial_filters, spatial_filters, (1, 15), padding=(0, 7),
                groups=spatial_filters, bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters * 2, (1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_filters * 2 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.object_head = nn.Linear(feature_dim, num_objects)
        self.pair_head = nn.Linear(feature_dim, num_pairs)

    def forward(self, eeg: torch.Tensor) -> Dict[str, torch.Tensor]:
        if eeg.ndim != 4:
            raise ValueError("Expected EEG [batch,session,channel,time].")
        batch, sessions, channels, samples = eeg.shape
        values = eeg.reshape(batch * sessions, 1, channels, samples)
        features = self.project(self.refine(self.spatial(self.temporal(values))))
        features = features.reshape(batch, sessions, -1)
        object_logits = self.object_head(features)
        pair_logits = self.pair_head(features)
        return {
            "features": features,
            "fused_feature": features.mean(dim=1),
            "session_object_logits": object_logits,
            "session_pair_logits": pair_logits,
            "fused_object_logits": object_logits.mean(dim=1),
            "fused_pair_logits": pair_logits.mean(dim=1),
        }


def pair_label_matrix(package: Dict[str, Any]) -> torch.Tensor:
    matrix = torch.zeros(len(package["pair_prefixes"]), len(package["object_names"]))
    lookup = {name: index for index, name in enumerate(package["object_names"])}
    for pair_index, prefix in enumerate(package["pair_prefixes"]):
        for name in package["pair_objects"][prefix]:
            matrix[pair_index, lookup[name]] = 1
    return matrix


def classification_metrics(
    object_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
    pair_targets: torch.Tensor,
    cardinalities: torch.Tensor,
    pair_labels: torch.Tensor,
    object_names: Sequence[str],
) -> Dict[str, Any]:
    probabilities = object_logits.sigmoid().cpu()
    labels = labels.cpu()
    predictions = torch.zeros_like(labels)
    for index in range(len(labels)):
        top = probabilities[index].topk(int(cardinalities[index])).indices
        predictions[index, top] = 1
    exact = predictions.eq(labels).all(dim=1).float().mean().item()
    recall = ((predictions * labels).sum(dim=1) / cardinalities.float()).mean().item()
    pair_prediction = pair_logits.argmax(dim=1).cpu()
    pair_accuracy = pair_prediction.eq(pair_targets.cpu()).float().mean().item()
    constrained_exact = pair_labels[pair_prediction].eq(labels).all(dim=1).float().mean().item()
    per_object_ap = {
        name: float(
            average_precision_score(labels[:, index].numpy(), probabilities[:, index].numpy())
        )
        for index, name in enumerate(object_names)
    }
    return {
        "macro_ap": float(np.mean(list(per_object_ap.values()))),
        "per_object_ap": per_object_ap,
        "top2_exact_set_accuracy": exact,
        "top2_recall": recall,
        "pair_head_accuracy": pair_accuracy,
        "pair_head_exact_set_accuracy": constrained_exact,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    pair_labels: torch.Tensor,
    object_names: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model.eval()
    accumulated: Dict[str, List[torch.Tensor]] = {
        "session_object_logits": [],
        "session_pair_logits": [],
        "fused_object_logits": [],
        "fused_pair_logits": [],
        "labels": [],
        "pair_targets": [],
        "cardinalities": [],
        "global_indices": [],
    }
    video_ids: List[str] = []
    for batch in loader:
        output = model(batch["eeg"].to(device, non_blocking=True))
        for key in (
            "session_object_logits", "session_pair_logits",
            "fused_object_logits", "fused_pair_logits",
        ):
            accumulated[key].append(output[key].cpu())
        accumulated["labels"].append(batch["label"].cpu())
        accumulated["pair_targets"].append(batch["pair_index"].cpu())
        accumulated["cardinalities"].append(batch["cardinality"].cpu())
        accumulated["global_indices"].append(batch["global_index"].cpu())
        video_ids.extend(batch["video_id"])
    raw = {key: torch.cat(values) for key, values in accumulated.items()}
    raw["video_ids"] = video_ids
    metrics: Dict[str, Any] = {}
    for session_index, session_name in enumerate(SESSIONS):
        metrics[session_name] = classification_metrics(
            raw["session_object_logits"][:, session_index],
            raw["session_pair_logits"][:, session_index],
            raw["labels"], raw["pair_targets"], raw["cardinalities"],
            pair_labels, object_names,
        )
    metrics["fused"] = classification_metrics(
        raw["fused_object_logits"], raw["fused_pair_logits"],
        raw["labels"], raw["pair_targets"], raw["cardinalities"],
        pair_labels, object_names,
    )
    return metrics, raw
