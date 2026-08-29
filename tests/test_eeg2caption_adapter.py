from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.ms_video_eval.eeg2caption_adapter import (
    CompactStructuredClassifier,
    CompactToraAlignmentModel,
    TemporalSegmentCompactClassifier,
    load_eeg2caption_fold,
    natural_object_caption,
    predicted_object_sets,
    subset_eeg2caption_fold,
)
from src.ms_video_eval.semantic_schema import semantic_record_from_source


def test_fold_adapter_aligns_three_sessions_and_existing_split(tmp_path: Path) -> None:
    ids = ("01-001", "02-001", "07-001")
    trial_rows = []
    for session_index, session in enumerate(("session1", "session2", "session3")):
        path = tmp_path / f"{session}.npz"
        eeg = np.stack([
            np.full((62, 800), video_index + session_index * 10, dtype=np.float32)
            for video_index in range(len(ids))
        ])
        np.savez_compressed(path, eeg=eeg)
        for index, video_id in enumerate(ids):
            trial_rows.append({
                "video_id": video_id, "session": session, "npz_path": str(path),
                "trial_index": index, "length_samples": 800,
            })
    trials = tmp_path / "trials.csv"
    with trials.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_rows[0])
        writer.writeheader()
        writer.writerows(trial_rows)
    labels = tmp_path / "labels.jsonl"
    records = [
        semantic_record_from_source({
            "video_id": video_id,
            "caption": "A person interacts with an object.",
            "caption_relations": ["person interacts object"],
        }, "fixture")
        for video_id in ids
    ]
    labels.write_text("".join(json.dumps(record.to_json()) + "\n" for record in records))
    plan = tmp_path / "split.json"
    plan.write_text(json.dumps({"experiments": [{
        "name": "fold1",
        "train_video_ids": [ids[0]],
        "validation_video_ids": [ids[1]],
        "test_video_ids": [ids[2]],
    }]}))
    fold = load_eeg2caption_fold(
        tmp_path, trials, labels, plan, "fold1",
        ("session1", "session2", "session3"), 800,
    )
    assert fold.eeg.shape == (3, 3, 62, 800)
    assert fold.split_indices == {"train": [0], "validation": [1], "test": [2]}
    assert fold.cardinalities.tolist() == [2, 2, 3]
    assert float(fold.eeg[0, 2, 0, 0]) == 20.0


def test_compact_adapter_preserves_three_session_fusion_shapes() -> None:
    model = CompactStructuredClassifier(
        num_channels=62,
        num_objects=6,
        num_pairs=8,
        temporal_filters=4,
        spatial_multiplier=1,
        feature_dim=16,
        dropout=0.0,
        semantic_classes={"coarse_action": 5},
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 62, 128))
    assert output["session_object_logits"].shape == (2, 3, 6)
    assert output["fused_object_logits"].shape == (2, 6)
    assert output["session_pair_logits"].shape == (2, 3, 8)
    assert output["fused_semantic_logits"]["coarse_action"].shape == (2, 5)
    assert torch.allclose(
        output["fused_object_logits"], output["session_object_logits"].mean(dim=1)
    )


def test_compact_tora_model_uses_same_session_encoder_and_condition_shape() -> None:
    model = CompactToraAlignmentModel(
        num_channels=62, num_objects=6, num_pairs=8,
        temporal_filters=4, spatial_multiplier=1, feature_dim=16, dropout=0.0,
        condition_slots=12, condition_dim=32, decoder_layers=1, decoder_heads=4,
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 62, 128))
    assert output["features"].shape == (2, 3, 16)
    assert output["latent"].shape == (2, 12, 32)
    assert output["fused_object_logits"].shape == (2, 6)


def test_temporal_compact_model_fuses_segments_then_sessions() -> None:
    model = TemporalSegmentCompactClassifier(
        segment_samples=32, total_samples=128,
        num_channels=62, num_objects=6, num_pairs=6,
        temporal_filters=4, spatial_multiplier=1, feature_dim=16, dropout=0.0,
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 62, 128))
    assert output["segment_features"].shape == (2, 3, 4, 16)
    assert output["segment_category_logits"].shape == (2, 3, 4, 6)
    assert output["session_category_logits"].shape == (2, 3, 6)
    assert output["fused_category_logits"].shape == (2, 6)
    assert torch.allclose(
        output["fused_category_logits"],
        output["segment_category_logits"].mean(dim=(1, 2)),
    )


def test_first_six_subset_preserves_video_partitions_and_remaps_categories(tmp_path: Path) -> None:
    ids = ("01-001", "02-001", "07-001")
    trial_rows = []
    for session_index, session in enumerate(("session1", "session2", "session3")):
        path = tmp_path / f"subset_{session}.npz"
        np.savez_compressed(path, eeg=np.zeros((3, 62, 800), dtype=np.float32))
        for index, video_id in enumerate(ids):
            trial_rows.append({
                "video_id": video_id, "session": session, "npz_path": str(path),
                "trial_index": index, "length_samples": 800,
            })
    trials = tmp_path / "subset_trials.csv"
    with trials.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_rows[0])
        writer.writeheader()
        writer.writerows(trial_rows)
    records = [
        semantic_record_from_source({
            "video_id": video_id, "caption": "A video.", "caption_relations": [],
        }, "fixture") for video_id in ids
    ]
    labels = tmp_path / "subset_labels.jsonl"
    labels.write_text("".join(json.dumps(record.to_json()) + "\n" for record in records))
    plan = tmp_path / "subset_split.json"
    plan.write_text(json.dumps({"experiments": [{
        "name": "fold1", "train_video_ids": [ids[0]],
        "validation_video_ids": [ids[1]], "test_video_ids": [ids[2]],
    }]}))
    fold = load_eeg2caption_fold(
        tmp_path, trials, labels, plan, "fold1",
        ("session1", "session2", "session3"), 800,
    )
    subset = subset_eeg2caption_fold(fold, ("01", "02", "03", "04", "05", "06"))
    assert subset.video_ids == ("01-001", "02-001")
    assert subset.category_targets.tolist() == [0, 1]
    assert subset.split_indices == {"train": [0], "validation": [1], "test": []}


def test_predicted_category_controls_cardinality_without_ground_truth() -> None:
    object_logits = torch.tensor([
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ])
    category_logits = torch.zeros(2, 8)
    category_logits[0, 0] = 10.0  # category 01 has two objects
    category_logits[1, 6] = 10.0  # category 07 has three objects
    predictions, categories = predicted_object_sets(object_logits, category_logits)
    assert predictions.sum(dim=1).tolist() == [2.0, 3.0]
    assert categories.tolist() == [0, 6]


def test_object_caption_supports_two_or_three_entities() -> None:
    assert natural_object_caption(["person", "ball"]) == (
        "A realistic video showing a person and a ball together."
    )
    assert natural_object_caption(["person", "dog", "ball"]) == (
        "A realistic video showing a person, a dog, and a ball together."
    )
