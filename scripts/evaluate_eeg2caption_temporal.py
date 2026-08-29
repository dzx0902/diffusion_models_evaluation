"""Evaluate and export captions from the 01--06 temporal-window pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ms_video_eval.eeg2caption_adapter import (
    AdaptedThreeSessionDataset,
    CATEGORY_NAMES,
    OBJECT_NAMES,
    TemporalSegmentCompactClassifier,
    category_object_matrix,
    load_eeg2caption_fold,
    natural_object_caption,
    subset_eeg2caption_fold,
)
from scripts.train_eeg2caption_temporal import evaluate_temporal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data = config["data"]
    allowed_categories = tuple(checkpoint["allowed_categories"])
    fold = subset_eeg2caption_fold(
        load_eeg2caption_fold(
            ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
            resolve(data["split_plan"]), data["fold"],
            tuple(data.get("sessions", ("session1", "session2", "session3"))),
            int(config["model"].get("sample_points", 800)),
        ),
        allowed_categories,
    )
    dataset = AdaptedThreeSessionDataset(
        eeg=fold.eeg, labels=fold.object_labels,
        pair_indices=fold.category_targets, cardinalities=fold.cardinalities,
        ids=fold.video_ids, indices=fold.split_indices[args.partition],
        mean=checkpoint["normalization_mean"], std=checkpoint["normalization_std"],
        eeg_scale=float(checkpoint["eeg_scale"]),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False, num_workers=int(config["training"].get("workers", 0)),
    )
    model = TemporalSegmentCompactClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, raw = evaluate_temporal(model, loader, device, allowed_categories)
    category_indices = torch.tensor(
        [CATEGORY_NAMES.index(name) for name in allowed_categories], dtype=torch.long
    )
    matrix = category_object_matrix()[category_indices]
    predictions = raw["fused_category_logits"].argmax(dim=-1)
    records = []
    segment_records = []
    sessions = tuple(data.get("sessions", ("session1", "session2", "session3")))
    for index, video_id in enumerate(raw["video_ids"]):
        category_index = int(predictions[index])
        objects = [
            OBJECT_NAMES[item] for item in matrix[category_index].nonzero().flatten().tolist()
        ]
        records.append({
            "video_id": video_id, "caption": natural_object_caption(objects),
            "predicted_category": allowed_categories[category_index],
            "predicted_objects": objects,
            "aggregation": "mean_segment_then_mean_session_logits",
            "segment_seconds": checkpoint["model_config"]["segment_samples"]
            / float(config["model"].get("sampling_rate", 200)),
            "segment_count_per_session": model.segment_count,
            "sessions": list(sessions),
        })
        for session_index, session in enumerate(sessions):
            for segment_index in range(model.segment_count):
                prediction = int(raw["segment_category_logits"][
                    index, session_index, segment_index
                ].argmax())
                segment_records.append({
                    "video_id": video_id, "session": session,
                    "segment_index": segment_index,
                    "predicted_category": allowed_categories[prediction],
                })
    counts = Counter(row["caption"] for row in records)
    result = {
        "schema_version": 1,
        "implementation": "EEG2Caption Compact temporal",
        "checkpoint": str(args.checkpoint.resolve()), "partition": args.partition,
        "allowed_categories": list(allowed_categories),
        "metrics": metrics, "video_aggregation_metrics": {"fused": {
            "category_accuracy": metrics["video_category_accuracy"],
            "predicted_cardinality_exact_set": metrics["category_derived_object_exact"],
            "macro_ap": metrics["object_macro_ap"],
        }},
        "video_aggregation_records": records, "segment_records": segment_records,
        "collapse_audit": {
            "video_count": len(records), "unique_captions": len(counts),
            "largest_caption_group": max(counts.values()), "passed": len(counts) > 1,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "predictions.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "collapse": result["collapse_audit"]}, indent=2))
    print(f"[eeg2caption-temporal-eval] output={path}", flush=True)


if __name__ == "__main__":
    main()
