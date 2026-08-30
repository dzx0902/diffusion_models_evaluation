"""Select temporal EEG decoders on validation and evaluate the locked rules on test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ms_video_eval.eeg2caption_adapter import (
    AdaptedThreeSessionDataset,
    OBJECT_NAMES,
    TemporalSegmentCompactClassifier,
    load_eeg2caption_fold,
    natural_object_caption,
    subset_eeg2caption_fold,
)
from ms_video_eval.temporal_decoding import decoder_metrics, temporal_decoder_predictions
from scripts.train_eeg2caption_temporal import evaluate_temporal


DECODER_ORDER = (
    "mean_logit", "mean_probability", "majority_vote",
    "object_top2", "valid_pair_object", "hybrid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hybrid-alphas", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_partition(
    fold: Any,
    partition: str,
    checkpoint: dict[str, Any],
    model: TemporalSegmentCompactClassifier,
    device: torch.device,
    allowed_categories: tuple[str, ...],
) -> dict[str, Any]:
    config = checkpoint["config"]
    dataset = AdaptedThreeSessionDataset(
        eeg=fold.eeg, labels=fold.object_labels, pair_indices=fold.category_targets,
        cardinalities=fold.cardinalities, ids=fold.video_ids,
        indices=fold.split_indices[partition], mean=checkpoint["normalization_mean"],
        std=checkpoint["normalization_std"], eeg_scale=float(checkpoint["eeg_scale"]),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False, num_workers=int(config["training"].get("workers", 0)),
    )
    _, raw = evaluate_temporal(model, loader, device, allowed_categories)
    return raw


def metrics_for_alpha(
    raw: dict[str, Any], allowed_categories: tuple[str, ...], alpha: float
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    predictions = temporal_decoder_predictions(
        raw["segment_category_logits"], raw["segment_object_logits"],
        allowed_categories, hybrid_alpha=alpha,
    )
    return decoder_metrics(predictions, raw["categories"], raw["labels"]), predictions


def compact_metrics(values: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ignored = {"category_correct", "object_correct"}
    return {
        decoder: {key: value for key, value in metrics.items() if key not in ignored}
        for decoder, metrics in values.items()
    }


def main() -> None:
    args = parse_args()
    if not args.hybrid_alphas or any(not 0.0 <= value <= 1.0 for value in args.hybrid_alphas):
        raise ValueError("All hybrid alphas must lie in [0,1]")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "temporal_category":
        raise ValueError("Checkpoint is not a temporal_category model")
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
    model = TemporalSegmentCompactClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    validation_raw = load_partition(
        fold, "validation", checkpoint, model, device, allowed_categories
    )
    test_raw = load_partition(fold, "test", checkpoint, model, device, allowed_categories)

    alpha_validation: dict[str, float] = {}
    for alpha in args.hybrid_alphas:
        values, _ = metrics_for_alpha(validation_raw, allowed_categories, alpha)
        alpha_validation[str(alpha)] = values["hybrid"]["category_accuracy"]
    selected_alpha = max(
        args.hybrid_alphas,
        key=lambda value: (alpha_validation[str(value)], -value),
    )
    validation_metrics, validation_predictions = metrics_for_alpha(
        validation_raw, allowed_categories, selected_alpha
    )
    test_metrics, test_predictions = metrics_for_alpha(
        test_raw, allowed_categories, selected_alpha
    )
    selected_decoder = max(
        DECODER_ORDER,
        key=lambda name: (
            validation_metrics[name]["category_accuracy"],
            validation_metrics[name]["object_exact"],
            -DECODER_ORDER.index(name),
        ),
    )

    per_video_rows = []
    for partition, raw, predictions, metrics in (
        ("validation", validation_raw, validation_predictions, validation_metrics),
        ("test", test_raw, test_predictions, test_metrics),
    ):
        for decoder in DECODER_ORDER:
            categories = predictions[decoder]["category"].cpu()
            objects = predictions[decoder]["objects"].cpu()
            for index, video_id in enumerate(raw["video_ids"]):
                category = int(categories[index])
                predicted_objects = [
                    OBJECT_NAMES[item] for item in objects[index].nonzero().flatten().tolist()
                ]
                per_video_rows.append({
                    "partition": partition, "decoder": decoder, "video_id": video_id,
                    "true_category": allowed_categories[int(raw["categories"][index])],
                    "predicted_category": (
                        allowed_categories[category] if category >= 0 else "invalid"
                    ),
                    "predicted_objects": json.dumps(predicted_objects),
                    "category_correct": int(metrics[decoder]["category_correct"][index]),
                    "object_correct": int(metrics[decoder]["object_correct"][index]),
                })

    selected_records = []
    selected = test_predictions[selected_decoder]
    for index, video_id in enumerate(test_raw["video_ids"]):
        category = int(selected["category"][index].cpu())
        objects = [
            OBJECT_NAMES[item]
            for item in selected["objects"][index].cpu().nonzero().flatten().tolist()
        ]
        selected_records.append({
            "video_id": video_id, "caption": natural_object_caption(objects),
            "predicted_category": (
                allowed_categories[category] if category >= 0 else "invalid"
            ),
            "predicted_objects": objects, "decoder": selected_decoder,
        })
    counts = Counter(row["caption"] for row in selected_records)
    report = {
        "schema_version": 1, "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "scope": "first6_468_videos", "split_unit": "video",
            "validation_video_count": len(validation_raw["video_ids"]),
            "test_video_count": len(test_raw["video_ids"]),
            "chance_category_accuracy": 1.0 / len(allowed_categories),
            "allowed_categories": list(allowed_categories),
            "selection_partition": "validation", "evaluation_partition": "test",
        },
        "hybrid_alpha_search": alpha_validation,
        "selected_hybrid_alpha": selected_alpha,
        "validation_metrics": compact_metrics(validation_metrics),
        "test_metrics": compact_metrics(test_metrics),
        "selected_decoder": selected_decoder,
        "selected_test_metrics": compact_metrics(test_metrics)[selected_decoder],
        "collapse_audit": {
            "unique_captions": len(counts), "largest_caption_group": max(counts.values()),
            "passed": len(counts) > 1,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decoding_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "decoder_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "partition", "decoder", "video_count", "category_accuracy",
            "object_exact", "invalid_pair_rate", "unique_predicted_categories",
            "hybrid_alpha",
        ])
        writer.writeheader()
        for partition, values in (
            ("validation", validation_metrics), ("test", test_metrics)
        ):
            for decoder in DECODER_ORDER:
                writer.writerow({
                    "partition": partition, "decoder": decoder,
                    **compact_metrics(values)[decoder],
                    "hybrid_alpha": selected_alpha if decoder == "hybrid" else "",
                })
    with (args.output_dir / "per_video.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0]))
        writer.writeheader()
        writer.writerows(per_video_rows)
    selected_output = {
        "schema_version": 1, "implementation": "validation-selected temporal decoder",
        "checkpoint": str(args.checkpoint.resolve()), "partition": "test",
        "selected_decoder": selected_decoder, "selected_hybrid_alpha": selected_alpha,
        "video_aggregation_records": selected_records,
        "collapse_audit": report["collapse_audit"],
    }
    (args.output_dir / "selected_predictions.json").write_text(
        json.dumps(selected_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
