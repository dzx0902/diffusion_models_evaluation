"""Evaluate a Method A/B checkpoint and export confidence-aware captions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditionerConfig
from ms_video_eval.eeg_semantic import SemanticSlotModel
from ms_video_eval.semantic_caption import (
    COARSE_ACTION_VERBALIZATIONS,
    SlotPrediction,
    verbalize_relations,
    verbalize_semantics,
)
from ms_video_eval.semantic_data import (
    EEGSemanticDataset,
    SemanticVocabulary,
    load_semantic_record_map,
    load_trial_rows,
    load_video_partitions,
    select_partition_trials,
    semantic_collate,
)
from ms_video_eval.semantic_metrics import multilabel_slot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def restore_vocabulary(payload: dict[str, Any]) -> SemanticVocabulary:
    return SemanticVocabulary(
        values={key: tuple(value) for key, value in payload["values"].items()},
        min_frequency={key: int(value) for key, value in payload["min_frequency"].items()},
        unknown_token=payload.get("unknown_token", "__unknown__"),
    )


def selected_predictions(
    logits: dict[str, torch.Tensor],
    vocabulary: SemanticVocabulary,
    thresholds: dict[str, float],
    index: int,
) -> dict[str, list[SlotPrediction]]:
    result: dict[str, list[SlotPrediction]] = {}
    for slot, values in vocabulary.values.items():
        probabilities = logits[slot][index].sigmoid()
        selected = [
            SlotPrediction(value, float(probabilities[class_index]))
            for class_index, value in enumerate(values)
            if value != vocabulary.unknown_token
            and float(probabilities[class_index]) >= float(thresholds.get(slot, 0.5))
        ]
        result[slot] = sorted(selected, key=lambda item: item.confidence, reverse=True)
    return result


def caption_from_predictions(
    method: str,
    predictions: dict[str, list[SlotPrediction]],
    thresholds: dict[str, float],
) -> str:
    relations = predictions.get("relation", [])
    if method == "structured_semantic" and relations:
        return verbalize_relations(relations, thresholds.get("relation", 0.5))
    fine = predictions.get("fine_action", [])
    coarse = predictions.get("coarse_action", [])
    actions = fine[:1]
    if not actions and coarse:
        actions = [
            SlotPrediction(
                COARSE_ACTION_VERBALIZATIONS.get(coarse[0].value, "is shown with"),
                coarse[0].confidence,
            )
        ]
    return verbalize_semantics(
        {
            "subject": predictions.get("subject", []),
            "object": predictions.get("object", []),
            "count": predictions.get("count", []),
            "action": actions,
        },
        {
            "subject": thresholds.get("subject", 0.5),
            "object": thresholds.get("object", 0.5),
            "count": thresholds.get("count", 0.5),
            "action": min(
                thresholds.get("fine_action", 0.5), thresholds.get("coarse_action", 0.5)
            ),
        },
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data = config["data"]
    partitions = load_video_partitions(resolve_path(data["split_plan"]), data["fold"])
    rows = select_partition_trials(
        load_trial_rows(resolve_path(data["trials"])),
        partitions[args.partition],
        tuple(data.get("sessions", ["session1", "session2", "session3"])),
    )
    record_map = load_semantic_record_map(resolve_path(data["semantic_labels"]))
    vocabulary = restore_vocabulary(checkpoint["vocabulary"])
    encoder_config = EEGConditionerConfig(**checkpoint["encoder_config"])
    dataset = EEGSemanticDataset(
        rows, record_map, vocabulary, ROOT, encoder_config.sample_points
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 24)),
        shuffle=False,
        num_workers=int(config["training"].get("workers", 0)),
        collate_fn=semantic_collate,
    )
    model = SemanticSlotModel(encoder_config, checkpoint["slot_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logits: dict[str, list[torch.Tensor]] = {}
    targets: dict[str, list[torch.Tensor]] = {}
    masks: dict[str, list[torch.Tensor]] = {}
    metadata: list[tuple[str, str]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            output = model(batch["eeg"].to(device))
            for slot, value in output["logits"].items():
                logits.setdefault(slot, []).append(value.cpu())
                targets.setdefault(slot, []).append(batch["targets"][slot])
                masks.setdefault(slot, []).append(batch["target_masks"][slot])
            metadata.extend(zip(batch["video_id"], batch["session"]))
            if args.smoke:
                break
    merged_logits = {key: torch.cat(value) for key, value in logits.items()}
    merged_targets = {key: torch.cat(value) for key, value in targets.items()}
    merged_masks = {key: torch.cat(value) for key, value in masks.items()}
    thresholds = {
        key: float(value)
        for key, value in checkpoint["validation"].get(
            "thresholds", config["semantic"].get("thresholds", {})
        ).items()
    }
    metrics = multilabel_slot_metrics(
        merged_logits, merged_targets, merged_masks, thresholds
    )
    trial_records = []
    for index, (video_id, session) in enumerate(metadata):
        predictions = selected_predictions(merged_logits, vocabulary, thresholds, index)
        trial_records.append(
            {
                "video_id": video_id,
                "session": session,
                "caption": caption_from_predictions(
                    checkpoint["method"], predictions, thresholds
                ),
                "slots": {
                    slot: [
                        {"value": item.value, "confidence": item.confidence}
                        for item in values
                    ]
                    for slot, values in predictions.items()
                },
            }
        )
    grouped: dict[str, list[int]] = {}
    for index, (video_id, _) in enumerate(metadata):
        grouped.setdefault(video_id, []).append(index)
    video_ids = list(grouped)
    video_logits = {
        slot: torch.stack([value[grouped[video_id]].mean(dim=0) for video_id in video_ids])
        for slot, value in merged_logits.items()
    }
    video_targets = {
        slot: torch.stack([value[grouped[video_id][0]] for video_id in video_ids])
        for slot, value in merged_targets.items()
    }
    video_masks = {
        slot: torch.stack([value[grouped[video_id][0]] for video_id in video_ids])
        for slot, value in merged_masks.items()
    }
    video_metrics = multilabel_slot_metrics(
        video_logits, video_targets, video_masks, thresholds
    )
    video_records = []
    for index, video_id in enumerate(video_ids):
        predictions = selected_predictions(video_logits, vocabulary, thresholds, index)
        video_records.append(
            {
                "video_id": video_id,
                "aggregation": "mean_session_logits",
                "sessions": [metadata[item][1] for item in grouped[video_id]],
                "caption": caption_from_predictions(
                    checkpoint["method"], predictions, thresholds
                ),
                "slots": {
                    slot: [
                        {"value": item.value, "confidence": item.confidence}
                        for item in values
                    ]
                    for slot, values in predictions.items()
                },
            }
        )
    output_dir = args.output_dir or args.checkpoint.parent / f"{args.partition}_semantic"
    if args.smoke:
        output_dir = ROOT / "outputs" / "semantic_smoke" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "partition": args.partition,
        "single_trial_primary": True,
        "thresholds_selected_on_validation": thresholds,
        "trial_metrics": metrics,
        "video_aggregation_metrics": video_metrics,
        "trial_records": trial_records,
        "video_aggregation_records": video_records,
    }
    path = output_dir / "predictions.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eeg-semantic-eval] trials={len(trial_records)} output={path}")
    print(json.dumps({"trial": metrics["aggregate"], "video": video_metrics["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
