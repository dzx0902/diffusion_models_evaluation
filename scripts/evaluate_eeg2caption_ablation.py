"""Evaluate adapted EEG2Caption A/B checkpoints and export deterministic captions."""

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
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ms_video_eval.eeg2caption_adapter import (
    AdaptedThreeSessionDataset,
    CompactStructuredClassifier,
    OBJECT_NAMES,
    build_semantic_targets,
    load_eeg2caption_fold,
    natural_object_caption,
    predicted_object_sets,
)
from ms_video_eval.semantic_caption import (
    COARSE_ACTION_VERBALIZATIONS,
    SlotPrediction,
    verbalize_relations,
    verbalize_semantics,
)
from ms_video_eval.semantic_data import SemanticVocabulary
from scripts.train_eeg2caption_ablation import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def restore_vocabulary(payload: dict[str, Any]) -> SemanticVocabulary:
    return SemanticVocabulary(
        values={key: tuple(value) for key, value in payload.get("values", {}).items()},
        min_frequency={key: int(value) for key, value in payload.get("min_frequency", {}).items()},
        unknown_token=payload.get("unknown_token", "__unknown__"),
    )


def semantic_predictions(
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


def caption_for(
    method: str,
    objects: list[str],
    predictions: dict[str, list[SlotPrediction]],
    thresholds: dict[str, float],
) -> str:
    base = natural_object_caption(objects)
    if method != "structured_semantic":
        return base
    relations = predictions.get("relation", [])
    if relations:
        relation = verbalize_relations(relations[:1], thresholds.get("relation", 0.5))
        return f"{base} {relation}" if relation != "A video." else base
    action = predictions.get("fine_action", [])[:1]
    if not action and predictions.get("coarse_action"):
        coarse = predictions["coarse_action"][0]
        action = [SlotPrediction(
            COARSE_ACTION_VERBALIZATIONS.get(coarse.value, "is shown with"),
            coarse.confidence,
        )]
    if not action:
        return base
    animate = [value for value in objects if value in {"person", "dog", "bird"}]
    inanimate = [value for value in objects if value not in {"person", "dog", "bird"}]
    action_caption = verbalize_semantics(
        {
            "subject": [SlotPrediction(value, 1.0) for value in animate],
            "object": [SlotPrediction(value, 1.0) for value in inanimate],
            "action": action,
        },
        {"subject": 0.0, "object": 0.0, "action": 0.0},
    )
    return f"{base} {action_caption}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data = config["data"]
    fold = load_eeg2caption_fold(
        ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
        resolve(data["split_plan"]), data["fold"],
        tuple(data.get("sessions", ("session1", "session2", "session3"))),
        int(config["model"].get("sample_points", 800)),
    )
    vocabulary = restore_vocabulary(checkpoint["vocabulary"])
    targets, masks = build_semantic_targets(fold, vocabulary)
    dataset = AdaptedThreeSessionDataset(
        eeg=fold.eeg, labels=fold.object_labels, pair_indices=fold.category_targets,
        cardinalities=fold.cardinalities, ids=fold.video_ids,
        indices=fold.split_indices[args.partition],
        mean=checkpoint["normalization_mean"], std=checkpoint["normalization_std"],
        eeg_scale=float(checkpoint["eeg_scale"]), semantic_targets=targets,
        semantic_masks=masks,
    )
    loader = DataLoader(
        dataset, batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False, num_workers=int(config["training"].get("workers", 0)),
    )
    model = CompactStructuredClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    threshold_candidates = [float(value) for value in config.get("semantic", {}).get(
        "threshold_search", (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    )]
    thresholds = {
        key: float(value) for key, value in checkpoint["validation"].get(
            "structured_thresholds", {}
        ).items()
    }
    metrics, raw = evaluate(
        model, loader, device, threshold_candidates, fixed_thresholds=thresholds
    )
    method = checkpoint["method"]
    object_predictions, category_predictions = predicted_object_sets(
        raw["fused_object_logits"], raw["fused_pair_logits"]
    )
    video_records = []
    trial_records = []
    for index, video_id in enumerate(raw["video_ids"]):
        object_indices = object_predictions[index].nonzero().flatten().tolist()
        objects = [OBJECT_NAMES[item] for item in object_indices]
        structured = semantic_predictions(
            raw["semantic_logits"], vocabulary, thresholds, index
        )
        caption = caption_for(method, objects, structured, thresholds)
        video_records.append({
            "video_id": video_id,
            "aggregation": "mean_session_logits",
            "sessions": list(data.get("sessions", ("session1", "session2", "session3"))),
            "caption": caption,
            "predicted_objects": objects,
            "predicted_category": int(category_predictions[index]) + 1,
            "object_probabilities": {
                name: float(raw["fused_object_logits"][index, item].sigmoid())
                for item, name in enumerate(OBJECT_NAMES)
            },
            "slots": {
                slot: [{"value": item.value, "confidence": item.confidence} for item in values]
                for slot, values in structured.items()
            },
        })
        for session_index, session in enumerate(data.get("sessions", ("session1", "session2", "session3"))):
            session_objects_tensor, session_categories = predicted_object_sets(
                raw["session_object_logits"][index:index + 1, session_index],
                raw["session_pair_logits"][index:index + 1, session_index],
            )
            session_objects = [
                OBJECT_NAMES[item]
                for item in session_objects_tensor[0].nonzero().flatten().tolist()
            ]
            trial_records.append({
                "video_id": video_id, "session": session,
                "caption": natural_object_caption(session_objects),
                "predicted_objects": session_objects,
                "predicted_category": int(session_categories[0]) + 1,
            })
    unique_captions = len({row["caption"] for row in video_records})
    result = {
        "schema_version": 2,
        "implementation": "EEG2Caption Compact adapted",
        "checkpoint": str(args.checkpoint.resolve()),
        "partition": args.partition,
        "single_trial_primary": False,
        "session_fusion_primary": True,
        "thresholds_selected_on_validation": thresholds,
        "trial_metrics": {},
        "video_aggregation_metrics": metrics,
        "trial_records": trial_records,
        "video_aggregation_records": video_records,
        "collapse_audit": {
            "video_count": len(video_records), "unique_captions": unique_captions,
            "largest_caption_group": max(
                sum(other["caption"] == row["caption"] for other in video_records)
                for row in video_records
            ),
            "passed": unique_captions > 1,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "predictions.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"fused": metrics["fused"], "collapse": result["collapse_audit"]}, indent=2))
    print(f"[eeg2caption-eval] videos={len(video_records)} output={path}", flush=True)


if __name__ == "__main__":
    main()
