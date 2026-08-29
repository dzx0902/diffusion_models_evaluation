"""Convert fused EEG test predictions into aligned two-object video captions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from common import normalize_video_id, torch_load


DEFAULT_DATASET = Path("/userhome2/zhoutianyi/Dataset/Multi-Object")
ARTICLES = {
    "person": "a person",
    "dog": "a dog",
    "car": "a car",
    "ball": "a ball",
    "flower": "a flower",
    "bird": "a bird",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--prompt-suffix",
        default=(
            "Both subjects remain clearly visible throughout the shot. Natural motion, "
            "stable composition, photorealistic, high detail."
        ),
    )
    return parser.parse_args()


def load_ground_truth_captions(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"id", "caption"}.issubset(rows[0]):
        raise ValueError("Caption CSV must contain id and caption columns.")
    return {
        normalize_video_id(row["id"]): row["caption"].strip()
        for row in rows
    }


def find_video(video_root: Path, video_id: str) -> Path:
    prefix = video_id.split("_", 1)[0]
    directories = sorted(path for path in video_root.glob(prefix + "_*") if path.is_dir())
    if len(directories) != 1:
        raise FileNotFoundError("Cannot uniquely locate video directory for {}.".format(video_id))
    candidates = (
        directories[0] / (video_id + ".mp4"),
        directories[0] / (video_id.replace("_", "-") + ".mp4"),
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError("Cannot uniquely locate video {}.".format(video_id))
    return matches[0].resolve()


def top2(
    logits: torch.Tensor, object_names: Sequence[str]
) -> Tuple[List[List[str]], torch.Tensor]:
    probabilities = logits.sigmoid().cpu()
    indices = probabilities.topk(2, dim=1).indices
    names = [
        [object_names[int(index)] for index in row]
        for row in indices
    ]
    return names, probabilities


def natural_caption(objects: Sequence[str]) -> str:
    if len(objects) != 2:
        raise ValueError("Exactly two predicted objects are required.")
    return "A realistic video showing {} and {} together.".format(
        ARTICLES[objects[0]], ARTICLES[objects[1]]
    )


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    package = torch_load(args.predictions)
    required = (
        "video_ids", "labels", "session_object_logits",
        "fused_object_logits", "object_names",
    )
    missing = [key for key in required if key not in package]
    if missing:
        raise KeyError("Prediction package is missing {}.".format(missing))
    video_ids = [normalize_video_id(value) for value in package["video_ids"]]
    object_names = list(package["object_names"])
    labels = package["labels"].float().cpu()
    if labels.shape != (len(video_ids), 6) or len(video_ids) != len(set(video_ids)):
        raise ValueError("Test IDs and labels are not uniquely aligned.")

    fused_names, fused_probabilities = top2(
        package["fused_object_logits"], object_names
    )
    session_logits = package["session_object_logits"].cpu()
    if session_logits.shape[:2] != (len(video_ids), 3):
        raise ValueError("Expected three session predictions per test video.")
    session_names, session_probabilities = [], []
    for session_index in range(3):
        names, probabilities = top2(session_logits[:, session_index], object_names)
        session_names.append(names)
        session_probabilities.append(probabilities)

    ground_truth_captions = load_ground_truth_captions(
        args.dataset_root / "Videos/structured_v2_captions.csv"
    )
    records = []
    for test_index, video_id in enumerate(video_ids):
        ground_truth_objects = [
            name for object_index, name in enumerate(object_names)
            if labels[test_index, object_index] > 0.5
        ]
        predicted_objects = fused_names[test_index]
        eeg_caption = natural_caption(predicted_objects)
        generation_prompt = "{} {}".format(eeg_caption, args.prompt_suffix.strip()).strip()
        session_records = {}
        for session_index in range(3):
            objects = session_names[session_index][test_index]
            session_records["session{}".format(session_index + 1)] = {
                "predicted_objects": objects,
                "caption": natural_caption(objects),
                "probabilities": {
                    name: float(
                        session_probabilities[session_index][test_index, object_index]
                    )
                    for object_index, name in enumerate(object_names)
                },
            }
        records.append({
            "test_index": test_index,
            "video_id": video_id,
            "original_video_path": str(
                find_video(args.dataset_root / "Videos/single_stimuli", video_id)
            ),
            "eeg_fused_predicted_objects": predicted_objects,
            "eeg_fused_probabilities": {
                name: float(fused_probabilities[test_index, object_index])
                for object_index, name in enumerate(object_names)
            },
            "eeg_caption": eeg_caption,
            "generation_prompt": generation_prompt,
            "session_predictions": session_records,
            "audit_ground_truth_objects": ground_truth_objects,
            "audit_ground_truth_caption": ground_truth_captions[video_id],
            "top2_exact": set(predicted_objects) == set(ground_truth_objects),
        })

    output_dir = args.output_dir or args.predictions.parent / "captions"
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_count = sum(record["top2_exact"] for record in records)
    result = {
        "schema_version": 1,
        "subject": package.get("subject", "unknown"),
        "split": package.get("split", "test"),
        "source_predictions": str(args.predictions.resolve()),
        "checkpoint": package.get("checkpoint"),
        "fusion": "mean of three session logits",
        "caption_source": "top-2 fused EEG object probabilities only",
        "caption_scope": "object-presence caption; no action is inferred",
        "ground_truth_fields_are_audit_only": True,
        "object_names": object_names,
        "num_videos": len(records),
        "top2_exact_count": exact_count,
        "top2_exact_accuracy": exact_count / len(records),
        "records": records,
    }
    json_path = output_dir / "test_captions.json"
    atomic_json(result, json_path)

    csv_path = output_dir / "test_captions.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "test_index", "video_id", "eeg_fused_predicted_objects",
                "eeg_caption", "generation_prompt", "top2_exact",
                "audit_ground_truth_objects", "audit_ground_truth_caption",
                "original_video_path",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "test_index": record["test_index"],
                "video_id": record["video_id"],
                "eeg_fused_predicted_objects": " ".join(
                    record["eeg_fused_predicted_objects"]
                ),
                "eeg_caption": record["eeg_caption"],
                "generation_prompt": record["generation_prompt"],
                "top2_exact": int(record["top2_exact"]),
                "audit_ground_truth_objects": " ".join(
                    record["audit_ground_truth_objects"]
                ),
                "audit_ground_truth_caption": record["audit_ground_truth_caption"],
                "original_video_path": record["original_video_path"],
            })
    temporary_csv.replace(csv_path)

    prompt_path = output_dir / "generation_prompts.txt"
    prompt_path.write_text(
        "".join(
            "{}\t{}\n".format(record["video_id"], record["generation_prompt"])
            for record in records
        ),
        encoding="utf-8",
    )
    print(
        "generated {} EEG captions; top2 exact={}/{}={:.4f}".format(
            len(records), exact_count, len(records), exact_count / len(records)
        )
    )
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("prompts:", prompt_path)


if __name__ == "__main__":
    main()
