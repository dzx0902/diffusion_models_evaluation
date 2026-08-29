"""Evaluate one Compact EEG checkpoint on its untouched held-out test videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    CompactEEGClassifier,
    ThreeSessionDataset,
    evaluate,
    load_label_package,
    load_three_sessions,
    pair_label_matrix,
    resolve_subject,
    save_json,
    torch_load,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/userhome2/zhoutianyi/Dataset/Multi-Object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--label-package", type=Path,
        default=PROJECT_ROOT / "data/video_multilabels_2object.pt",
    )
    parser.add_argument("--subject", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    checkpoint = torch_load(args.checkpoint)
    package = load_label_package(args.label_package)
    if list(package["ids"]) != list(checkpoint["ids"]):
        raise ValueError("Checkpoint and label package have different video order.")

    subject = args.subject or checkpoint["subject"]
    subject_dir = resolve_subject(args.dataset_root / "Subjects", subject)
    if subject_dir.name != checkpoint["subject"]:
        raise ValueError(
            "Checkpoint normalization belongs to {!r}, not {!r}. Train a subject-specific model.".format(
                checkpoint["subject"], subject_dir.name
            )
        )
    eeg, _ = load_three_sessions(
        subject_dir, package["ids"], int(checkpoint["num_samples"])
    )
    test_indices = checkpoint["splits"]["test"]
    if len(test_indices) != 96:
        print("warning: checkpoint does not use the released 96-video test20 split")
    dataset = ThreeSessionDataset(
        eeg=eeg,
        labels=package["labels"],
        pair_indices=package["pair_indices"],
        cardinalities=package["cardinalities"],
        ids=package["ids"],
        indices=test_indices,
        mean=checkpoint["normalization_mean"],
        std=checkpoint["normalization_std"],
        eeg_scale=float(checkpoint["eeg_scale"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = CompactEEGClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    metrics, predictions = evaluate(
        model, loader, device, pair_label_matrix(package), package["object_names"]
    )

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "subject": subject_dir.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_ap": float(checkpoint["best_validation_macro_ap"]),
        "split": "test",
        "num_test_videos": len(dataset),
        "fusion": "mean of session1/session2/session3 logits; raw EEG is not averaged",
        "metrics": metrics,
        "chance": {
            "top2_exact_set_accuracy": 1.0 / 15.0,
            "pair_head_accuracy": 1.0 / 6.0,
        },
    }
    save_json(report, output_dir / "test_metrics.json")
    predictions.update({
        "schema_version": 1,
        "metrics": metrics,
        "object_names": package["object_names"],
        "pair_prefixes": package["pair_prefixes"],
        "pair_names": package["pair_names"],
        "checkpoint": str(args.checkpoint.resolve()),
        "subject": subject_dir.name,
        "split": "test",
    })
    prediction_path = output_dir / "test_predictions.pt"
    temporary = prediction_path.with_suffix(".pt.tmp")
    torch.save(predictions, temporary)
    temporary.replace(prediction_path)

    for name in ("session1", "session2", "session3", "fused"):
        values = metrics[name]
        print(
            "{}: mAP={:.4f} top2_exact={:.4f} recall={:.4f} pair={:.4f}".format(
                name, values["macro_ap"], values["top2_exact_set_accuracy"],
                values["top2_recall"], values["pair_head_accuracy"],
            )
        )
    print("predictions:", prediction_path)
    print("metrics:", output_dir / "test_metrics.json")


if __name__ == "__main__":
    main()
