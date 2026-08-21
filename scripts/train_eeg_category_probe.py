"""Train a six-class EEG diagnostic on held-out video splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_classifiers import build_eeg_classifier
from ms_video_eval.eeg_conditioner import EEGCategoryProbe, EEGConditionerConfig
from ms_video_eval.eeg_protocol import filter_trial_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a subject-dependent EEG category probe.")
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("mlp", "eegnet", "shallownet", "deepnet", "tsconv", "conformer", "multiscale"),
        default="multiscale",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--sample-points", type=int, default=800)
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def category_from_video_id(video_id: str) -> str:
    parts = video_id.replace("_", "-").split("-", maxsplit=1)
    if len(parts) != 2 or len(parts[0]) != 2 or not parts[0].isdigit():
        raise ValueError(f"Invalid video_id: {video_id!r}")
    return parts[0]


def read_split(path: Path, experiment: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = next((row for row in payload["experiments"] if row["name"] == experiment), None)
    if result is None:
        raise KeyError(f"Unknown experiment {experiment!r}")
    return result


class EEGCategoryDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], class_to_index: dict[str, int]) -> None:
        self.rows = rows
        self.class_to_index = class_to_index
        self.cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = row["npz_path"]
        if path not in self.cache:
            self.cache[path] = np.load(path, allow_pickle=False)
        length = int(row["length_samples"])
        eeg = np.asarray(
            self.cache[path]["eeg"][int(row["trial_index"]), :, :length],
            dtype=np.float32,
        )
        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-6)
        label = self.class_to_index[category_from_video_id(row["video_id"])]
        return torch.from_numpy(eeg), torch.tensor(label, dtype=torch.long)


def classification_metrics(
    predictions: list[int],
    labels: list[int],
    classes: int,
) -> dict[str, Any]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for target, predicted in zip(labels, predictions, strict=True):
        confusion[target, predicted] += 1
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    recalls = [
        float(confusion[index, index] / confusion[index].sum())
        if confusion[index].sum()
        else 0.0
        for index in range(classes)
    ]
    return {
        "accuracy": accuracy,
        "macro_accuracy": float(np.mean(recalls)),
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: int,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for eeg, target in loader:
            logits = model(eeg.to(device))
            loss_sum += float(nn.functional.cross_entropy(logits, target.to(device), reduction="sum"))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(target.tolist())
    metrics = classification_metrics(predictions, labels, classes)
    metrics["loss"] = loss_sum / len(labels)
    metrics["samples"] = len(labels)
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = filter_trial_duration(list(csv.DictReader(handle)), args.duration_sec)
    split = read_split(args.split_plan, args.experiment)
    partitions = {
        name: [row for row in rows if row["video_id"] in set(split[f"{name}_video_ids"])]
        for name in ("train", "validation", "test")
    }
    categories = sorted({category_from_video_id(row["video_id"]) for row in rows})
    if categories != ["01", "02", "03", "04", "05", "06"]:
        raise ValueError(f"Expected 4-second categories 01..06, got {categories}")
    if any(not partition for partition in partitions.values()):
        raise ValueError("Split produced an empty category-probe partition")
    class_to_index = {category: index for index, category in enumerate(categories)}
    loaders = {
        name: DataLoader(
            EEGCategoryDataset(partition, class_to_index),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for name, partition in partitions.items()
    }

    config = EEGConditionerConfig(
        sample_points=args.sample_points,
        hidden_dim=args.hidden_dim,
        slots=1,
        latent_dim=args.hidden_dim,
        token_count=40,
        encoder_layers=args.encoder_layers,
        decoder_layers=1,
        min_tokens=1,
        max_tokens=1,
        architecture="multiscale",
        sampling_rate=args.sampling_rate,
    )
    if args.model == "multiscale":
        model = EEGCategoryProbe(config, len(categories))
    else:
        model = build_eeg_classifier(
            args.model,
            len(categories),
            channels=config.channels,
            samples=args.sample_points,
            sampling_rate=args.sampling_rate,
        )
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    if history_path.exists() and args.resume is None:
        raise FileExistsError(f"History exists; pass --resume or use another output: {history_path}")

    start_epoch = 1
    best = float("-inf")
    stale = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["early_stopping"]["best"])
        stale = int(checkpoint["early_stopping"]["stale_epochs"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for eeg, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(eeg.to(device)), target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        validation = evaluate(model, loaders["validation"], device, len(categories))
        improved = validation["macro_accuracy"] > best + args.early_stop_min_delta
        if improved:
            best = float(validation["macro_accuracy"])
            stale = 0
        else:
            stale += 1
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "validation": validation}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "config": asdict(config),
            "model": args.model,
            "parameter_count": parameter_count,
            "classes": categories,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "early_stopping": {"best": best, "stale_epochs": stale},
        }
        torch.save(payload, args.output_dir / "last.pt")
        if improved:
            torch.save(payload, args.output_dir / "best.pt")
        print(f"[eeg-category] epoch={epoch} validation={validation}", flush=True)
        if epoch >= args.min_epochs and stale >= args.early_stop_patience:
            print(f"[eeg-category] early stop at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test = evaluate(model, loaders["test"], device, len(categories))
    report = {
        "model": args.model,
        "parameter_count": parameter_count,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "categories": categories,
        "chance_accuracy": 1.0 / len(categories),
        "partition_trials": {name: len(partition) for name, partition in partitions.items()},
        "validation": checkpoint["validation"],
        "test": test,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[eeg-category] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
