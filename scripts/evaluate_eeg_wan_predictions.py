"""Rank held-out EEG-to-Wan predictions without loading or running Wan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditioner, EEGConditionerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and rank EEG-to-Wan conditions per held-out video.")
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    parser.add_argument("--sessions", nargs="+", default=["session1", "session2", "session3"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_targets(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["video_id"])] = row
    return result


def selected_ids(args: argparse.Namespace) -> set[str]:
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    experiment = next((item for item in plan["experiments"] if item["name"] == args.experiment), None)
    if experiment is None:
        raise KeyError(f"Unknown experiment {args.experiment!r}")
    return set(experiment[f"{args.partition}_video_ids"])


def load_trial(row: dict[str, str]) -> torch.Tensor:
    with np.load(row["npz_path"], allow_pickle=False) as payload:
        length = int(row["length_samples"])
        signal = np.asarray(payload["eeg"][int(row["trial_index"]), :, :length], dtype=np.float32)
    signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-6)
    return torch.from_numpy(signal).unsqueeze(0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        all_trials = list(csv.DictReader(handle))
    ids = selected_ids(args)
    trials = [row for row in all_trials if row["video_id"] in ids and row["session"] in set(args.sessions)]
    if not trials:
        raise ValueError("No trials selected by the requested split partition and sessions")
    targets = read_targets(args.targets)
    missing = {row["video_id"] for row in trials} - set(targets)
    if missing:
        raise KeyError(f"Targets missing video IDs: {sorted(missing)[:5]}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = EEGConditionerConfig(**checkpoint["config"])
    model = EEGConditioner(config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    target_cache: dict[str, dict[str, Any]] = {}
    trial_metrics: list[dict[str, Any]] = []

    for index, row in enumerate(trials, start=1):
        video_id = row["video_id"]
        if video_id not in target_cache:
            target_cache[video_id] = torch.load(targets[video_id]["latent_path"], map_location="cpu", weights_only=True)
        target = target_cache[video_id]
        target_latent = target["latent"].float()
        target_tokens = int(target["tokens"])
        with torch.inference_mode():
            predicted, logits = model(load_trial(row).to(device))
        predicted = predicted.squeeze(0).float().cpu()
        predicted_tokens = int(logits.argmax(dim=-1).item() + config.min_tokens)
        confidence = float(logits.softmax(dim=-1).max().item())
        valid = min(target_tokens, predicted.shape[0])
        cosine = float(F.cosine_similarity(predicted[:valid].mean(0), target_latent[:valid].mean(0), dim=0).item())
        mse = float((predicted[:valid] - target_latent[:valid]).square().mean().item())
        trial_metrics.append(
            {
                "video_id": video_id,
                "session": row["session"],
                "trial_index": int(row["trial_index"]),
                "target_tokens": target_tokens,
                "predicted_tokens": predicted_tokens,
                "length_error": abs(predicted_tokens - target_tokens),
                "length_correct": int(predicted_tokens == target_tokens),
                "length_confidence": confidence,
                "valid_latent_mse": mse,
                "pooled_cosine": cosine,
            }
        )
        if index % 25 == 0 or index == len(trials):
            print(f"[eeg-wan-rank] {index}/{len(trials)}", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trial_metrics:
        grouped[str(row["video_id"])].append(row)
    video_metrics = []
    for video_id, rows in grouped.items():
        video_metrics.append(
            {
                "video_id": video_id,
                "trial_count": len(rows),
                "mean_pooled_cosine": float(np.mean([row["pooled_cosine"] for row in rows])),
                "min_pooled_cosine": float(np.min([row["pooled_cosine"] for row in rows])),
                "mean_valid_latent_mse": float(np.mean([row["valid_latent_mse"] for row in rows])),
                "mean_length_error": float(np.mean([row["length_error"] for row in rows])),
                "length_accuracy": float(np.mean([row["length_correct"] for row in rows])),
                "mean_length_confidence": float(np.mean([row["length_confidence"] for row in rows])),
            }
        )
    video_metrics.sort(key=lambda row: (-row["mean_pooled_cosine"], row["mean_valid_latent_mse"], row["video_id"]))
    for rank, row in enumerate(video_metrics, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trial_metrics.csv", trial_metrics)
    write_csv(args.output_dir / "video_ranking.csv", video_metrics)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "experiment": args.experiment,
        "partition": args.partition,
        "sessions": args.sessions,
        "trial_count": len(trial_metrics),
        "video_count": len(video_metrics),
        "mean_pooled_cosine": float(np.mean([row["pooled_cosine"] for row in trial_metrics])),
        "mean_valid_latent_mse": float(np.mean([row["valid_latent_mse"] for row in trial_metrics])),
        "mean_length_accuracy": float(np.mean([row["length_correct"] for row in trial_metrics])),
        "top_videos": video_metrics[:10],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
