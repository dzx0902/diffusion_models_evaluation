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

from ms_video_eval.eeg_conditioner import EEGConditioner, EEGConditionerConfig, add_condition_offset
from ms_video_eval.eeg_protocol import filter_trial_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and rank EEG-to-Wan conditions per held-out video.")
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    parser.add_argument("--sessions", nargs="+", default=["session1", "session2", "session3"])
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Evaluate only trials with this stimulus duration, e.g. 4.",
    )
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
    experiment = load_experiment(args)
    return set(experiment[f"{args.partition}_video_ids"])


def load_experiment(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    experiment = next((item for item in plan["experiments"] if item["name"] == args.experiment), None)
    if experiment is None:
        raise KeyError(f"Unknown experiment {args.experiment!r}")
    return experiment


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


def prompt_retrieval_metrics(
    predicted: torch.Tensor,
    candidate_targets: torch.Tensor,
    true_index: int,
) -> dict[str, float | int]:
    """Measure whether a prediction is closer to its caption than other captions."""
    similarities = F.cosine_similarity(predicted.unsqueeze(0), candidate_targets, dim=-1)
    true_similarity = similarities[true_index]
    rank = int((similarities > true_similarity).sum().item()) + 1
    if similarities.numel() == 1:
        margin = 0.0
    else:
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask[true_index] = False
        margin = float((true_similarity - similarities[mask].max()).item())
    return {
        "prompt_retrieval_rank": rank,
        "prompt_retrieval_top1": int(rank == 1),
        "prompt_retrieval_margin": margin,
    }


def compute_target_center(
    targets: dict[str, dict[str, Any]],
    video_ids: set[str],
) -> torch.Tensor:
    """Compute a slot-wise center from training targets without test leakage."""
    total: torch.Tensor | None = None
    count = 0
    for video_id in sorted(video_ids):
        if video_id not in targets:
            raise KeyError(f"Training target missing video ID: {video_id}")
        payload = torch.load(targets[video_id]["latent_path"], map_location="cpu", weights_only=True)
        latent = payload["latent"].float()
        if total is None:
            total = torch.zeros_like(latent)
        elif latent.shape != total.shape:
            raise ValueError(
                f"Target shape mismatch for {video_id}: {tuple(latent.shape)} != {tuple(total.shape)}"
            )
        total += latent
        count += 1
    if total is None or count == 0:
        raise ValueError("Cannot compute target center from an empty training partition")
    return total / count


def residual_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    center: torch.Tensor,
) -> dict[str, float]:
    predicted_residual = predicted - center
    target_residual = target - center
    target_energy = float(target_residual.square().mean().item())
    predicted_energy = float(predicted_residual.square().mean().item())
    mse = float((predicted_residual - target_residual).square().mean().item())
    cosine = float(
        F.cosine_similarity(
            predicted_residual.mean(0),
            target_residual.mean(0),
            dim=0,
        ).item()
    )
    return {
        "residual_mse": mse,
        "residual_mse_fraction": mse / max(target_energy, 1e-12),
        "residual_pooled_cosine": cosine,
        "residual_energy_ratio": predicted_energy / max(target_energy, 1e-12),
    }


def main() -> None:
    args = parse_args()
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        all_trials = list(csv.DictReader(handle))
    duration_trials = filter_trial_duration(all_trials, args.duration_sec)
    ids = selected_ids(args)
    trials = [row for row in duration_trials if row["video_id"] in ids and row["session"] in set(args.sessions)]
    if not trials:
        raise ValueError("No trials selected by the requested split partition and sessions")
    targets = read_targets(args.targets)
    missing = {row["video_id"] for row in trials} - set(targets)
    if missing:
        raise KeyError(f"Targets missing video IDs: {sorted(missing)[:5]}")
    experiment = load_experiment(args)
    duration_video_ids = {row["video_id"] for row in duration_trials}
    center_video_ids = set(experiment["train_video_ids"]) & duration_video_ids
    target_center = compute_target_center(targets, center_video_ids)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = EEGConditionerConfig(**checkpoint["config"])
    model = EEGConditioner(config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    condition_offset = checkpoint.get("target_mean")
    if condition_offset is not None:
        condition_offset = condition_offset.float().cpu()
    target_cache: dict[str, dict[str, Any]] = {}
    trial_metrics: list[dict[str, Any]] = []
    predicted_pooled: list[torch.Tensor] = []
    predicted_residual_pooled: list[torch.Tensor] = []

    prompt_targets: dict[str, torch.Tensor] = {}
    prompt_residual_targets: dict[str, torch.Tensor] = {}
    prompt_video_ids: dict[str, str] = {}
    for video_id in sorted({row["video_id"] for row in trials}):
        target_row = targets[video_id]
        prompt = str(target_row.get("prompt") or video_id)
        if prompt in prompt_targets:
            continue
        payload = torch.load(target_row["latent_path"], map_location="cpu", weights_only=True)
        tokens = int(payload["tokens"])
        latent = payload["latent"].float()[:tokens]
        prompt_targets[prompt] = latent.mean(0)
        prompt_residual_targets[prompt] = (latent - target_center[:tokens]).mean(0)
        prompt_video_ids[prompt] = video_id
    prompt_names = sorted(prompt_targets)
    prompt_indices = {prompt: index for index, prompt in enumerate(prompt_names)}
    candidate_targets = torch.stack([prompt_targets[prompt] for prompt in prompt_names])
    candidate_residual_targets = torch.stack(
        [prompt_residual_targets[prompt] for prompt in prompt_names]
    )

    for index, row in enumerate(trials, start=1):
        video_id = row["video_id"]
        if video_id not in target_cache:
            target_cache[video_id] = torch.load(targets[video_id]["latent_path"], map_location="cpu", weights_only=True)
        target = target_cache[video_id]
        target_latent = target["latent"].float()
        target_tokens = int(target["tokens"])
        with torch.inference_mode():
            predicted, logits = model(load_trial(row).to(device))
            predicted = add_condition_offset(predicted, condition_offset)
        predicted = predicted.squeeze(0).float().cpu()
        predicted_tokens = int(logits.argmax(dim=-1).item() + config.min_tokens)
        confidence = float(logits.softmax(dim=-1).max().item())
        valid = min(target_tokens, predicted.shape[0])
        pooled_prediction = predicted[:valid].mean(0)
        residual_prediction = predicted[:valid] - target_center[:valid]
        pooled_residual_prediction = residual_prediction.mean(0)
        cosine = float(F.cosine_similarity(pooled_prediction, target_latent[:valid].mean(0), dim=0).item())
        mse = float((predicted[:valid] - target_latent[:valid]).square().mean().item())
        metrics = {
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
        metrics.update(
            residual_metrics(
                predicted[:valid],
                target_latent[:valid],
                target_center[:valid],
            )
        )
        if condition_offset is not None:
            metrics["mean_baseline_mse"] = float(F.mse_loss(condition_offset, target_latent).item())
            metrics["mean_baseline_pooled_cosine"] = float(
                F.cosine_similarity(condition_offset.mean(0), target_latent.mean(0), dim=0).item()
            )
            metrics["mse_improvement_over_mean"] = metrics["mean_baseline_mse"] - mse
            metrics["cosine_improvement_over_mean"] = (
                cosine - metrics["mean_baseline_pooled_cosine"]
            )
        trial_metrics.append(metrics)
        predicted_pooled.append(pooled_prediction)
        predicted_residual_pooled.append(pooled_residual_prediction)
        if index % 25 == 0 or index == len(trials):
            print(f"[eeg-wan-rank] {index}/{len(trials)}", flush=True)

    for metrics, pooled_prediction, pooled_residual_prediction in zip(
        trial_metrics,
        predicted_pooled,
        predicted_residual_pooled,
        strict=True,
    ):
        prompt = str(targets[str(metrics["video_id"])].get("prompt") or metrics["video_id"])
        metrics.update(
            prompt_retrieval_metrics(
                pooled_prediction,
                candidate_targets,
                prompt_indices[prompt],
            )
        )
        residual_retrieval = prompt_retrieval_metrics(
            pooled_residual_prediction,
            candidate_residual_targets,
            prompt_indices[prompt],
        )
        metrics.update({f"residual_{key}": value for key, value in residual_retrieval.items()})
        similarities = F.cosine_similarity(
            pooled_residual_prediction.unsqueeze(0),
            candidate_residual_targets,
            dim=-1,
        )
        nearest_index = int(similarities.argmax().item())
        nearest_prompt = prompt_names[nearest_index]
        metrics["residual_nearest_video_id"] = prompt_video_ids[nearest_prompt]
        metrics["residual_nearest_prompt"] = nearest_prompt
        metrics["residual_nearest_cosine"] = float(similarities[nearest_index].item())

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
                "mean_prompt_retrieval_rank": float(
                    np.mean([row["prompt_retrieval_rank"] for row in rows])
                ),
                "prompt_retrieval_top1": float(
                    np.mean([row["prompt_retrieval_top1"] for row in rows])
                ),
                "mean_prompt_retrieval_margin": float(
                    np.mean([row["prompt_retrieval_margin"] for row in rows])
                ),
                "mean_residual_pooled_cosine": float(
                    np.mean([row["residual_pooled_cosine"] for row in rows])
                ),
                "mean_residual_mse_fraction": float(
                    np.mean([row["residual_mse_fraction"] for row in rows])
                ),
                "mean_residual_energy_ratio": float(
                    np.mean([row["residual_energy_ratio"] for row in rows])
                ),
                "mean_residual_prompt_retrieval_rank": float(
                    np.mean([row["residual_prompt_retrieval_rank"] for row in rows])
                ),
                "residual_prompt_retrieval_top1": float(
                    np.mean([row["residual_prompt_retrieval_top1"] for row in rows])
                ),
                "mean_residual_prompt_retrieval_margin": float(
                    np.mean([row["residual_prompt_retrieval_margin"] for row in rows])
                ),
            }
        )
    video_metrics.sort(
        key=lambda row: (
            -row["residual_prompt_retrieval_top1"],
            -row["mean_residual_prompt_retrieval_margin"],
            -row["mean_residual_pooled_cosine"],
            row["mean_residual_mse_fraction"],
            row["video_id"],
        )
    )
    for rank, row in enumerate(video_metrics, start=1):
        row["rank"] = rank

    representative_indices = {
        "top": 0,
        "median": len(video_metrics) // 2,
        "bottom": len(video_metrics) - 1,
    }
    representatives: dict[str, dict[str, Any]] = {}
    for label, representative_index in representative_indices.items():
        video = video_metrics[representative_index]
        video_trials = grouped[str(video["video_id"])]
        best_trial = max(
            video_trials,
            key=lambda row: (
                row["residual_prompt_retrieval_top1"],
                row["residual_prompt_retrieval_margin"],
                row["residual_pooled_cosine"],
                -row["residual_mse_fraction"],
            ),
        )
        representatives[label] = {"video": video, "best_trial": best_trial}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trial_metrics.csv", trial_metrics)
    write_csv(args.output_dir / "video_ranking.csv", video_metrics)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "experiment": args.experiment,
        "partition": args.partition,
        "sessions": args.sessions,
        "duration_sec": args.duration_sec,
        "unfiltered_trial_count": len(all_trials),
        "duration_filtered_trial_count": len(duration_trials),
        "trial_count": len(trial_metrics),
        "video_count": len(video_metrics),
        "mean_pooled_cosine": float(np.mean([row["pooled_cosine"] for row in trial_metrics])),
        "mean_valid_latent_mse": float(np.mean([row["valid_latent_mse"] for row in trial_metrics])),
        "mean_length_accuracy": float(np.mean([row["length_correct"] for row in trial_metrics])),
        "mean_prompt_retrieval_rank": float(
            np.mean([row["prompt_retrieval_rank"] for row in trial_metrics])
        ),
        "prompt_retrieval_top1": float(
            np.mean([row["prompt_retrieval_top1"] for row in trial_metrics])
        ),
        "mean_prompt_retrieval_margin": float(
            np.mean([row["prompt_retrieval_margin"] for row in trial_metrics])
        ),
        "target_center_source": "train_partition",
        "target_center_video_count": len(center_video_ids),
        "mean_residual_pooled_cosine": float(
            np.mean([row["residual_pooled_cosine"] for row in trial_metrics])
        ),
        "mean_residual_mse_fraction": float(
            np.mean([row["residual_mse_fraction"] for row in trial_metrics])
        ),
        "mean_residual_energy_ratio": float(
            np.mean([row["residual_energy_ratio"] for row in trial_metrics])
        ),
        "mean_residual_prompt_retrieval_rank": float(
            np.mean([row["residual_prompt_retrieval_rank"] for row in trial_metrics])
        ),
        "residual_prompt_retrieval_top1": float(
            np.mean([row["residual_prompt_retrieval_top1"] for row in trial_metrics])
        ),
        "mean_residual_prompt_retrieval_margin": float(
            np.mean([row["residual_prompt_retrieval_margin"] for row in trial_metrics])
        ),
        "top_videos": video_metrics[:10],
        "representatives": representatives,
    }
    if condition_offset is not None:
        summary.update({
            "mean_baseline_mse": float(np.mean([row["mean_baseline_mse"] for row in trial_metrics])),
            "mean_baseline_pooled_cosine": float(
                np.mean([row["mean_baseline_pooled_cosine"] for row in trial_metrics])
            ),
            "mean_mse_improvement_over_mean": float(
                np.mean([row["mse_improvement_over_mean"] for row in trial_metrics])
            ),
            "mean_cosine_improvement_over_mean": float(
                np.mean([row["cosine_improvement_over_mean"] for row in trial_metrics])
            ),
        })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
