"""Train EEG to retrieve centered pooled text conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_pooled_retriever import (
    EEGPooledRetriever,
    EEGPooledRetrieverConfig,
    pooled_retrieval_loss,
    positive_mask,
    retrieval_ranks,
)
from ms_video_eval.eeg_protocol import filter_trial_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--validation-partition",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume output-dir/last.pt when present; suitable for a restarted systemd unit.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--token-count", type=int, default=75)
    parser.add_argument("--sample-points", type=int, default=800)
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument(
        "--architecture",
        choices=("baseline", "multiscale"),
        default="multiscale",
    )
    parser.add_argument("--group-sessions", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--covariance-weight", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_trials(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_experiment(path: Path, name: str) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    experiment = next((row for row in plan["experiments"] if row["name"] == name), None)
    if experiment is None:
        raise KeyError(f"Unknown experiment {name!r}")
    return experiment


def split_rows(
    rows: list[dict[str, str]],
    experiment: dict[str, Any],
    partition: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_ids = set(experiment["train_video_ids"])
    valid_ids = set(experiment[f"{partition}_video_ids"])
    train_sessions = set(experiment["train_sessions"])
    valid_sessions = set(experiment[f"{partition}_sessions"])
    train = [
        row for row in rows
        if row["video_id"] in train_ids and row["session"] in train_sessions
    ]
    valid = [
        row for row in rows
        if row["video_id"] in valid_ids and row["session"] in valid_sessions
    ]
    if not train or not valid:
        raise ValueError(f"Empty split: train={len(train)}, {partition}={len(valid)}")
    overlap = {row["video_id"] for row in train} & {row["video_id"] for row in valid}
    if overlap:
        raise ValueError(f"Train/{partition} video overlap: {next(iter(overlap))}")
    return train, valid


def target_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["video_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate video_id in {path}")
    return result


def pooled_target(row: dict[str, Any]) -> torch.Tensor:
    payload = torch.load(row["latent_path"], map_location="cpu", weights_only=True)
    latent = payload["latent"].float()
    tokens = int(payload.get("tokens", latent.shape[0]))
    return latent[:tokens].mean(dim=0)


def build_prompt_bank(
    targets: dict[str, dict[str, Any]],
    video_ids: set[str],
) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    for video_id in sorted(video_ids):
        row = targets[video_id]
        prompt = str(row.get("prompt") or video_id)
        vector = pooled_target(row)
        if prompt in bank:
            torch.testing.assert_close(bank[prompt]["vector"], vector, atol=1e-5, rtol=1e-5)
            continue
        bank[prompt] = {"video_id": video_id, "vector": vector}
    if not bank:
        raise ValueError("Cannot build an empty prompt bank")
    return bank


def target_statistics(bank: dict[str, dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = torch.stack([row["vector"] for row in bank.values()])
    mean = vectors.mean(dim=0)
    residual = vectors - mean
    scale = residual.square().mean().sqrt().clamp_min(1e-6)
    return mean, scale


class PooledEEGDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        targets: dict[str, dict[str, Any]],
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.rows = rows
        self.targets = targets
        self.mean = mean
        self.scale = scale
        self.npz_cache: dict[str, Any] = {}
        self.target_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = row["npz_path"]
        if path not in self.npz_cache:
            self.npz_cache[path] = np.load(path, allow_pickle=False)
        length = int(row["length_samples"])
        signal = np.asarray(
            self.npz_cache[path]["eeg"][int(row["trial_index"]), :, :length],
            dtype=np.float32,
        )
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-6
        )
        video_id = row["video_id"]
        target_row = self.targets[video_id]
        prompt = str(target_row.get("prompt") or video_id)
        if video_id not in self.target_cache:
            self.target_cache[video_id] = (pooled_target(target_row) - self.mean) / self.scale
        return {
            "eeg": torch.from_numpy(signal),
            "target": self.target_cache[video_id],
            "prompt": prompt,
            "video_id": video_id,
            "session": row["session"],
            "trial_index": int(row["trial_index"]),
        }


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eeg": torch.stack([row["eeg"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "prompt": [row["prompt"] for row in rows],
        "video_id": [row["video_id"] for row in rows],
        "session": [row["session"] for row in rows],
        "trial_index": [row["trial_index"] for row in rows],
    }


class VideoGroupedSampler(Sampler[list[int]]):
    def __init__(self, rows: list[dict[str, str]], batch_size: int, shuffle: bool) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[row["video_id"]].append(index)
        self.groups = list(grouped.values())
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            random.shuffle(groups)
        batch: list[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(sum(map(len, self.groups)) / self.batch_size)


def make_loader(
    dataset: PooledEEGDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    grouped: bool,
    shuffle: bool,
) -> DataLoader:
    options = {
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    if grouped:
        return DataLoader(
            dataset,
            batch_sampler=VideoGroupedSampler(dataset.rows, batch_size, shuffle),
            **options,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **options)


def loss_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "temperature": args.temperature,
        "mse_weight": args.mse_weight,
        "cosine_weight": args.cosine_weight,
        "contrastive_weight": args.contrastive_weight,
        "variance_weight": args.variance_weight,
        "covariance_weight": args.covariance_weight,
    }


def evaluate(
    model: EEGPooledRetriever,
    loader: DataLoader,
    candidate_bank: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    weights: dict[str, float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    prompts = sorted(candidate_bank)
    prompt_indices = {prompt: index for index, prompt in enumerate(prompts)}
    candidates = torch.stack(
        [(candidate_bank[prompt]["vector"] - mean) / scale for prompt in prompts]
    ).to(device)
    predictions = []
    targets = []
    metadata: list[dict[str, Any]] = []
    losses: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            target = batch["target"].to(device)
            predicted = model(eeg)
            _, values = pooled_retrieval_loss(
                predicted,
                target,
                positive_mask(batch["prompt"], device),
                **weights,
            )
            losses.append({**values, "count": len(batch["prompt"])})
            predictions.append(predicted.cpu())
            targets.append(target.cpu())
            for index, prompt in enumerate(batch["prompt"]):
                metadata.append(
                    {
                        "video_id": batch["video_id"][index],
                        "session": batch["session"][index],
                        "trial_index": batch["trial_index"][index],
                        "prompt": prompt,
                    }
                )
    predicted = torch.cat(predictions).to(device)
    target = torch.cat(targets).to(device)
    true_indices = torch.tensor(
        [prompt_indices[row["prompt"]] for row in metadata],
        dtype=torch.long,
        device=device,
    )
    ranks, similarities = retrieval_ranks(predicted, candidates, true_indices)
    nearest = similarities.argmax(dim=1)
    true_similarities = similarities.gather(1, true_indices[:, None]).squeeze(1)
    other_similarities = similarities.clone()
    other_similarities.scatter_(1, true_indices[:, None], float("-inf"))
    retrieval_margin = true_similarities - other_similarities.max(dim=1).values
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
    mse = (predicted - target).square().mean(dim=-1)
    energy = predicted.square().mean(dim=-1) / target.square().mean(dim=-1).clamp_min(1e-12)
    total_count = sum(int(row["count"]) for row in losses)
    metrics = {
        key: sum(row[key] * int(row["count"]) for row in losses) / total_count
        for key in (
            "loss",
            "mse",
            "cosine_loss",
            "contrastive_loss",
            "variance_loss",
            "covariance_loss",
        )
    }
    metrics.update(
        {
            "recall_at_1": float((ranks <= 1).float().mean().item()),
            "recall_at_5": float((ranks <= 5).float().mean().item()),
            "mrr": float((1.0 / ranks).mean().item()),
            "mean_rank": float(ranks.mean().item()),
            "mean_cosine": float(cosine.mean().item()),
            "mean_standardized_mse": float(mse.mean().item()),
            "mean_energy_ratio": float(energy.mean().item()),
            "candidate_prompts": len(prompts),
            "chance_recall_at_1": 1.0 / len(prompts),
            "chance_recall_at_5": min(5.0 / len(prompts), 1.0),
        }
    )
    trial_rows = []
    for index, row in enumerate(metadata):
        nearest_prompt = prompts[int(nearest[index].item())]
        trial_rows.append(
            {
                **row,
                "rank": float(ranks[index].item()),
                "reciprocal_rank": float((1.0 / ranks[index]).item()),
                "cosine": float(cosine[index].item()),
                "standardized_mse": float(mse[index].item()),
                "energy_ratio": float(energy[index].item()),
                "nearest_prompt": nearest_prompt,
                "nearest_video_id": candidate_bank[nearest_prompt]["video_id"],
                "nearest_cosine": float(similarities[index, nearest[index]].item()),
                "retrieval_margin": float(retrieval_margin[index].item()),
            }
        )
    return metrics, trial_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    all_trials = read_trials(args.trials)
    filtered = filter_trial_duration(all_trials, args.duration_sec)
    experiment = load_experiment(args.split_plan, args.experiment)
    train_rows, valid_rows = split_rows(filtered, experiment, args.validation_partition)
    targets = target_index(args.targets)
    selected_ids = {row["video_id"] for row in train_rows + valid_rows}
    missing = selected_ids - set(targets)
    if missing:
        raise KeyError(f"Targets missing video IDs: {sorted(missing)[:5]}")
    train_bank = build_prompt_bank(targets, {row["video_id"] for row in train_rows})
    valid_bank = build_prompt_bank(targets, {row["video_id"] for row in valid_rows})
    mean, scale = target_statistics(train_bank)
    config = EEGPooledRetrieverConfig(
        sample_points=args.sample_points,
        sampling_rate=args.sampling_rate,
        hidden_dim=args.hidden_dim,
        target_dim=mean.numel(),
        token_count=args.token_count,
        encoder_layers=args.encoder_layers,
        heads=args.heads,
        dropout=args.dropout,
        architecture=args.architecture,
    )
    model = EEGPooledRetriever(config).to(device)
    train_dataset = PooledEEGDataset(train_rows, targets, mean, scale)
    valid_dataset = PooledEEGDataset(valid_rows, targets, mean, scale)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        device,
        args.group_sessions,
        True,
    )
    valid_loader = make_loader(
        valid_dataset,
        args.batch_size,
        args.workers,
        device,
        args.group_sessions,
        False,
    )
    weights = loss_kwargs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "duration_sec": args.duration_sec,
        "experiment": args.experiment,
        "validation_partition": args.validation_partition,
        "train_trials": len(train_rows),
        "validation_trials": len(valid_rows),
        "train_videos": len({row["video_id"] for row in train_rows}),
        "validation_videos": len({row["video_id"] for row in valid_rows}),
        "train_prompts": len(train_bank),
        "validation_prompts": len(valid_bank),
        "group_sessions": args.group_sessions,
    }
    print(f"[eeg-pooled] protocol={json.dumps(protocol)}", flush=True)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        checkpoint_config = EEGPooledRetrieverConfig(**checkpoint["config"])
        model = EEGPooledRetriever(checkpoint_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        if not torch.allclose(checkpoint["target_mean"].cpu(), mean, atol=1e-6, rtol=1e-6):
            raise ValueError("Checkpoint target mean differs from this training split")
        if not torch.allclose(checkpoint["target_scale"].cpu(), scale, atol=1e-6, rtol=1e-6):
            raise ValueError("Checkpoint target scale differs from this training split")
        metrics, rows = evaluate(model, valid_loader, valid_bank, mean, scale, device, weights)
        result = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "protocol": protocol,
            "metrics": metrics,
        }
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(args.output_dir / "trial_metrics.csv", rows)
        print(f"[eeg-pooled] evaluation={json.dumps(metrics)}", flush=True)
        return

    history_path = args.output_dir / "history.jsonl"
    if args.resume is not None and args.auto_resume:
        raise ValueError("Use either --resume or --auto-resume, not both")
    resume_path = args.resume
    automatic_checkpoint = args.output_dir / "last.pt"
    if args.auto_resume and automatic_checkpoint.exists():
        resume_path = automatic_checkpoint
    if history_path.exists() and resume_path is None:
        raise FileExistsError(f"History exists; pass --resume: {history_path}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 1
    best_mrr = float("-inf")
    stale_epochs = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if EEGPooledRetrieverConfig(**checkpoint["config"]) != config:
            raise ValueError("Resume model configuration differs")
        if checkpoint.get("protocol") != protocol:
            raise ValueError("Resume data protocol differs")
        if checkpoint.get("loss_weights") != weights:
            raise ValueError("Resume loss weights differ")
        if not torch.allclose(checkpoint["target_mean"].cpu(), mean, atol=1e-6, rtol=1e-6):
            raise ValueError("Resume target mean differs")
        if not torch.allclose(checkpoint["target_scale"].cpu(), scale, atol=1e-6, rtol=1e-6):
            raise ValueError("Resume target scale differs")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_mrr = float(checkpoint["early_stopping"]["best_mrr"])
        stale_epochs = int(checkpoint["early_stopping"]["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[eeg-pooled] resumed epoch {start_epoch - 1}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch["eeg"].to(device))
            loss, _ = pooled_retrieval_loss(
                predicted,
                batch["target"].to(device),
                positive_mask(batch["prompt"], device),
                **weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        valid, _ = evaluate(model, valid_loader, valid_bank, mean, scale, device, weights)
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "valid": valid}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        improved = valid["mrr"] > best_mrr + args.early_stop_min_delta
        if improved:
            best_mrr = valid["mrr"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = {
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "valid": valid,
            "target_mean": mean,
            "target_scale": scale,
            "protocol": protocol,
            "loss_weights": weights,
            "early_stopping": {
                "metric": "mrr",
                "best_mrr": best_mrr,
                "stale_epochs": stale_epochs,
                "patience": args.early_stop_patience,
                "min_delta": args.early_stop_min_delta,
            },
        }
        atomic_save(payload, args.output_dir / "last.pt")
        if improved:
            atomic_save(payload, args.output_dir / "best.pt")
        print(f"[eeg-pooled] epoch={epoch} valid={json.dumps(valid)}", flush=True)
        if (
            epoch >= args.min_epochs
            and args.early_stop_patience > 0
            and stale_epochs >= args.early_stop_patience
        ):
            print(f"[eeg-pooled] early stop at epoch {epoch}", flush=True)
            break
    print(f"[eeg-pooled] best checkpoint: {args.output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
