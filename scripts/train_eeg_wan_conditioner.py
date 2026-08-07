"""Train raw EEG -> fixed text-condition latents.

The split is session based by default. A video must never be joined by row order:
the trial manifest carries the NPZ trial_index and the target manifest is keyed
by video_id.
"""

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
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditioner, EEGConditionerConfig, fixed_pca_loss
from ms_video_eval.eeg_protocol import filter_trial_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EEG-to-video fixed text-condition model.")
    parser.add_argument("--trials", type=Path, required=True, help="eeg_trials.csv from build_eeg_video_manifest.py")
    parser.add_argument("--targets", type=Path, required=True, help="video_id -> fixed condition target JSONL")
    parser.add_argument("--train-sessions", nargs="+", default=["session1", "session2"])
    parser.add_argument("--val-sessions", nargs="+", default=["session3"])
    parser.add_argument("--split-plan", type=Path, default=None, help="Six-fold plan from build_eeg_split_plans.py.")
    parser.add_argument("--experiment", default=None, help="Experiment name in --split-plan, e.g. video_6fold_1.")
    parser.add_argument("--validation-partition", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "eeg_wan" / "conditioner")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Evaluate this checkpoint only; do not train.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from last.pt after an interruption.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs; early stopping may finish sooner.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--padding-weight", type=float, default=0.1)
    parser.add_argument("--length-weight", type=float, default=0.2)
    parser.add_argument("--pooled-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument(
        "--selection-metric",
        choices=["loss", "valid_mse", "pooled_cosine_loss"],
        default="loss",
        help="Validation metric used for best.pt and early stopping (lower is better).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop after this many unimproved epochs; use 0 to disable.",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--sample-points", type=int, default=800)
    parser.add_argument("--architecture", choices=["baseline", "multiscale"], default="baseline")
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument(
        "--group-sessions",
        action="store_true",
        help="Keep all available sessions of a video in the same batch for multi-positive contrastive loss.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Train and validate only trials with this stimulus duration, e.g. 4.",
    )
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=None, help="Infer from targets when omitted.")
    parser.add_argument("--min-tokens", type=int, default=None, help="Defaults to the minimum training-target length.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Defaults to the maximum training-target length.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_targets(path: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["video_id"] in targets:
            raise ValueError(f"duplicate target video_id: {row['video_id']}")
        targets[row["video_id"]] = row
    return targets


def resolve_split(args: argparse.Namespace) -> tuple[set[str], set[str], set[str] | None, set[str] | None]:
    if args.split_plan is None:
        return set(args.train_sessions), set(args.val_sessions), None, None
    if not args.experiment:
        raise ValueError("--experiment is required when --split-plan is supplied")
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    experiment = next((item for item in plan["experiments"] if item["name"] == args.experiment), None)
    if experiment is None:
        raise KeyError(f"Unknown experiment {args.experiment!r} in {args.split_plan}")
    partition = args.validation_partition
    return (
        set(experiment["train_sessions"]),
        set(experiment[f"{partition}_sessions"]),
        set(experiment["train_video_ids"]),
        set(experiment[f"{partition}_video_ids"]),
    )


def history_early_stop_state(path: Path, metric: str, min_delta: float) -> tuple[float, int]:
    if not path.exists():
        return float("inf"), 0
    values = [
        float(json.loads(line)["valid"][metric])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best = float("inf")
    stale_epochs = 0
    for value in values:
        if value < best - min_delta:
            best = value
            stale_epochs = 0
        else:
            stale_epochs += 1
    return best, stale_epochs


class EEGWanDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], targets: dict[str, dict[str, Any]], slots: int, latent_dim: int) -> None:
        self.rows = rows
        self.targets = targets
        self.slots = slots
        self.latent_dim = latent_dim
        self.npz_cache: dict[str, Any] = {}
        self.target_cache: dict[str, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _npz(self, path: str) -> Any:
        if path not in self.npz_cache:
            self.npz_cache[path] = np.load(path, allow_pickle=False)
        return self.npz_cache[path]

    def _target(self, video_id: str) -> dict[str, torch.Tensor]:
        if video_id not in self.target_cache:
            payload = torch.load(self.targets[video_id]["latent_path"], map_location="cpu", weights_only=True)
            latent = payload["latent"].float()
            tokens = int(payload["tokens"])
            if latent.shape != (self.slots, self.latent_dim) or not 1 <= tokens <= self.slots:
                raise ValueError(f"Invalid fixed target for {video_id}: {tuple(latent.shape)}, tokens={tokens}")
            self.target_cache[video_id] = {"latent": latent, "tokens": torch.tensor(tokens, dtype=torch.long)}
        return self.target_cache[video_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload = self._npz(row["npz_path"])
        length = int(row["length_samples"])
        signal = np.asarray(payload["eeg"][int(row["trial_index"]), :, :length], dtype=np.float32)
        # Per-trial/channel normalization avoids fitting statistics on validation sessions.
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-6)
        target = self._target(row["video_id"])
        return {"eeg": torch.from_numpy(signal), **target, "video_id": row["video_id"]}


class VideoGroupedBatchSampler(Sampler[list[int]]):
    def __init__(self, rows: list[dict[str, str]], batch_size: int, shuffle: bool) -> None:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["video_id"], []).append(index)
        if any(len(indices) > batch_size for indices in groups.values()):
            raise ValueError("--batch-size is smaller than a complete video session group")
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):  # type: ignore[no-untyped-def]
        groups = list(self.groups)
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
        count = 0
        size = 0
        for group in self.groups:
            if size and size + len(group) > self.batch_size:
                count += 1
                size = 0
            size += len(group)
        return count + int(size > 0)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["eeg"].shape[-1] for item in batch)
    eeg = torch.zeros(len(batch), batch[0]["eeg"].shape[0], max_length)
    for index, item in enumerate(batch):
        eeg[index, :, : item["eeg"].shape[-1]] = item["eeg"]
    return {
        "eeg": eeg,
        "latent": torch.stack([item["latent"] for item in batch]),
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "video_id": [item["video_id"] for item in batch],
    }


def evaluate(
    model: EEGConditioner,
    loader: DataLoader,
    device: torch.device,
    *,
    padding_weight: float,
    length_weight: float,
    pooled_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
) -> dict[str, float]:
    model.eval()
    aggregate = {"loss": 0.0, "valid_mse": 0.0, "padding_mse": 0.0, "length_ce": 0.0, "pooled_cosine_loss": 0.0, "contrastive_loss": 0.0, "length_accuracy": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            predicted, logits = model(batch["eeg"].to(device))
            lengths = batch["tokens"].to(device)
            _, values = fixed_pca_loss(
                predicted,
                batch["latent"].to(device),
                lengths,
                logits,
                model.config.min_tokens,
                padding_weight=padding_weight,
                length_weight=length_weight,
                pooled_weight=pooled_weight,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
                positive_mask=video_positive_mask(batch["video_id"], device),
            )
            size = len(lengths)
            for key in values:
                aggregate[key] += values[key] * size
            aggregate["length_accuracy"] += float((logits.argmax(dim=-1) + model.config.min_tokens == lengths).sum())
            count += size
    return {key: value / count for key, value in aggregate.items()}


def video_positive_mask(video_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in video_ids] for left in video_ids],
        device=device,
        dtype=torch.bool,
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None and args.resume is not None:
        raise ValueError("--checkpoint and --resume cannot be used together")
    if min(args.padding_weight, args.length_weight, args.pooled_weight, args.contrastive_weight) < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.contrastive_temperature <= 0:
        raise ValueError("--contrastive-temperature must be positive")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0 or args.min_epochs < 1:
        raise ValueError("Invalid early-stopping configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = filter_trial_duration(all_rows, args.duration_sec)
    if not rows:
        raise ValueError(f"No trials remain after --duration-sec={args.duration_sec}")
    targets = read_targets(args.targets)
    train_sessions, val_sessions, train_ids, val_ids = resolve_split(args)
    train_rows = [
        row for row in rows
        if row["session"] in train_sessions and (train_ids is None or row["video_id"] in train_ids)
    ]
    val_rows = [
        row for row in rows
        if row["session"] in val_sessions and (val_ids is None or row["video_id"] in val_ids)
    ]
    if not train_rows or not val_rows:
        raise ValueError("Session split produced an empty train or validation set")
    selected_ids = {row["video_id"] for row in train_rows + val_rows}
    unknown = selected_ids - set(targets)
    if unknown:
        raise KeyError(f"Targets missing {len(unknown)} video_ids, e.g. {sorted(unknown)[:5]}")
    train_lengths = [int(targets[row["video_id"]]["tokens"]) for row in train_rows]
    min_tokens = args.min_tokens if args.min_tokens is not None else min(train_lengths)
    max_tokens = args.max_tokens if args.max_tokens is not None else max(train_lengths)
    if not 1 <= min_tokens <= max_tokens <= 128:
        raise ValueError(f"Invalid token range [{min_tokens}, {max_tokens}]")
    all_lengths = [int(targets[video_id]["tokens"]) for video_id in selected_ids]
    if min(all_lengths) < min_tokens or max(all_lengths) > max_tokens:
        raise ValueError(
            f"Target lengths [{min(all_lengths)}, {max(all_lengths)}] exceed classifier range "
            f"[{min_tokens}, {max_tokens}]. Pass --min-tokens/--max-tokens explicitly."
        )
    example_id = next(iter(selected_ids))
    example_payload = torch.load(targets[example_id]["latent_path"], map_location="cpu", weights_only=True)
    target_shape = tuple(example_payload["latent"].shape)
    if len(target_shape) != 2 or target_shape[0] != args.slots:
        raise ValueError(f"Expected target shape [{args.slots}, latent_dim], got {target_shape}")
    latent_dim = args.latent_dim if args.latent_dim is not None else int(target_shape[1])
    if target_shape[1] != latent_dim:
        raise ValueError(f"Target latent_dim={target_shape[1]} differs from --latent-dim={latent_dim}")
    config = EEGConditionerConfig(
        hidden_dim=args.hidden_dim, encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers, sample_points=args.sample_points,
        slots=args.slots, latent_dim=latent_dim, min_tokens=min_tokens, max_tokens=max_tokens,
        architecture=args.architecture, sampling_rate=args.sampling_rate,
    )
    model = EEGConditioner(config).to(device)
    data_protocol = {
        "duration_sec": args.duration_sec,
        "all_trial_count": len(all_rows),
        "duration_filtered_trial_count": len(rows),
        "train_trial_count": len(train_rows),
        "validation_trial_count": len(val_rows),
        "train_video_count": len({row["video_id"] for row in train_rows}),
        "validation_video_count": len({row["video_id"] for row in val_rows}),
        "group_sessions": args.group_sessions,
    }
    loss_weights = {
        "padding_weight": args.padding_weight,
        "length_weight": args.length_weight,
        "pooled_weight": args.pooled_weight,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_temperature": args.contrastive_temperature,
    }
    print(
        f"[eeg-wan] trials total={len(all_rows)} duration_filtered={len(rows)} "
        f"train={len(train_rows)} validation={len(val_rows)} duration_sec={args.duration_sec}",
        flush=True,
    )
    train_dataset = EEGWanDataset(train_rows, targets, args.slots, latent_dim)
    val_dataset = EEGWanDataset(val_rows, targets, args.slots, latent_dim)
    loader_kwargs = {
        "num_workers": args.workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    if args.group_sessions:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=VideoGroupedBatchSampler(train_rows, args.batch_size, shuffle=True),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=VideoGroupedBatchSampler(val_rows, args.batch_size, shuffle=False),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        checkpoint_config = EEGConditionerConfig(**checkpoint["config"])
        model = EEGConditioner(checkpoint_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        result = evaluate(
            model,
            val_loader,
            device,
            padding_weight=args.padding_weight,
            length_weight=args.length_weight,
            pooled_weight=args.pooled_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
        )
        print(f"[eeg-wan] checkpoint={args.checkpoint} evaluation={result}")
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    if history_path.exists() and args.resume is None:
        raise FileExistsError(
            f"Training history already exists; pass --resume or choose a new --output-dir: {history_path}"
        )
    best, stale_epochs = history_early_stop_state(
        history_path,
        args.selection_metric,
        args.early_stop_min_delta,
    )
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_config = EEGConditionerConfig(**checkpoint["config"])
        if checkpoint_config != config:
            raise ValueError(f"Resume configuration differs from requested configuration: {checkpoint_config} != {config}")
        checkpoint_protocol = checkpoint.get("data_protocol")
        if checkpoint_protocol is not None and checkpoint_protocol != data_protocol:
            raise ValueError(
                "Resume data protocol differs from the current selection: "
                f"{checkpoint_protocol} != {data_protocol}"
            )
        checkpoint_weights = checkpoint.get("loss_weights")
        if checkpoint_weights is not None and checkpoint_weights != loss_weights:
            raise ValueError(
                "Resume loss weights differ from the current objective: "
                f"{checkpoint_weights} != {loss_weights}"
            )
        checkpoint_early_stopping = checkpoint.get("early_stopping")
        if checkpoint_early_stopping is not None:
            if checkpoint_early_stopping["metric"] != args.selection_metric:
                raise ValueError(
                    "Resume selection metric differs from checkpoint: "
                    f"{checkpoint_early_stopping['metric']} != {args.selection_metric}"
                )
            best = float(checkpoint_early_stopping["best"])
            stale_epochs = int(checkpoint_early_stopping["stale_epochs"])
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if "optimizer_state_dict" in checkpoint and "scheduler_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"[eeg-wan] resumed full optimizer state from epoch {start_epoch - 1}")
        else:
            # Compatibility with checkpoints created before resume support.
            for _ in range(start_epoch - 1):
                scheduler.step()
            print(f"[eeg-wan] resumed model-only checkpoint from epoch {start_epoch - 1}")
    if start_epoch > args.epochs:
        print(f"[eeg-wan] checkpoint already reached epoch {start_epoch - 1}; nothing to train")
        return
    if (
        args.early_stop_patience > 0
        and start_epoch - 1 >= args.min_epochs
        and stale_epochs >= args.early_stop_patience
    ):
        print(
            f"[eeg-wan] checkpoint already satisfies early stopping at epoch {start_epoch - 1}; "
            "nothing to train",
            flush=True,
        )
        return
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted, logits = model(batch["eeg"].to(device))
            loss, _ = fixed_pca_loss(
                predicted,
                batch["latent"].to(device),
                batch["tokens"].to(device),
                logits,
                config.min_tokens,
                padding_weight=args.padding_weight,
                length_weight=args.length_weight,
                pooled_weight=args.pooled_weight,
                contrastive_weight=args.contrastive_weight,
                contrastive_temperature=args.contrastive_temperature,
                positive_mask=video_positive_mask(batch["video_id"], device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        valid = evaluate(
            model,
            val_loader,
            device,
            padding_weight=args.padding_weight,
            length_weight=args.length_weight,
            pooled_weight=args.pooled_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
        )
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "valid": valid}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[eeg-wan] epoch={epoch} valid={valid}", flush=True)
        selection_value = float(valid[args.selection_metric])
        improved = selection_value < best - args.early_stop_min_delta
        if improved:
            best = selection_value
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
            "args": vars(args),
            "data_protocol": data_protocol,
            "loss_weights": loss_weights,
            "early_stopping": {
                "metric": args.selection_metric,
                "best": best,
                "stale_epochs": stale_epochs,
                "patience": args.early_stop_patience,
                "min_delta": args.early_stop_min_delta,
            },
        }
        torch.save(payload, args.output_dir / "last.pt")
        if improved:
            torch.save(payload, args.output_dir / "best.pt")
        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and stale_epochs >= args.early_stop_patience
        ):
            print(
                f"[eeg-wan] early stop at epoch {epoch}: {args.selection_metric} did not improve "
                f"by {args.early_stop_min_delta:g} for {stale_epochs} epochs",
                flush=True,
            )
            break
    print(f"[eeg-wan] best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
