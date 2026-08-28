"""Train Method C1: raw EEG to the official Tora full T5 condition space."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditionerConfig
from ms_video_eval.eeg_semantic import (
    DirectSemanticAlignmentModel,
    cross_session_consistency_loss,
    full_text_alignment_loss,
    multi_positive_contrastive_loss,
    semantic_slot_loss,
)
from ms_video_eval.semantic_data import (
    EEGSemanticDataset,
    SemanticVocabulary,
    VideoGroupedBatchSampler,
    load_semantic_record_map,
    load_trial_rows,
    load_video_partitions,
    select_partition_trials,
    semantic_collate,
)
from ms_video_eval.tora_conditioning import (
    TORA_T5_HIDDEN_DIM,
    TORA_TEXT_TOKENS,
    load_tora_condition,
    read_tora_condition_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume output-dir/last.pt exactly")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


class EEGToraAlignmentDataset(EEGSemanticDataset):
    def __init__(
        self,
        *args: Any,
        target_index: dict[str, dict[str, Any]],
        target_kind: str,
        expected_shape: tuple[int, int],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_index = target_index
        self.target_kind = target_kind
        self.expected_shape = expected_shape
        self.target_cache: dict[str, torch.Tensor] = {}
        missing = {row["video_id"] for row in self.rows} - set(target_index)
        if missing:
            raise KeyError(f"Tora targets missing video IDs: {sorted(missing)[:5]}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        video_id = item["video_id"]
        if video_id not in self.target_cache:
            row = self.target_index[video_id]
            if self.target_kind == "direct_tora_text":
                value = load_tora_condition(Path(row["condition_path"])).hidden_state
            elif self.target_kind == "tora_pca":
                payload = torch.load(row["latent_path"], map_location="cpu", weights_only=False)
                value = payload["latent"].float()
            else:
                raise ValueError(f"Unknown target kind: {self.target_kind}")
            if tuple(value.shape) != self.expected_shape:
                raise ValueError(
                    f"Target {video_id} shape {tuple(value.shape)} != {self.expected_shape}"
                )
            self.target_cache[video_id] = value
        item["text_state"] = self.target_cache[video_id]
        return item


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = semantic_collate(batch)
    result["text_state"] = torch.stack([item["text_state"] for item in batch])
    return result


def positive_mask(video_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in video_ids] for left in video_ids],
        device=device,
        dtype=torch.bool,
    )


def compute_train_target_mean(
    index: dict[str, dict[str, Any]],
    video_ids: set[str],
    target_kind: str,
) -> torch.Tensor:
    if not video_ids:
        raise ValueError("Cannot center an empty training target set")
    total: torch.Tensor | None = None
    count = 0
    for video_id in sorted(video_ids):
        if target_kind == "direct_tora_text":
            value = load_tora_condition(Path(index[video_id]["condition_path"])).hidden_state
        elif target_kind == "tora_pca":
            payload = torch.load(
                index[video_id]["latent_path"], map_location="cpu", weights_only=False
            )
            value = payload["latent"].float()
        else:
            raise ValueError(f"Unknown target kind: {target_kind}")
        total = value.double() if total is None else total + value.double()
        count += 1
    assert total is not None
    return (total / count).float()


def make_loader(
    dataset: EEGToraAlignmentDataset,
    rows: list[dict[str, str]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=VideoGroupedBatchSampler(
            rows, batch_size, shuffle=shuffle, generator=generator
        ),
        num_workers=workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def loss_for_batch(
    model: DirectSemanticAlignmentModel,
    batch: dict[str, Any],
    device: torch.device,
    weights: dict[str, float],
    contrastive_temperature: float,
    target_mean: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["eeg"].to(device))
    target = batch["text_state"].to(device)
    predicted = output["latent"]
    if target_mean is not None:
        predicted = predicted + target_mean
    alignment, values = full_text_alignment_loss(
        predicted,
        target,
        mse_weight=weights["mse"],
        cosine_weight=weights["cosine"],
    )
    contrastive = multi_positive_contrastive_loss(
        predicted.mean(dim=1),
        target.mean(dim=1),
        positive_mask(batch["video_id"], device),
        temperature=contrastive_temperature,
    )
    session = cross_session_consistency_loss(output["feature"], batch["video_id"])
    auxiliary, aux_values = semantic_slot_loss(
        output["auxiliary_logits"], batch["targets"], batch["target_masks"]
    )
    total = (
        alignment
        + weights["contrastive"] * contrastive
        + weights["auxiliary_classification"] * auxiliary
        + weights["session_consistency"] * session
    )
    return total, {
        **values,
        "contrastive": float(contrastive.detach()),
        "auxiliary": float(auxiliary.detach()),
        "session_consistency": float(session.detach()),
        **{f"aux_{key}": value for key, value in aux_values.items()},
    }


@torch.no_grad()
def evaluate(
    model: DirectSemanticAlignmentModel,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    temperature: float,
    target_mean: torch.Tensor | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    aggregate: dict[str, float] = {}
    examples = 0
    for batch_index, batch in enumerate(loader):
        loss, values = loss_for_batch(
            model, batch, device, weights, temperature, target_mean
        )
        values["loss"] = float(loss)
        size = batch["eeg"].shape[0]
        examples += size
        for key, value in values.items():
            aggregate[key] = aggregate.get(key, 0.0) + value * size
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return {key: value / max(1, examples) for key, value in aggregate.items()}


def main() -> None:
    args = parse_args()
    if args.smoke and args.resume:
        raise ValueError("--smoke and --resume cannot be combined")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method = config["experiment"]["method"]
    if method not in {"direct_tora_text", "tora_pca"}:
        raise ValueError("Expected direct_tora_text or tora_pca")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device(
        args.device or config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data = config["data"]
    partitions = load_video_partitions(resolve_path(data["split_plan"]), data["fold"])
    all_rows = load_trial_rows(resolve_path(data["trials"]))
    sessions = tuple(data.get("sessions", ["session1", "session2", "session3"]))
    train_rows = select_partition_trials(all_rows, partitions["train"], sessions)
    validation_rows = select_partition_trials(all_rows, partitions["validation"], sessions)
    record_map = load_semantic_record_map(resolve_path(data["semantic_labels"]))
    auxiliary_slots = ("subject", "object", "coarse_action")
    vocabulary = SemanticVocabulary.fit(
        [record_map[key] for key in sorted(partitions["train"])], auxiliary_slots
    )
    target_index = read_tora_condition_index(resolve_path(data["tora_target_index"]))
    model_values = config["model"]
    slots = int(model_values["condition_slots"])
    dimension = int(model_values["condition_dim"])
    if slots != TORA_TEXT_TOKENS:
        raise ValueError(f"Official Tora target must use {TORA_TEXT_TOKENS} text slots")
    if method == "direct_tora_text" and dimension != TORA_T5_HIDDEN_DIM:
        raise ValueError(
            f"Official Tora target must be [{TORA_TEXT_TOKENS}, {TORA_T5_HIDDEN_DIM}]"
        )
    encoder_config = EEGConditionerConfig(
        channels=int(model_values.get("channels", 62)),
        sample_points=int(model_values.get("sample_points", 800)),
        sampling_rate=int(model_values.get("sampling_rate", 200)),
        hidden_dim=int(model_values.get("hidden_dim", 256)),
        token_count=int(model_values.get("token_count", 75)),
        encoder_layers=int(model_values.get("encoder_layers", 2)),
        decoder_layers=int(model_values.get("decoder_layers", 2)),
        heads=int(model_values.get("heads", 8)),
        dropout=float(model_values.get("dropout", 0.15)),
        architecture=str(model_values.get("architecture", "baseline")),
        slots=slots,
        latent_dim=dimension,
        min_tokens=slots,
        max_tokens=slots,
    )
    train_dataset = EEGToraAlignmentDataset(
        train_rows, record_map, vocabulary, ROOT, encoder_config.sample_points,
        target_index=target_index,
        target_kind=method,
        expected_shape=(slots, dimension),
    )
    validation_dataset = EEGToraAlignmentDataset(
        validation_rows, record_map, vocabulary, ROOT, encoder_config.sample_points,
        target_index=target_index,
        target_kind=method,
        expected_shape=(slots, dimension),
    )
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        train_dataset, train_rows, int(training["batch_size"]), int(training.get("workers", 0)),
        True, device, generator,
    )
    validation_loader = make_loader(
        validation_dataset, validation_rows, int(training["batch_size"]), int(training.get("workers", 0)),
        False, device, generator,
    )
    model = DirectSemanticAlignmentModel(
        encoder_config,
        {slot: len(values) for slot, values in vocabulary.values.items()},
    ).to(device)
    target_mean = None
    if bool(model_values.get("target_centering", True)):
        target_mean = compute_train_target_mean(
            target_index, partitions["train"], method
        ).to(device)
        torch.nn.init.zeros_(model.latent_head[-1].weight)
        torch.nn.init.zeros_(model.latent_head[-1].bias)
    loss_config = config["loss"]
    weights = {
        key: float(loss_config.get(key, default))
        for key, default in (
            ("mse", 1.0), ("cosine", 0.2), ("contrastive", 0.2),
            ("auxiliary_classification", 0.1), ("session_consistency", 0.1),
        )
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("Loss weights must be non-negative")
    temperature = float(config["contrastive"].get("temperature", 0.07))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = 1 if args.smoke else int(training.get("epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = (
        ROOT / "outputs" / "semantic_smoke" / config["experiment"]["name"]
        if args.smoke else resolve_path(config["experiment"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "semantic_vocabulary.json").write_text(
        json.dumps(vocabulary.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path = output_dir / "history.jsonl"
    if history_path.exists() and not args.smoke and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {history_path}")
    if args.resume and not (output_dir / "last.pt").is_file():
        raise FileNotFoundError(f"Cannot resume without {output_dir / 'last.pt'}")
    best = float("inf")
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(output_dir / "last.pt", map_location=device, weights_only=False)
        if checkpoint.get("method") != method:
            raise ValueError("Resume checkpoint method does not match config")
        required = {
            "optimizer_state", "scheduler_state", "scaler_state", "best_loss", "rng_state"
        }
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint predates exact resume support; missing {sorted(missing)}")
        if checkpoint.get("config") != config:
            raise ValueError("Resume checkpoint config does not exactly match the requested config")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"], generator)
        best = float(checkpoint["best_loss"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[eeg-tora] resuming at epoch {start_epoch} with best={best:.6f}", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_sum = 0.0
        examples = 0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss, _ = loss_for_batch(
                    model, batch, device, weights, temperature, target_mean
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            size = batch["eeg"].shape[0]
            train_sum += float(loss.detach()) * size
            examples += size
            if args.smoke:
                break
        scheduler.step()
        valid = evaluate(
            model, validation_loader, device, weights, temperature, target_mean,
            max_batches=1 if args.smoke else None,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_sum / max(1, examples),
            "validation": valid,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        score = float(valid["loss"])
        is_best = score < best
        best = min(best, score)
        payload = {
            "schema_version": 1,
            "method": method,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "encoder_config": asdict(encoder_config),
            "vocabulary": vocabulary.to_json(),
            "target_mean": None if target_mean is None else target_mean.detach().cpu(),
            "validation": valid,
            "config": config,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_loss": best,
            "rng_state": capture_rng_state(generator),
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if is_best:
            atomic_torch_save(payload, output_dir / "best.pt")
        print(
            f"[eeg-tora] epoch={epoch}/{epochs} train_loss={record['train_loss']:.4f} "
            f"val_loss={valid['loss']:.4f} val_mse={valid['mse']:.6f}",
            flush=True,
        )
    if start_epoch > epochs:
        print(f"[eeg-tora] run already completed {epochs} epochs", flush=True)
    print(f"[eeg-tora] best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
