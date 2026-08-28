"""Train Method A/B semantic heads using a strict video-held-out protocol."""

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
    SemanticSlotModel,
    cross_session_consistency_loss,
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
from ms_video_eval.semantic_metrics import multilabel_slot_metrics, search_slot_thresholds
from ms_video_eval.semantic_tricks import (
    EEGAugmentationConfig,
    augment_eeg,
    classification_prototype_loss,
    curriculum_multiplier,
    hierarchical_positive_weights,
    weighted_positive_contrastive_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one train/validation batch only")
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


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def make_loader(
    dataset: EEGSemanticDataset,
    rows: list[dict[str, str]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    group_sessions: bool,
    device: torch.device,
    generator: torch.Generator,
) -> DataLoader:
    kwargs = {
        "num_workers": workers,
        "collate_fn": semantic_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if group_sessions:
        return DataLoader(
            dataset,
            batch_sampler=VideoGroupedBatchSampler(
                rows, batch_size, shuffle=shuffle, generator=generator
            ),
            **kwargs,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, **kwargs)


@torch.no_grad()
def evaluate(
    model: SemanticSlotModel,
    loader: DataLoader,
    device: torch.device,
    slot_weights: dict[str, float],
    thresholds: dict[str, float],
    threshold_candidates: list[float] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    collected_logits: dict[str, list[torch.Tensor]] = {}
    collected_targets: dict[str, list[torch.Tensor]] = {}
    collected_masks: dict[str, list[torch.Tensor]] = {}
    loss_sum = 0.0
    examples = 0
    for batch_index, batch in enumerate(loader):
        output = model(batch["eeg"].to(device))
        loss, _ = semantic_slot_loss(
            output["logits"], batch["targets"], batch["target_masks"], slot_weights
        )
        size = batch["eeg"].shape[0]
        loss_sum += float(loss) * size
        examples += size
        for slot, logits in output["logits"].items():
            collected_logits.setdefault(slot, []).append(logits.cpu())
            collected_targets.setdefault(slot, []).append(batch["targets"][slot])
            collected_masks.setdefault(slot, []).append(batch["target_masks"][slot])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    merged_logits = {key: torch.cat(value) for key, value in collected_logits.items()}
    merged_targets = {key: torch.cat(value) for key, value in collected_targets.items()}
    merged_masks = {key: torch.cat(value) for key, value in collected_masks.items()}
    selected_thresholds = (
        search_slot_thresholds(
            merged_logits, merged_targets, merged_masks, threshold_candidates
        )
        if threshold_candidates
        else thresholds
    )
    metrics = multilabel_slot_metrics(
        merged_logits, merged_targets, merged_masks, selected_thresholds
    )
    return {
        "loss": loss_sum / max(1, examples),
        "examples": examples,
        "thresholds": selected_thresholds,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if args.smoke and args.resume:
        raise ValueError("--smoke and --resume cannot be combined")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method = config["experiment"]["method"]
    if method not in {"coarse_template", "structured_semantic"}:
        raise ValueError("This trainer supports coarse_template and structured_semantic")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device(
        args.device or config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data = config["data"]
    trials = load_trial_rows(resolve_path(data["trials"]))
    partitions = load_video_partitions(resolve_path(data["split_plan"]), data["fold"])
    if not partitions["train"].isdisjoint(partitions["validation"] | partitions["test"]):
        raise RuntimeError("Video split leakage")
    sessions = tuple(data.get("sessions", ["session1", "session2", "session3"]))
    train_rows = select_partition_trials(trials, partitions["train"], sessions)
    validation_rows = select_partition_trials(trials, partitions["validation"], sessions)
    record_map = load_semantic_record_map(resolve_path(data["semantic_labels"]))
    train_records = [record_map[video_id] for video_id in sorted(partitions["train"])]
    semantic = config["semantic"]
    slots = tuple(semantic["slots"])
    vocabulary = SemanticVocabulary.fit(
        train_records, slots, semantic.get("min_frequency", {})
    )
    sample_points = int(config["model"].get("sample_points", 800))
    train_dataset = EEGSemanticDataset(train_rows, record_map, vocabulary, ROOT, sample_points)
    validation_dataset = EEGSemanticDataset(
        validation_rows, record_map, vocabulary, ROOT, sample_points
    )

    training = config["training"]
    batch_size = int(training.get("batch_size", 24))
    workers = int(training.get("workers", 0))
    generator = torch.Generator().manual_seed(seed)
    group_sessions = bool(training.get("group_sessions", True))
    train_loader = make_loader(
        train_dataset, train_rows, batch_size, workers, True,
        group_sessions, device, generator,
    )
    validation_loader = make_loader(
        validation_dataset, validation_rows, batch_size, workers, False,
        group_sessions, device, generator,
    )
    model_values = config["model"]
    encoder_config = EEGConditionerConfig(
        channels=int(model_values.get("channels", 62)),
        sample_points=sample_points,
        hidden_dim=int(model_values.get("hidden_dim", 256)),
        token_count=int(model_values.get("token_count", 75)),
        encoder_layers=int(model_values.get("encoder_layers", 2)),
        decoder_layers=int(model_values.get("decoder_layers", 2)),
        heads=int(model_values.get("heads", 8)),
        dropout=float(model_values.get("dropout", 0.15)),
        architecture=str(model_values.get("architecture", "baseline")),
        sampling_rate=int(model_values.get("sampling_rate", 200)),
    )
    model = SemanticSlotModel(
        encoder_config,
        {slot: len(values) for slot, values in vocabulary.values.items()},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = 1 if args.smoke else int(training.get("epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_config = config.get("loss", {})
    slot_weights = {key: float(value) for key, value in loss_config.get("slot_weights", {}).items()}
    session_weight = float(loss_config.get("session_consistency", 0.0))
    contrastive_weight = float(loss_config.get("contrastive", 0.0))
    prototype_weight = float(loss_config.get("prototype", 0.0))
    if min(session_weight, contrastive_weight, prototype_weight) < 0:
        raise ValueError("Loss weights must be non-negative")
    contrastive_config = config.get("contrastive", {})
    contrastive_temperature = float(contrastive_config.get("temperature", 0.07))
    positive_config = contrastive_config.get("positive_weights", {})
    augmentation = EEGAugmentationConfig.from_mapping(config.get("augmentation"))
    curriculum = config.get("curriculum", {})
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    thresholds = {key: float(value) for key, value in semantic.get("thresholds", {}).items()}
    threshold_candidates = [float(value) for value in semantic.get("threshold_search", [])]
    output_dir = (
        ROOT / "outputs" / "semantic_smoke" / config["experiment"]["name"]
        if args.smoke
        else resolve_path(config["experiment"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if history_path.exists() and not args.smoke and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {history_path}")
    if args.resume and not (output_dir / "last.pt").is_file():
        raise FileNotFoundError(f"Cannot resume without {output_dir / 'last.pt'}")
    atomic_json(config, output_dir / "resolved_config.json")
    atomic_json(vocabulary.to_json(), output_dir / "semantic_vocabulary.json")
    atomic_json(
        {
            "train_videos": len(partitions["train"]),
            "validation_videos": len(partitions["validation"]),
            "test_videos": len(partitions["test"]),
            "train_trials": len(train_rows),
            "validation_trials": len(validation_rows),
            "time_policy": f"first_{sample_points}_samples",
            "sessions_are_independent_samples": True,
            "group_sessions_in_batch": group_sessions,
        },
        output_dir / "data_protocol.json",
    )
    writer = None
    if bool(config.get("logging", {}).get("tensorboard", True)):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("TensorBoard logging is enabled; install requirements-eeg-semantic.txt") from exc
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    best = float("-inf")
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(output_dir / "last.pt", map_location=device, weights_only=False)
        if checkpoint.get("method") != method:
            raise ValueError("Resume checkpoint method does not match config")
        required = {
            "optimizer_state", "scheduler_state", "scaler_state", "best_score", "rng_state"
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
        best = float(checkpoint["best_score"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[eeg-semantic] resuming at epoch {start_epoch} with best={best:.6f}", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        examples = 0
        component_sums: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            eeg = augment_eeg(batch["eeg"].to(device), augmentation)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(eeg)
                slot_loss, components = semantic_slot_loss(
                    output["logits"], batch["targets"], batch["target_masks"], slot_weights
                )
                session_loss = cross_session_consistency_loss(
                    output["feature"], batch["video_id"]
                )
                positive_weights = hierarchical_positive_weights(
                    batch["video_id"],
                    {key: value.to(device) for key, value in batch["targets"].items()},
                    {key: value.to(device) for key, value in batch["target_masks"].items()},
                    positive_config,
                )
                contrastive_loss = weighted_positive_contrastive_loss(
                    output["feature"], output["feature"], positive_weights,
                    temperature=contrastive_temperature,
                )
                prototype_loss = classification_prototype_loss(
                    output["feature"], model.heads, batch["targets"], batch["target_masks"],
                    loss_config.get("prototype_slot_weights"),
                )
                classification_scale = curriculum_multiplier(
                    epoch, epochs, curriculum, "classification"
                )
                alignment_scale = curriculum_multiplier(epoch, epochs, curriculum, "alignment")
                loss = (
                    classification_scale * slot_loss
                    + session_weight * session_loss
                    + alignment_scale * contrastive_weight * contrastive_loss
                    + prototype_weight * prototype_loss
                )
            scaler.scale(loss / accumulation_steps).backward()
            should_step = (
                args.smoke
                or (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            size = batch["eeg"].shape[0]
            train_loss += float(loss.detach()) * size
            examples += size
            batch_components = {
                **{f"slot_{key}": value for key, value in components.items()},
                "session_consistency": float(session_loss.detach()),
                "contrastive": float(contrastive_loss.detach()),
                "prototype": float(prototype_loss.detach()),
            }
            for key, value in batch_components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * size
            if args.smoke:
                break
        scheduler.step()
        valid = evaluate(
            model, validation_loader, device, slot_weights, thresholds,
            threshold_candidates=threshold_candidates,
            max_batches=1 if args.smoke else None,
        )
        score = float(valid["metrics"]["aggregate"]["macro_f1"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, examples),
            "train_components": {
                key: value / max(1, examples) for key, value in component_sums.items()
            },
            "validation": valid,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if writer is not None:
            writer.add_scalar("loss/train_total", record["train_loss"], epoch)
            writer.add_scalar("loss/validation", valid["loss"], epoch)
            writer.add_scalar("metrics/validation_macro_f1", score, epoch)
            writer.add_scalar("metrics/validation_micro_f1", valid["metrics"]["aggregate"]["micro_f1"], epoch)
            writer.add_scalar("optimization/learning_rate", record["learning_rate"], epoch)
            for key, value in record["train_components"].items():
                writer.add_scalar(f"loss_components/{key}", value, epoch)
            writer.flush()
        is_best = score > best
        best = max(best, score)
        payload = {
            "schema_version": 1,
            "method": method,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "encoder_config": asdict(encoder_config),
            "slot_classes": {slot: len(values) for slot, values in vocabulary.values.items()},
            "vocabulary": vocabulary.to_json(),
            "validation": valid,
            "config": config,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_score": best,
            "rng_state": capture_rng_state(generator),
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if is_best:
            atomic_torch_save(payload, output_dir / "best.pt")
            atomic_torch_save(payload, output_dir / "best_semantic.pt")
            atomic_torch_save(payload, output_dir / "best_overall.pt")
        print(
            f"[eeg-semantic] epoch={epoch}/{epochs} train_loss={record['train_loss']:.4f} "
            f"val_macro_f1={score:.4f}",
            flush=True,
        )
    if start_epoch > epochs:
        print(f"[eeg-semantic] run already completed {epochs} epochs", flush=True)
    if writer is not None:
        writer.close()
    print(f"[eeg-semantic] best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
