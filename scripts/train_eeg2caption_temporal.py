"""Train the 01--06 EEG2Caption temporal-window classification pilot."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ms_video_eval.eeg2caption_adapter import (
    AdaptedThreeSessionDataset,
    CATEGORY_NAMES,
    OBJECT_NAMES,
    TemporalSegmentCompactClassifier,
    category_object_matrix,
    load_eeg2caption_fold,
    normalization_stats,
    subset_eeg2caption_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def positive_weights(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0)
    return ((len(labels) - positives) / positives.clamp_min(1)).clamp(max=20.0)


@torch.no_grad()
def evaluate_temporal(
    model: TemporalSegmentCompactClassifier,
    loader: DataLoader,
    device: torch.device,
    allowed_categories: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "segment_object_logits", "segment_category_logits",
            "session_object_logits", "session_category_logits",
            "fused_object_logits", "fused_category_logits",
        )
    }
    labels, categories, video_ids = [], [], []
    for batch in loader:
        output = model(batch["eeg"].to(device, non_blocking=True))
        for key in collected:
            collected[key].append(output[key].cpu())
        labels.append(batch["label"])
        categories.append(batch["pair_index"])
        video_ids.extend(batch["video_id"])
    raw: dict[str, Any] = {key: torch.cat(value) for key, value in collected.items()}
    raw["labels"] = torch.cat(labels)
    raw["categories"] = torch.cat(categories)
    raw["video_ids"] = video_ids

    segment_truth = raw["categories"][:, None, None].expand(
        raw["segment_category_logits"].shape[:3]
    )
    session_truth = raw["categories"][:, None].expand(
        raw["session_category_logits"].shape[:2]
    )
    category_predictions = raw["fused_category_logits"].argmax(dim=-1)
    category_indices = torch.tensor(
        [CATEGORY_NAMES.index(name) for name in allowed_categories], dtype=torch.long
    )
    derived_objects = category_object_matrix()[category_indices[category_predictions]]
    object_probabilities = raw["fused_object_logits"].sigmoid()
    per_object_ap = {
        name: float(average_precision_score(raw["labels"][:, index], object_probabilities[:, index]))
        for index, name in enumerate(OBJECT_NAMES)
    }
    category_accuracy = category_predictions.eq(raw["categories"]).float().mean().item()
    metrics = {
        "video_count": len(video_ids),
        "segment_count_per_session": model.segment_count,
        "segment_accuracy": raw["segment_category_logits"].argmax(dim=-1).eq(
            segment_truth
        ).float().mean().item(),
        "session_accuracy": raw["session_category_logits"].argmax(dim=-1).eq(
            session_truth
        ).float().mean().item(),
        "video_category_accuracy": category_accuracy,
        "category_derived_object_exact": derived_objects.eq(raw["labels"]).all(dim=-1).float().mean().item(),
        "object_macro_ap": float(np.mean(list(per_object_ap.values()))),
        "per_object_ap": per_object_ap,
    }
    return metrics, raw


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["experiment"]["method"] != "temporal_category":
        raise ValueError("Expected temporal_category method")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device(args.device or config["training"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = config["data"]
    model_values = config["model"]
    sample_points = int(model_values.get("sample_points", 800))
    allowed_categories = tuple(str(value) for value in data["allowed_categories"])
    if allowed_categories != CATEGORY_NAMES[:6]:
        raise ValueError("Temporal pilot currently requires categories 01--06 in order")
    fold = subset_eeg2caption_fold(
        load_eeg2caption_fold(
            ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
            resolve(data["split_plan"]), data["fold"],
            tuple(data.get("sessions", ("session1", "session2", "session3"))),
            sample_points,
        ),
        allowed_categories,
    )
    expected = {"train": 312, "validation": 78, "test": 78}
    actual = {name: len(values) for name, values in fold.split_indices.items()}
    if actual != expected:
        raise ValueError(f"Unexpected 01--06 fold sizes: {actual}, expected {expected}")
    eeg_scale = float(model_values.get("eeg_scale", 1e6))
    mean, std = normalization_stats(fold.eeg, fold.split_indices["train"], eeg_scale)
    shared = dict(
        eeg=fold.eeg, labels=fold.object_labels, pair_indices=fold.category_targets,
        cardinalities=fold.cardinalities, ids=fold.video_ids, mean=mean, std=std,
        eeg_scale=eeg_scale,
    )
    augmentation = config.get("augmentation", {})
    train_set = AdaptedThreeSessionDataset(
        **shared, indices=fold.split_indices["train"], training=True,
        noise_std=float(augmentation.get("noise_std", 0.0)),
        time_mask_samples=int(augmentation.get("time_mask_samples", 0)),
    )
    validation_set = AdaptedThreeSessionDataset(
        **shared, indices=fold.split_indices["validation"]
    )
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(
        batch_size=int(training.get("batch_size", 32)),
        num_workers=int(training.get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model_config = {
        "segment_samples": int(model_values["segment_samples"]),
        "total_samples": sample_points,
        "num_channels": 62, "num_objects": len(OBJECT_NAMES),
        "num_pairs": len(allowed_categories),
        "temporal_filters": int(model_values.get("temporal_filters", 16)),
        "spatial_multiplier": int(model_values.get("spatial_multiplier", 2)),
        "feature_dim": int(model_values.get("feature_dim", 128)),
        "dropout": float(model_values.get("dropout", 0.35)),
    }
    model = TemporalSegmentCompactClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = 1 if args.smoke else int(training.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs),
        eta_min=float(training.get("learning_rate", 1e-3)) * 0.05,
    )
    amp = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    object_pos_weight = positive_weights(
        fold.object_labels[fold.split_indices["train"]]
    ).to(device)
    loss_values = config.get("loss", {})
    output_dir = (
        ROOT / "outputs/eeg2caption_temporal_smoke" / config["experiment"]["name"]
        if args.smoke else resolve(config["experiment"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if history_path.exists() and not args.resume and not args.smoke:
        raise FileExistsError(f"Refusing to overwrite {history_path}")
    atomic_json(config, output_dir / "resolved_config.json")
    atomic_json({
        "allowed_categories": list(allowed_categories), "fold_sizes": actual,
        "train": [fold.video_ids[index] for index in fold.split_indices["train"]],
        "validation": [fold.video_ids[index] for index in fold.split_indices["validation"]],
        "test": [fold.video_ids[index] for index in fold.split_indices["test"]],
        "normalization_fit": "filtered_train_only", "split_unit": "video",
        "temporal_fusion": "mean_logits", "session_fusion": "mean_logits",
    }, output_dir / "data_protocol.json")
    writer = None
    if bool(config.get("logging", {}).get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best_score = -math.inf
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume config mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng(checkpoint["rng_state"], generator)
        best_score = float(checkpoint["best_validation_score"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[eeg2caption-temporal] resuming epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch_index, batch in enumerate(train_loader):
            eeg = batch["eeg"].to(device, non_blocking=True)
            labels = batch["label"].to(device)
            categories = batch["pair_index"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                output = model(eeg)
                segment_categories = categories[:, None, None].expand(
                    output["segment_category_logits"].shape[:3]
                )
                segment_labels = labels[:, None, None, :].expand_as(
                    output["segment_object_logits"]
                )
                fused_category = F.cross_entropy(
                    output["fused_category_logits"], categories
                )
                segment_category = F.cross_entropy(
                    output["segment_category_logits"].reshape(-1, len(allowed_categories)),
                    segment_categories.reshape(-1),
                )
                object_auxiliary = (
                    F.binary_cross_entropy_with_logits(
                        output["fused_object_logits"], labels,
                        pos_weight=object_pos_weight,
                    )
                    + F.binary_cross_entropy_with_logits(
                        output["segment_object_logits"], segment_labels,
                        pos_weight=object_pos_weight,
                    )
                )
                features = F.normalize(output["segment_features"], dim=-1)
                consistency = (
                    features - features.mean(dim=(1, 2), keepdim=True)
                ).pow(2).mean()
                loss = (
                    float(loss_values.get("fused_category_weight", 1.0)) * fused_category
                    + float(loss_values.get("segment_category_weight", 0.5)) * segment_category
                    + float(loss_values.get("object_auxiliary_weight", 0.2)) * object_auxiliary
                    + float(loss_values.get("consistency_weight", 0.1)) * consistency
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip", 5.0))
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(eeg)
            examples += len(eeg)
            if args.smoke:
                break
        scheduler.step()
        metrics, _ = evaluate_temporal(
            model, validation_loader, device, allowed_categories
        )
        score = metrics["video_category_accuracy"] + 0.01 * metrics["object_macro_ap"]
        record = {
            "epoch": epoch, "train_loss": total_loss / max(1, examples),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics, "selection_score": score,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if writer is not None:
            writer.add_scalar("loss/train", record["train_loss"], epoch)
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"validation/{name}", value, epoch)
            writer.flush()
        improved = score > best_score
        best_score = max(best_score, score)
        payload = {
            "schema_version": 1, "implementation": "EEG2Caption Compact temporal",
            "method": "temporal_category", "epoch": epoch,
            "model_state": model.state_dict(), "model_config": model_config,
            "normalization_mean": mean, "normalization_std": std,
            "eeg_scale": eeg_scale, "allowed_categories": allowed_categories,
            "validation": metrics, "best_validation_score": best_score,
            "config": config, "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "rng_state": capture_rng(generator),
        }
        atomic_save(payload, output_dir / "last.pt")
        if improved:
            atomic_save(payload, output_dir / "best.pt")
            atomic_save(payload, output_dir / "best_overall.pt")
        print(
            f"[eeg2caption-temporal] epoch={epoch}/{epochs} "
            f"loss={record['train_loss']:.4f} "
            f"segment_acc={metrics['segment_accuracy']:.4f} "
            f"session_acc={metrics['session_accuracy']:.4f} "
            f"video_acc={metrics['video_category_accuracy']:.4f}",
            flush=True,
        )
    if start_epoch > epochs:
        print(f"[eeg2caption-temporal] already completed {epochs} epochs", flush=True)
    if writer is not None:
        writer.close()
    atomic_json(
        {"schema_version": 1, "completed_epochs": epochs, "checkpoint": "last.pt"},
        output_dir / "completed.json",
    )
    print(f"[eeg2caption-temporal] best checkpoint: {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
