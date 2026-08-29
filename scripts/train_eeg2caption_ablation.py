"""Train Method A/B with the released EEG2Caption Compact three-session pipeline."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg2caption_adapter import (
    AdaptedThreeSessionDataset,
    CATEGORY_NAMES,
    CompactStructuredClassifier,
    OBJECT_NAMES,
    build_semantic_targets,
    category_object_matrix,
    load_eeg2caption_fold,
    normalization_stats,
    predicted_object_sets,
)
from ms_video_eval.semantic_data import SemanticVocabulary
from ms_video_eval.semantic_metrics import multilabel_slot_metrics, search_slot_thresholds


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


def atomic_torch_save(value: Any, path: Path) -> None:
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


def positive_weights(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    selected = values if mask is None else values[mask.bool()]
    positives = selected.sum(dim=0)
    negatives = len(selected) - positives
    return (negatives / positives.clamp_min(1)).clamp(max=20.0)


def semantic_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    if not logits:
        return next(iter(targets.values())).new_zeros(()) if targets else torch.tensor(0.0)
    total = next(iter(logits.values())).new_zeros(())
    active = 0
    for slot, prediction in logits.items():
        target = targets[slot].to(prediction.device)
        mask = masks[slot].to(prediction.device).bool()
        if mask.any():
            total = total + F.binary_cross_entropy_with_logits(
                prediction[mask], target[mask], pos_weight=weights[slot]
            )
            active += 1
    return total / max(1, active)


@torch.no_grad()
def evaluate(
    model: CompactStructuredClassifier,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    fixed_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    keys = ("session_object_logits", "session_pair_logits", "fused_object_logits", "fused_pair_logits")
    values: dict[str, list[Any]] = {key: [] for key in keys}
    labels: list[torch.Tensor] = []
    categories: list[torch.Tensor] = []
    cardinalities: list[torch.Tensor] = []
    video_ids: list[str] = []
    semantic_logits: dict[str, list[torch.Tensor]] = {}
    semantic_targets: dict[str, list[torch.Tensor]] = {}
    semantic_masks: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        output = model(batch["eeg"].to(device, non_blocking=True))
        for key in keys:
            values[key].append(output[key].cpu())
        for slot, logits in output["fused_semantic_logits"].items():
            semantic_logits.setdefault(slot, []).append(logits.cpu())
            semantic_targets.setdefault(slot, []).append(batch["semantic_targets"][slot])
            semantic_masks.setdefault(slot, []).append(batch["semantic_masks"][slot])
        labels.append(batch["label"])
        categories.append(batch["pair_index"])
        cardinalities.append(batch["cardinality"])
        video_ids.extend(batch["video_id"])
    raw = {key: torch.cat(items) for key, items in values.items()}
    raw["labels"] = torch.cat(labels)
    raw["categories"] = torch.cat(categories)
    raw["cardinalities"] = torch.cat(cardinalities)
    raw["video_ids"] = video_ids
    raw["semantic_logits"] = {key: torch.cat(items) for key, items in semantic_logits.items()}
    raw["semantic_targets"] = {key: torch.cat(items) for key, items in semantic_targets.items()}
    raw["semantic_masks"] = {key: torch.cat(items) for key, items in semantic_masks.items()}

    def classification_block(object_logits: torch.Tensor, category_logits: torch.Tensor) -> dict[str, Any]:
        probabilities = object_logits.sigmoid()
        per_object_ap = {
            name: float(average_precision_score(raw["labels"][:, index], probabilities[:, index]))
            for index, name in enumerate(OBJECT_NAMES)
        }
        predictions, category_predictions = predicted_object_sets(object_logits, category_logits)
        exact = predictions.eq(raw["labels"]).all(dim=1).float().mean().item()
        recall = (
            (predictions * raw["labels"]).sum(dim=1) / raw["cardinalities"].float()
        ).mean().item()
        category_accuracy = category_predictions.eq(raw["categories"]).float().mean().item()
        constrained = category_object_matrix()[category_predictions]
        return {
            "macro_ap": float(np.mean(list(per_object_ap.values()))),
            "per_object_ap": per_object_ap,
            "predicted_cardinality_exact_set": exact,
            "predicted_cardinality_recall": recall,
            "category_accuracy": category_accuracy,
            "category_constrained_exact_set": constrained.eq(raw["labels"]).all(dim=1).float().mean().item(),
        }
    selected_thresholds: dict[str, float] = {}
    structured_metrics: dict[str, Any] = {}
    if raw["semantic_logits"]:
        selected_thresholds = (
            {key: float(value) for key, value in fixed_thresholds.items()}
            if fixed_thresholds is not None
            else search_slot_thresholds(
                raw["semantic_logits"], raw["semantic_targets"], raw["semantic_masks"], thresholds
            )
        )
        structured_metrics = multilabel_slot_metrics(
            raw["semantic_logits"], raw["semantic_targets"], raw["semantic_masks"],
            selected_thresholds,
        )
    metrics = {
        "session1": classification_block(
            raw["session_object_logits"][:, 0], raw["session_pair_logits"][:, 0]
        ),
        "session2": classification_block(
            raw["session_object_logits"][:, 1], raw["session_pair_logits"][:, 1]
        ),
        "session3": classification_block(
            raw["session_object_logits"][:, 2], raw["session_pair_logits"][:, 2]
        ),
        "fused": classification_block(raw["fused_object_logits"], raw["fused_pair_logits"]),
        "structured": structured_metrics,
        "structured_thresholds": selected_thresholds,
    }
    return metrics, raw


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method = config["experiment"]["method"]
    if method not in {"coarse_template", "structured_semantic"}:
        raise ValueError("Expected Method A or B")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device(args.device or config["training"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = config["data"]
    sessions = tuple(data.get("sessions", ("session1", "session2", "session3")))
    sample_points = int(config["model"].get("sample_points", 800))
    fold = load_eeg2caption_fold(
        ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
        resolve(data["split_plan"]), data["fold"], sessions, sample_points,
    )
    slots = tuple(config.get("semantic", {}).get("slots", ())) if method == "structured_semantic" else ()
    train_records = [fold.records[fold.video_ids[index]] for index in fold.split_indices["train"]]
    vocabulary = SemanticVocabulary.fit(
        train_records, slots, config.get("semantic", {}).get("min_frequency", {})
    ) if slots else SemanticVocabulary(values={}, min_frequency={})
    targets, masks = build_semantic_targets(fold, vocabulary)
    eeg_scale = float(config["model"].get("eeg_scale", 1e6))
    mean, std = normalization_stats(fold.eeg, fold.split_indices["train"], eeg_scale)
    shared = dict(
        eeg=fold.eeg, labels=fold.object_labels, pair_indices=fold.category_targets,
        cardinalities=fold.cardinalities, ids=fold.video_ids, mean=mean, std=std,
        eeg_scale=eeg_scale, semantic_targets=targets, semantic_masks=masks,
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
    loader_kwargs = dict(
        batch_size=int(training.get("batch_size", 32)),
        num_workers=int(training.get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_kwargs)
    model_config = {
        "num_channels": 62, "num_objects": len(OBJECT_NAMES),
        "num_pairs": len(CATEGORY_NAMES),
        "temporal_filters": int(config["model"].get("temporal_filters", 16)),
        "spatial_multiplier": int(config["model"].get("spatial_multiplier", 2)),
        "feature_dim": int(config["model"].get("feature_dim", 128)),
        "dropout": float(config["model"].get("dropout", 0.35)),
        "semantic_classes": {key: len(value) for key, value in vocabulary.values.items()},
    }
    model = CompactStructuredClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = 1 if args.smoke else int(training.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(training.get("learning_rate", 1e-3)) * 0.05
    )
    amp = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    object_pos_weight = positive_weights(fold.object_labels[fold.split_indices["train"]]).to(device)
    semantic_pos_weights = {
        slot: positive_weights(values[fold.split_indices["train"]], masks[slot][fold.split_indices["train"]]).to(device)
        for slot, values in targets.items()
    }
    loss_config = config.get("loss", {})
    output_dir = (
        ROOT / "outputs/eeg2caption_smoke" / config["experiment"]["name"]
        if args.smoke
        else resolve(config["experiment"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if history_path.exists() and not args.resume and not args.smoke:
        raise FileExistsError(f"Refusing to overwrite {history_path}")
    atomic_json(config, output_dir / "resolved_config.json")
    atomic_json(
        {"train": [fold.video_ids[i] for i in fold.split_indices["train"]],
         "validation": [fold.video_ids[i] for i in fold.split_indices["validation"]],
         "test": [fold.video_ids[i] for i in fold.split_indices["test"]],
         "normalization_fit": "train_only", "session_fusion": "mean_logits"},
        output_dir / "data_protocol.json",
    )
    writer = None
    if bool(config.get("logging", {}).get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best = -math.inf
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
        best = float(checkpoint["best_validation_macro_ap"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[eeg2caption] resuming at epoch {start_epoch} best_mAP={best:.4f}", flush=True)
    threshold_candidates = [float(value) for value in config.get("semantic", {}).get("threshold_search", (0.3, 0.4, 0.5, 0.6, 0.7, 0.8))]
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
                session_labels = labels[:, None, :].expand_as(output["session_object_logits"])
                session_categories = categories[:, None].expand(-1, 3)
                object_loss = F.binary_cross_entropy_with_logits(
                    output["session_object_logits"], session_labels, pos_weight=object_pos_weight
                ) + float(loss_config.get("fused_weight", 1.0)) * F.binary_cross_entropy_with_logits(
                    output["fused_object_logits"], labels, pos_weight=object_pos_weight
                )
                category_loss = F.cross_entropy(
                    output["session_pair_logits"].reshape(-1, len(CATEGORY_NAMES)),
                    session_categories.reshape(-1),
                ) + float(loss_config.get("fused_weight", 1.0)) * F.cross_entropy(
                    output["fused_pair_logits"], categories
                )
                structured = eeg.new_zeros(())
                if output["fused_semantic_logits"]:
                    batch_targets = {key: value.to(device) for key, value in batch["semantic_targets"].items()}
                    batch_masks = {key: value.to(device) for key, value in batch["semantic_masks"].items()}
                    session_structured = semantic_loss(
                        {key: value.reshape(-1, value.shape[-1]) for key, value in output["session_semantic_logits"].items()},
                        {key: value[:, None, :].expand(-1, 3, -1).reshape(-1, value.shape[-1]) for key, value in batch_targets.items()},
                        {key: value[:, None].expand(-1, 3).reshape(-1) for key, value in batch_masks.items()},
                        semantic_pos_weights,
                    )
                    fused_structured = semantic_loss(
                        output["fused_semantic_logits"], batch_targets, batch_masks,
                        semantic_pos_weights,
                    )
                    structured = session_structured + float(loss_config.get("fused_weight", 1.0)) * fused_structured
                normalized = F.normalize(output["features"], dim=-1)
                consistency = (normalized - normalized.mean(dim=1, keepdim=True)).pow(2).mean()
                loss = (
                    object_loss
                    + float(loss_config.get("category_weight", 0.5)) * category_loss
                    + float(loss_config.get("structured_weight", 1.0)) * structured
                    + float(loss_config.get("consistency_weight", 0.05)) * consistency
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            examples += len(labels)
            if args.smoke:
                break
        scheduler.step()
        metrics, _ = evaluate(model, validation_loader, device, threshold_candidates)
        score = float(metrics["fused"]["macro_ap"])
        record = {
            "epoch": epoch, "train_loss": total_loss / max(1, examples),
            "learning_rate": optimizer.param_groups[0]["lr"], "validation": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if writer is not None:
            writer.add_scalar("loss/train_total", record["train_loss"], epoch)
            writer.add_scalar("metrics/validation_fused_macro_ap", score, epoch)
            writer.add_scalar(
                "metrics/validation_object_exact",
                metrics["fused"]["predicted_cardinality_exact_set"], epoch,
            )
            writer.add_scalar(
                "metrics/validation_category_accuracy",
                metrics["fused"]["category_accuracy"], epoch,
            )
            writer.add_scalar("optimization/learning_rate", record["learning_rate"], epoch)
            writer.flush()
        improved = score > best
        best = max(best, score)
        payload = {
            "schema_version": 2, "implementation": "EEG2Caption Compact adapted",
            "method": method, "epoch": epoch, "model_state": model.state_dict(),
            "model_config": model_config, "normalization_mean": mean,
            "normalization_std": std, "eeg_scale": eeg_scale,
            "vocabulary": vocabulary.to_json(), "validation": metrics,
            "best_validation_macro_ap": best, "config": config,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "rng_state": capture_rng(generator),
            "object_names": OBJECT_NAMES, "category_names": CATEGORY_NAMES,
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
            atomic_torch_save(payload, output_dir / "best_semantic.pt")
            atomic_torch_save(payload, output_dir / "best_overall.pt")
        print(
            f"[eeg2caption] epoch={epoch}/{epochs} loss={record['train_loss']:.4f} "
            f"val_mAP={score:.4f} val_object_exact={metrics['fused']['predicted_cardinality_exact_set']:.4f} "
            f"val_category_acc={metrics['fused']['category_accuracy']:.4f}", flush=True,
        )
    if start_epoch > epochs:
        print(f"[eeg2caption] run already completed {epochs} epochs", flush=True)
    if writer is not None:
        writer.close()
    atomic_json(
        {"schema_version": 1, "completed_epochs": epochs, "checkpoint": "last.pt"},
        output_dir / "completed.json",
    )
    print(f"[eeg2caption] best checkpoint: {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
