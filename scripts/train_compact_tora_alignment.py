"""Train C1/C2/C3 using the shared EEG2Caption Compact three-session backbone."""

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
    CompactToraAlignmentModel,
    OBJECT_NAMES,
    load_eeg2caption_fold,
    normalization_stats,
)
from ms_video_eval.semantic_tricks import curriculum_multiplier
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


def load_target(row: dict[str, Any], method: str) -> torch.Tensor:
    if method == "direct_tora_text":
        return load_tora_condition(Path(row["condition_path"])).hidden_state.float()
    return torch.load(
        row["latent_path"], map_location="cpu", weights_only=False
    )["latent"].float()


class CompactAlignmentDataset(AdaptedThreeSessionDataset):
    def __init__(
        self, *args: Any, target_index: dict[str, dict[str, Any]],
        target_kind: str, expected_shape: tuple[int, int], **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_index = target_index
        self.target_kind = target_kind
        self.expected_shape = expected_shape
        self.target_cache: dict[str, torch.Tensor] = {}
        missing = {self.ids[index] for index in self.indices} - set(target_index)
        if missing:
            raise KeyError(f"Missing Tora targets: {sorted(missing)[:5]}")

    def __getitem__(self, item: int) -> dict[str, Any]:
        result = super().__getitem__(item)
        video_id = result["video_id"]
        if video_id not in self.target_cache:
            target = load_target(self.target_index[video_id], self.target_kind)
            if tuple(target.shape) != self.expected_shape:
                raise ValueError(
                    f"Target {video_id} shape {tuple(target.shape)} != {self.expected_shape}"
                )
            self.target_cache[video_id] = target
        result["text_state"] = self.target_cache[video_id]
        return result


def target_statistics(
    target_index: dict[str, dict[str, Any]], method: str,
    video_ids: list[str], category_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack([load_target(target_index[video_id], method) for video_id in video_ids])
    mean = values.mean(dim=0)
    pooled = values.mean(dim=1)
    prototypes = []
    global_prototype = pooled.mean(dim=0)
    for category in range(len(CATEGORY_NAMES)):
        mask = category_targets == category
        # A non-stratified future fold must not turn the loss into NaN merely
        # because one category is absent from its training partition.
        prototypes.append(pooled[mask].mean(dim=0) if mask.any() else global_prototype)
    return mean, torch.stack(prototypes)


def alignment_losses(
    output: dict[str, Any], target: torch.Tensor, categories: torch.Tensor,
    labels: torch.Tensor, object_pos_weight: torch.Tensor,
    target_mean: torch.Tensor | None, prototypes: torch.Tensor,
    weights: dict[str, float], contrastive: dict[str, Any], scales: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    predicted = output["latent"]
    if target_mean is not None:
        predicted = predicted + target_mean
    mse = F.mse_loss(predicted, target)
    cosine_loss = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
    predicted_pooled = predicted.mean(dim=1)
    target_pooled = target.mean(dim=1)
    temperature = float(contrastive.get("temperature", 0.07))
    similarity = F.normalize(predicted_pooled, dim=-1) @ F.normalize(target_pooled, dim=-1).t()
    mode = str(contrastive.get("mode", "hard_multi_positive"))
    if mode == "soft_semantic":
        target_similarity = F.normalize(target_pooled, dim=-1) @ F.normalize(target_pooled, dim=-1).t()
        target_distribution = F.softmax(
            target_similarity / float(contrastive.get("semantic_temperature", 0.07)), dim=-1
        )
        contrastive_loss = -(
            target_distribution * F.log_softmax(similarity / temperature, dim=-1)
        ).sum(dim=-1).mean()
    else:
        truth = torch.arange(len(target), device=target.device)
        contrastive_loss = F.cross_entropy(similarity / temperature, truth)
    session_features = F.normalize(output["features"], dim=-1)
    session_loss = (
        session_features - session_features.mean(dim=1, keepdim=True)
    ).pow(2).mean()
    session_labels = labels[:, None, :].expand_as(output["session_object_logits"])
    session_categories = categories[:, None].expand(-1, output["session_pair_logits"].shape[1])
    auxiliary = (
        F.binary_cross_entropy_with_logits(
            output["session_object_logits"], session_labels, pos_weight=object_pos_weight
        )
        + F.binary_cross_entropy_with_logits(
            output["fused_object_logits"], labels, pos_weight=object_pos_weight
        )
        + 0.5 * F.cross_entropy(
            output["session_pair_logits"].reshape(-1, len(CATEGORY_NAMES)),
            session_categories.reshape(-1),
        )
        + 0.5 * F.cross_entropy(output["fused_pair_logits"], categories)
    )
    prototype_loss = 1.0 - F.cosine_similarity(
        predicted_pooled, prototypes[categories], dim=-1
    ).mean()
    total = (
        scales["alignment"] * (weights["mse"] * mse + weights["cosine"] * cosine_loss)
        + scales["alignment"] * weights["contrastive"] * contrastive_loss
        + scales["classification"] * weights["auxiliary_classification"] * auxiliary
        + weights["session_consistency"] * session_loss
        + weights["prototype"] * prototype_loss
    )
    return total, {
        "mse": float(mse.detach()), "cosine_loss": float(cosine_loss.detach()),
        "contrastive": float(contrastive_loss.detach()),
        "auxiliary": float(auxiliary.detach()),
        "session_consistency": float(session_loss.detach()),
        "prototype": float(prototype_loss.detach()),
    }, predicted


@torch.no_grad()
def evaluate(
    model: CompactToraAlignmentModel, loader: DataLoader, device: torch.device,
    target_mean: torch.Tensor | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    predictions, targets, labels, categories = [], [], [], []
    video_ids: list[str] = []
    object_logits, category_logits = [], []
    for batch in loader:
        output = model(batch["eeg"].to(device, non_blocking=True))
        predicted = output["latent"]
        if target_mean is not None:
            predicted = predicted + target_mean
        predictions.append(predicted.cpu())
        targets.append(batch["text_state"])
        labels.append(batch["label"])
        categories.append(batch["pair_index"])
        object_logits.append(output["fused_object_logits"].cpu())
        category_logits.append(output["fused_pair_logits"].cpu())
        video_ids.extend(batch["video_id"])
    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    labels_tensor = torch.cat(labels)
    categories_tensor = torch.cat(categories)
    predicted_pooled = F.normalize(predicted.mean(dim=1), dim=-1)
    target_pooled = F.normalize(target.mean(dim=1), dim=-1)
    ranking = (predicted_pooled @ target_pooled.t()).argsort(dim=1, descending=True)
    truth = torch.arange(len(predicted)).unsqueeze(1)
    object_values = torch.cat(object_logits)
    probabilities = object_values.sigmoid()
    per_object_ap = [
        average_precision_score(labels_tensor[:, index], probabilities[:, index])
        for index in range(len(OBJECT_NAMES))
    ]
    metrics = {
        "mse": float(F.mse_loss(predicted, target)),
        "token_cosine": float(F.cosine_similarity(predicted, target, dim=-1).mean()),
        "retrieval_top1": float((ranking[:, :1] == truth).any(dim=1).float().mean()),
        "retrieval_top5": float(
            (ranking[:, : min(5, len(predicted))] == truth).any(dim=1).float().mean()
        ),
        "object_macro_ap": float(np.mean(per_object_ap)),
        "category_accuracy": float(
            torch.cat(category_logits).argmax(dim=1).eq(categories_tensor).float().mean()
        ),
    }
    return metrics, {
        "predicted": predicted, "target": target, "video_ids": video_ids,
        "labels": labels_tensor, "categories": categories_tensor,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method = config["experiment"]["method"]
    if method not in {"direct_tora_text", "tora_pca", "tora_autoencoder"}:
        raise ValueError("Expected C1/C2/C3 method")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device(args.device or config["training"].get("device", "cuda"))
    data = config["data"]
    model_values = config["model"]
    slots = int(model_values["condition_slots"])
    dimension = int(model_values["condition_dim"])
    if slots != TORA_TEXT_TOKENS:
        raise ValueError(f"Tora requires {TORA_TEXT_TOKENS} slots")
    if method == "direct_tora_text" and dimension != TORA_T5_HIDDEN_DIM:
        raise ValueError(f"Direct Tora requires hidden dim {TORA_T5_HIDDEN_DIM}")
    fold = load_eeg2caption_fold(
        ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
        resolve(data["split_plan"]), data["fold"],
        tuple(data.get("sessions", ("session1", "session2", "session3"))),
        int(model_values.get("sample_points", 800)),
    )
    target_index = read_tora_condition_index(resolve(data["tora_target_index"]))
    train_indices = fold.split_indices["train"]
    train_ids = [fold.video_ids[index] for index in train_indices]
    target_mean, prototypes = target_statistics(
        target_index, method, train_ids, fold.category_targets[train_indices]
    )
    if not bool(model_values.get("target_centering", True)):
        target_mean = None
    eeg_scale = float(model_values.get("eeg_scale", 1e6))
    mean, std = normalization_stats(fold.eeg, train_indices, eeg_scale)
    shared = dict(
        eeg=fold.eeg, labels=fold.object_labels, pair_indices=fold.category_targets,
        cardinalities=fold.cardinalities, ids=fold.video_ids, mean=mean, std=std,
        eeg_scale=eeg_scale, target_index=target_index, target_kind=method,
        expected_shape=(slots, dimension),
    )
    augmentation = config.get("augmentation", {})
    train_set = CompactAlignmentDataset(
        **shared, indices=train_indices, training=True,
        noise_std=float(augmentation.get("noise_std", 0.0)),
        time_mask_samples=int(augmentation.get("time_mask_samples", 0)),
    )
    validation_set = CompactAlignmentDataset(
        **shared, indices=fold.split_indices["validation"]
    )
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(
        batch_size=int(training.get("batch_size", 16)),
        num_workers=int(training.get("workers", 0)), pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model_config = {
        "num_channels": 62, "num_objects": len(OBJECT_NAMES), "num_pairs": len(CATEGORY_NAMES),
        "temporal_filters": int(model_values.get("temporal_filters", 16)),
        "spatial_multiplier": int(model_values.get("spatial_multiplier", 2)),
        "feature_dim": int(model_values.get("feature_dim", 128)),
        "dropout": float(model_values.get("dropout", 0.35)),
        "condition_slots": slots, "condition_dim": dimension,
        "decoder_layers": int(model_values.get("decoder_layers", 2)),
        "decoder_heads": int(model_values.get("decoder_heads", 8)),
    }
    model = CompactToraAlignmentModel(**model_config).to(device)
    if target_mean is not None:
        torch.nn.init.zeros_(model.condition_head[-1].weight)
        torch.nn.init.zeros_(model.condition_head[-1].bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = 1 if args.smoke else int(training.get("epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_labels = fold.object_labels[train_indices]
    object_pos_weight = (
        (len(train_labels) - train_labels.sum(dim=0)) / train_labels.sum(dim=0).clamp_min(1)
    ).to(device)
    weights = {
        name: float(config.get("loss", {}).get(name, default))
        for name, default in (
            ("mse", 1.0), ("cosine", 0.2), ("contrastive", 0.2),
            ("auxiliary_classification", 0.1), ("session_consistency", 0.1),
            ("prototype", 0.0),
        )
    }
    output_dir = (
        ROOT / "outputs/compact_tora_smoke" / config["experiment"]["name"]
        if args.smoke else resolve(config["experiment"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if history_path.exists() and not args.resume and not args.smoke:
        raise FileExistsError(f"Refusing to overwrite {history_path}")
    atomic_json(config, output_dir / "resolved_config.json")
    atomic_json({
        "train": train_ids,
        "validation": [fold.video_ids[index] for index in fold.split_indices["validation"]],
        "test": [fold.video_ids[index] for index in fold.split_indices["test"]],
        "normalization_fit": "train_only", "target_statistics_fit": "train_only",
        "session_fusion": "Compact features + query cross-attention",
    }, output_dir / "data_protocol.json")
    writer = None
    if bool(config.get("logging", {}).get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best_score = -math.inf
    best_alignment = math.inf
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
        best_score = float(checkpoint["best_overall_score"])
        best_alignment = float(checkpoint["best_alignment_mse"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[compact-tora] resuming epoch {start_epoch}", flush=True)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        examples = 0
        component_sums: dict[str, float] = {}
        scales = {
            "classification": curriculum_multiplier(epoch, epochs, config.get("curriculum", {}), "classification"),
            "alignment": curriculum_multiplier(epoch, epochs, config.get("curriculum", {}), "alignment"),
        }
        for batch_index, batch in enumerate(train_loader):
            eeg = batch["eeg"].to(device, non_blocking=True)
            target = batch["text_state"].to(device, non_blocking=True)
            categories = batch["pair_index"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                output = model(eeg)
                loss, components, _ = alignment_losses(
                    output, target, categories, labels, object_pos_weight,
                    None if target_mean is None else target_mean.to(device),
                    prototypes.to(device), weights, config.get("contrastive", {}), scales,
                )
            scaler.scale(loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(eeg)
            examples += len(eeg)
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + value * len(eeg)
            if args.smoke:
                break
        scheduler.step()
        validation, _ = evaluate(
            model, validation_loader, device,
            None if target_mean is None else target_mean.to(device),
        )
        score = (
            validation["retrieval_top5"] + 0.25 * validation["retrieval_top1"]
            + 0.01 * validation["token_cosine"]
        )
        record = {
            "epoch": epoch, "train_loss": total_loss / max(1, examples),
            "train_components": {
                name: value / max(1, examples) for name, value in component_sums.items()
            },
            "validation": validation, "selection_score": score,
            "curriculum_scales": scales, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if writer is not None:
            writer.add_scalar("loss/train_total", record["train_loss"], epoch)
            for name, value in record["train_components"].items():
                writer.add_scalar(f"loss_components/{name}", value, epoch)
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, epoch)
            writer.add_scalar("validation/selection_score", score, epoch)
            writer.add_scalar("optimization/learning_rate", record["learning_rate"], epoch)
            writer.flush()
        improved = score > best_score
        alignment_improved = validation["mse"] < best_alignment
        best_score = max(best_score, score)
        best_alignment = min(best_alignment, validation["mse"])
        payload = {
            "schema_version": 2, "implementation": "EEG2Caption Compact shared",
            "method": method, "epoch": epoch, "model_state": model.state_dict(),
            "model_config": model_config, "normalization_mean": mean,
            "normalization_std": std, "eeg_scale": eeg_scale,
            "target_mean": target_mean, "validation": validation, "config": config,
            "best_overall_score": best_score, "best_alignment_mse": best_alignment,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "rng_state": capture_rng(generator),
        }
        atomic_save(payload, output_dir / "last.pt")
        if improved:
            atomic_save(payload, output_dir / "best.pt")
            atomic_save(payload, output_dir / "best_overall.pt")
            atomic_save(payload, output_dir / "best_semantic.pt")
        if alignment_improved:
            atomic_save(payload, output_dir / "best_alignment.pt")
        print(
            f"[compact-tora] epoch={epoch}/{epochs} loss={record['train_loss']:.4f} "
            f"val_mse={validation['mse']:.6f} cos={validation['token_cosine']:.4f} "
            f"R@1={validation['retrieval_top1']:.4f} R@5={validation['retrieval_top5']:.4f}",
            flush=True,
        )
    if start_epoch > epochs:
        print(f"[compact-tora] already completed {epochs} epochs", flush=True)
    if writer is not None:
        writer.close()
    atomic_json(
        {"schema_version": 1, "completed_epochs": epochs, "checkpoint": "last.pt"},
        output_dir / "completed.json",
    )
    print(f"[compact-tora] best checkpoint: {output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
