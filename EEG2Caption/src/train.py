"""Train the Compact EEG classifier using shared three-session video splits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    CompactEEGClassifier,
    ThreeSessionDataset,
    evaluate,
    load_label_package,
    load_three_sessions,
    normalization_stats,
    pair_label_matrix,
    resolve_subject,
    save_json,
    seed_everything,
    stratified_video_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/userhome2/zhoutianyi/Dataset/Multi-Object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="zhoutianyi")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--label-package", type=Path,
        default=PROJECT_ROOT / "data/video_multilabels_2object.pt",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--experiment", default="compact_eeg_3session_fusion_test20")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--eeg-scale", type=float, default=1e6)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--test-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-filters", type=int, default=16)
    parser.add_argument("--spatial-multiplier", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--fused-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--time-mask-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def make_loader(
    dataset: ThreeSessionDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def save_checkpoint(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu.")
    if args.num_samples != 800:
        raise ValueError("This released protocol uses exactly 800 samples (4 s at 200 Hz).")
    if args.val_per_class != 8 or args.test_per_class != 16:
        print("warning: split differs from released train324/val48/test96 protocol")
    seed_everything(args.seed)
    device = torch.device(args.device)

    package = load_label_package(args.label_package)
    subject_dir = resolve_subject(args.dataset_root / "Subjects", args.subject)
    output_dir = args.output_root / args.experiment / subject_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading {} EEG from session1/2/3 ...".format(subject_dir.name), flush=True)
    eeg, data_metadata = load_three_sessions(
        subject_dir, package["ids"], args.num_samples
    )
    splits = stratified_video_split(
        package["pair_indices"], args.val_per_class, args.test_per_class, args.seed
    )
    mean, std = normalization_stats(eeg, splits["train"], args.eeg_scale)
    split_record = {
        "seed": args.seed,
        "strategy": "six-pair stratified unique-video split shared by all sessions",
        "train_indices": splits["train"],
        "val_indices": splits["val"],
        "test_indices": splits["test"],
        "train_ids": [package["ids"][index] for index in splits["train"]],
        "val_ids": [package["ids"][index] for index in splits["val"]],
        "test_ids": [package["ids"][index] for index in splits["test"]],
    }
    save_json(split_record, output_dir / "splits.json")

    shared = dict(
        eeg=eeg,
        labels=package["labels"],
        pair_indices=package["pair_indices"],
        cardinalities=package["cardinalities"],
        ids=package["ids"],
        mean=mean,
        std=std,
        eeg_scale=args.eeg_scale,
    )
    train_set = ThreeSessionDataset(
        **shared,
        indices=splits["train"],
        training=True,
        noise_std=args.noise_std,
        time_mask_samples=args.time_mask_samples,
    )
    validation_set = ThreeSessionDataset(**shared, indices=splits["val"])
    train_loader = make_loader(
        train_set, args.batch_size, True, args.num_workers, device
    )
    validation_loader = make_loader(
        validation_set, args.batch_size, False, args.num_workers, device
    )

    model_config = {
        "num_channels": 62,
        "num_objects": len(package["object_names"]),
        "num_pairs": len(package["pair_prefixes"]),
        "temporal_filters": args.temporal_filters,
        "spatial_multiplier": args.spatial_multiplier,
        "feature_dim": args.feature_dim,
        "dropout": args.dropout,
    }
    model = CompactEEGClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.05,
    )
    train_labels = package["labels"][splits["train"]]
    pos_weight = (
        (len(train_labels) - train_labels.sum(0))
        / train_labels.sum(0).clamp_min(1)
    ).to(device)
    pair_labels = pair_label_matrix(package)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    best_score = -math.inf

    print(
        "split: train={} val={} held-out test={}; three sessions/video; parameters={:,}".format(
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
            sum(parameter.numel() for parameter in model.parameters()),
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, examples = 0.0, 0
        for batch in train_loader:
            eeg_batch = batch["eeg"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            pairs = batch["pair_index"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(eeg_batch)
                expanded_labels = labels[:, None, :].expand_as(
                    output["session_object_logits"]
                )
                expanded_pairs = pairs[:, None].expand(
                    -1, output["session_pair_logits"].shape[1]
                )
                session_object_loss = F.binary_cross_entropy_with_logits(
                    output["session_object_logits"], expanded_labels,
                    pos_weight=pos_weight,
                )
                session_pair_loss = F.cross_entropy(
                    output["session_pair_logits"].reshape(
                        -1, len(package["pair_prefixes"])
                    ),
                    expanded_pairs.reshape(-1),
                )
                fused_object_loss = F.binary_cross_entropy_with_logits(
                    output["fused_object_logits"], labels, pos_weight=pos_weight
                )
                fused_pair_loss = F.cross_entropy(output["fused_pair_logits"], pairs)
                normalized = F.normalize(output["features"], dim=-1)
                consistency_loss = (
                    normalized - normalized.mean(dim=1, keepdim=True)
                ).pow(2).mean()
                loss = (
                    session_object_loss
                    + args.pair_weight * session_pair_loss
                    + args.fused_weight * (
                        fused_object_loss + args.pair_weight * fused_pair_loss
                    )
                    + args.consistency_weight * consistency_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.shape[0]
            examples += labels.shape[0]
        scheduler.step()

        validation_metrics, _ = evaluate(
            model, validation_loader, device, pair_labels, package["object_names"]
        )
        score = float(validation_metrics["fused"]["macro_ap"])
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, examples),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation_metrics,
        }
        history.append(record)
        checkpoint = {
            "schema_version": 1,
            "model_name": "CompactEEGClassifier",
            "epoch": epoch,
            "best_validation_macro_ap": max(best_score, score),
            "model_state": model.state_dict(),
            "model_config": model_config,
            "normalization_mean": mean,
            "normalization_std": std,
            "eeg_scale": args.eeg_scale,
            "num_samples": args.num_samples,
            "splits": splits,
            "ids": package["ids"],
            "object_names": package["object_names"],
            "pair_prefixes": package["pair_prefixes"],
            "pair_names": package["pair_names"],
            "pair_objects": package["pair_objects"],
            "subject": subject_dir.name,
            "subject_dir": str(subject_dir.resolve()),
            "data_metadata": data_metadata,
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        save_checkpoint(checkpoint, output_dir / "last.pt")
        improved = score > best_score
        if improved:
            best_score = score
            checkpoint["best_validation_macro_ap"] = score
            save_checkpoint(checkpoint, output_dir / "best.pt")
        print(
            "epoch {:03d}/{:03d} loss={:.4f} val fused: mAP={:.4f} top2_exact={:.4f}{}".format(
                epoch, args.epochs, record["train_loss"], score,
                validation_metrics["fused"]["top2_exact_set_accuracy"],
                " *" if improved else "",
            ),
            flush=True,
        )
        save_json(history, output_dir / "history.json")

    summary = {
        "status": "training_complete_test_not_evaluated",
        "subject": subject_dir.name,
        "best_validation_macro_ap": best_score,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "next_step": "Run scripts/infer_and_caption.sh on best.pt.",
    }
    save_json(summary, output_dir / "training_summary.json")
    print("best checkpoint:", output_dir / "best.pt")
    print("test set was not used during model selection")


if __name__ == "__main__":
    main()
