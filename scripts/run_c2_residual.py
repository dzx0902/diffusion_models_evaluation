"""C2-v2: first-six video residual regression, contrastive learning and variance control."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from ms_video_eval.c2_residual import (
    ResidualEEG, fit_pca, encode, decode, contrastive_loss,
    variance_covariance, retrieval, within_category_permutation,
)
from ms_video_eval.eeg2caption_adapter import load_eeg2caption_fold, normalization_stats
from ms_video_eval.tora_conditioning import read_tora_condition_index, ToraPCAProjector
from scripts.train_compact_tora_alignment import atomic_json, atomic_save, load_target


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(handle)


def _digest(handle):
    result = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def prepare(args):
    if args.prepared.exists():
        raise FileExistsError(f"Use a new path; prepared data already exists: {args.prepared}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    if config["experiment"]["method"] != "tora_pca":
        raise ValueError("C2-v2 currently requires the existing Tora token PCA target cache")
    fold = load_eeg2caption_fold(ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
                               resolve(data["split_plan"]), data["fold"], data["sessions"], 800)
    selected = [i for i, video in enumerate(fold.video_ids) if video.split("-")[0] in {"01", "02", "03", "04", "05", "06"}]
    lookup = {old: new for new, old in enumerate(selected)}
    splits = {key: [lookup[i] for i in indices if i in lookup] for key, indices in fold.split_indices.items()}
    if any(not splits.get(key) for key in ("train", "validation", "test")):
        raise ValueError("Require nonempty train/validation/test partitions")
    ids = [fold.video_ids[i] for i in selected]
    eeg = fold.eeg[selected]
    scale = float(config["model"].get("eeg_scale", 1e6))
    mean, std = normalization_stats(eeg, splits["train"], scale)
    eeg = (eeg * scale - mean[None, :, :, None]) / std[None, :, :, None]
    index = read_tora_condition_index(resolve(data["tora_target_index"]))
    targets = torch.stack([load_target(index[video], "tora_pca") for video in ids])
    if targets.shape[1:] != (226, 512) or not torch.isfinite(targets).all():
        raise ValueError("Expected finite [video,226,512] targets")
    print(f"[c2-residual] fitting PCA on {len(splits['train'])} first-six train videos", flush=True)
    pca = fit_pca(targets[splits["train"]], args.dim, args.scale_floor)
    z = encode(targets, pca)
    floor_mse = (targets - decode(z, pca)).square().flatten(1).mean(1)
    raw_mean_mse = (targets - pca["mean"].reshape(226, 512)).square().flatten(1).mean(1)
    captions = [str(index[video].get("caption") or fold.records[video].caption).strip() for video in ids]
    if any(not value for value in captions):
        raise ValueError("Missing captions needed for multi-positive matching")
    labels = {caption: i for i, caption in enumerate(sorted(set(captions)))}
    projector = ToraPCAProjector.load(resolve(data["pca_projector"]), dim=512)
    protocol = {"schema_version": 1, "fold": data["fold"], "categories": ["01", "02", "03", "04", "05", "06"],
                "duration_sec": 4, "session_fusion": "three_session_feature_mean",
                "fit_partition": "train", "dim": args.dim, "scale_floor": args.scale_floor,
                "splits": {key: [ids[i] for i in indices] for key, indices in splits.items()},
                "source_config": str(args.config.resolve()),
                "source_hashes": {key: digest(resolve(data[key])) for key in
                                  ("trials", "semantic_labels", "split_plan", "tora_target_index", "pca_projector")},
                "token_pca_note": "Existing fold-specific token PCA retained; its upstream train-only provenance must be audited."}
    package = {"protocol": protocol, "eeg": eeg, "z": z, "pca": pca, "splits": splits,
               "ids": ids, "captions": captions, "caption_ids": torch.tensor([labels[c] for c in captions]),
               "categories": fold.category_targets[selected], "floor_mse": floor_mse,
               "raw_mean_mse": raw_mean_mse, "normalization_mean": mean, "normalization_std": std,
               "token_projector": {"mean": projector.mean, "components": projector.components}}
    args.prepared.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(package, args.prepared)
    atomic_json(protocol, args.prepared.with_suffix(".protocol.json"))
    print(f"[c2-residual] prepared={args.prepared}", flush=True)


@torch.no_grad()
def predict(model, eeg, device, batch_size):
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in eeg.split(batch_size)])


def measure(pred, package, indices):
    target = package["z"][indices]
    labels = package["caption_ids"][indices]
    metrics = retrieval(pred, target, labels[:, None] == labels[None, :], package["categories"][indices])
    error = ((pred - target) * package["pca"]["scale"]).square().sum(1)
    width = package["pca"]["mean"].numel()
    metrics["token_pca_space_mse"] = float((error / width + package["floor_mse"][indices]).mean())
    metrics["pca_reconstruction_floor_mse"] = float(package["floor_mse"][indices].mean())
    metrics["mean_condition_mse"] = float(package["raw_mean_mse"][indices].mean())
    return metrics


def train(args, package, fingerprint, output):
    if args.variant == "mean":
        raise ValueError("mean has no training; use --stage evaluate --variant mean")
    signature = {"prepared_sha256": fingerprint, "variant": args.variant, "seed": args.seed,
                 "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                 "weight_decay": args.weight_decay, "dropout": args.dropout,
                 "temperature": args.temperature, "patience": args.patience, "min_epochs": args.min_epochs}
    if (output / "completed.json").exists():
        completed = json.loads((output / "completed.json").read_text())
        if completed["signature"] != signature:
            raise ValueError("Completed run settings differ; use a new --output-root")
        print(f"[c2-residual] already completed {output}", flush=True)
        return
    model = ResidualEEG(package["z"].shape[1], args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    start, best, stale = 1, -float("inf"), 0
    last_path = output / "last.pt"
    if last_path.exists():
        if not args.resume:
            raise FileExistsError("Pass --resume to resume this run")
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["signature"] != signature:
            raise ValueError("Resume signature mismatch")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        start, best, stale = last["epoch"] + 1, last["best"], last["stale"]
        torch.set_rng_state(last["rng"])
        generator.set_state(last["loader_rng"])
        if torch.cuda.is_available() and last["cuda_rng"]:
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        # A crash may leave a history row newer than the last atomic checkpoint.
        history = output / "history.jsonl"
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            temporary = history.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in rows if row["epoch"] < start), encoding="utf-8")
            temporary.replace(history)
    elif (output / "history.jsonl").exists():
        raise FileExistsError("History exists without resumable last.pt; use a new output root")
    train_indices = package["splits"]["train"]
    valid_indices = package["splits"]["validation"]
    dataset = TensorDataset(package["eeg"][train_indices], package["z"][train_indices],
                            package["caption_ids"][train_indices])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    # One candidate per caption avoids duplicate-caption frequency bias in the training bank.
    unique = {}
    for i in train_indices:
        unique.setdefault(int(package["caption_ids"][i]), i)
    bank_indices = list(unique.values())
    bank = package["z"][bank_indices].to(args.device)
    bank_ids = package["caption_ids"][bank_indices].to(args.device)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(signature, output / "run_config.json")
    for epoch in range(start, args.epochs + 1):
        if epoch > args.min_epochs and stale >= args.patience:
            break
        model.train()
        total, count = 0.0, 0
        for eeg, target, labels in loader:
            pred = model(eeg.to(args.device))
            target = target.to(args.device)
            loss = F.mse_loss(pred, target)
            if args.variant in {"contrastive", "variance"}:
                loss = loss + contrastive_loss(pred, bank, labels.to(args.device)[:, None] == bank_ids[None, :], args.temperature)
            if args.variant == "variance":
                var, cov = variance_covariance(pred)
                loss = loss + 0.1 * var + 0.001 * cov
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(eeg)
            count += len(eeg)
        scheduler.step()
        metrics = measure(predict(model, package["eeg"][valid_indices], args.device, args.batch_size), package, valid_indices)
        score = metrics["within_category_mrr"]
        improved = score > best + 1e-6
        best, stale = (score, 0) if improved else (best, stale + 1)
        record = {"epoch": epoch, "train_loss": total / count, "validation": metrics}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "stale": stale,
                   "validation": metrics, "rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        if improved:
            atomic_save(payload, output / "best.pt")
        atomic_save(payload, last_path)
        print(f"[c2-residual] {args.variant} epoch={epoch} loss={total/count:.4f} val={json.dumps(metrics)}", flush=True)
    atomic_json({"signature": signature, "best": best, "status": "complete"}, output / "completed.json")


def evaluate(args, package, fingerprint, output):
    indices = package["splits"][args.partition]
    target = package["z"][indices]
    epoch = None
    if args.variant == "mean":
        pred = torch.zeros_like(target)
    else:
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["signature"]["prepared_sha256"] != fingerprint:
            raise ValueError("Prepared data differs from training")
        model = ResidualEEG(target.shape[1], checkpoint["signature"]["dropout"]).to(args.device)
        model.load_state_dict(checkpoint["model"])
        pred = predict(model, package["eeg"][indices], args.device, args.batch_size)
        epoch = checkpoint["epoch"]
    metrics = measure(pred, package, indices)
    zero = measure(torch.zeros_like(target), package, indices)
    generator = torch.Generator().manual_seed(args.seed + 10000)
    shuffled = []
    for _ in range(args.shuffles):
        permutation = within_category_permutation(package["categories"][indices], generator)
        shuffled.append(measure(pred[permutation], package, indices)["within_category_mrr"])
    result = {"schema_version": 1, "variant": args.variant, "partition": args.partition,
              "prepared_sha256": fingerprint, "checkpoint_epoch": epoch,
              "video_count": len(indices), "candidate_unit": "video; identical captions are multi-positive",
              "within_category": "oracle category diagnostic", "metrics": metrics, "mean_baseline": zero,
              "shuffle": {"repetitions": args.shuffles, "seed": args.seed + 10000,
                          "within_category_mrr": shuffled,
                          "mean_mrr": float(np.mean(shuffled)) if shuffled else None,
                          "note": "Within-category derangements; diagnostic differences, not a calibrated p-value"}}
    destination = output / args.partition
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(result, destination / "report.json")
    atomic_save({"predicted": pred, "target": target,
                 "video_ids": [package["ids"][i] for i in indices]}, destination / "residual_predictions.pt")
    if args.export:
        projector = ToraPCAProjector(**package["token_projector"])
        directory = destination / "video_aggregated"
        directory.mkdir(exist_ok=True)
        rows = []
        for position, i in enumerate(indices):
            hidden = projector.decode(decode(pred[position:position+1], package["pca"])[0])
            if hidden.shape != (226, 4096) or not torch.isfinite(hidden).all():
                raise ValueError("Invalid exported Tora condition")
            path = directory / f"{package['ids'][i]}.pt"
            atomic_save({"schema_version": 2, "video_id": package["ids"][i], "hidden_state": hidden,
                         "caption": "EEG C2-v2 continuous residual condition", "source_method": "c2_residual",
                         "source_checkpoint": str((output / "best.pt").resolve()) if epoch else None}, path)
            rows.append({"video_id": package["ids"][i], "condition_path": str(path.resolve()), "trial_count": 3})
        (destination / "video_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eeg_semantic/method_c_pca_tora.yaml")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/eeg_semantic/c2_residual_v2")
    parser.add_argument("--variant", choices=("mean", "regression", "contrastive", "variance"), default="regression")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--scale-floor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.shuffles < 0:
        parser.error("epochs/batch-size/patience/threads must be positive and shuffles nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.stage == "prepare":
        prepare(args)
        return
    fingerprint = digest(args.prepared)
    package = torch.load(args.prepared, map_location="cpu", weights_only=False)
    output = args.output_root / f"dim{package['z'].shape[1]}" / args.variant / f"seed{args.seed}"
    if args.stage == "train":
        train(args, package, fingerprint, output)
    else:
        evaluate(args, package, fingerprint, output)


if __name__ == "__main__":
    main()
