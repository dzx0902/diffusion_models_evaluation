"""Train a fixed-slot continuous autoencoder on cached Wan T5 states."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.wan_condition_autoencoder import (
    WanConditionAutoencoder,
    WanConditionAutoencoderConfig,
    autoencoder_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a continuous fixed-slot autoencoder for Wan conditions.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True, help="JSONL/txt captions used to fit the autoencoder.")
    parser.add_argument("--validation-prompts", type=Path, required=True, help="Held-out JSONL/txt captions for validation.")
    parser.add_argument(
        "--allow-prompt-overlap",
        action="store_true",
        help=(
            "Allow identical caption states in train and validation for codec pretraining only. "
            "This invalidates caption-held-out validation and is never the default."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.25)
    parser.add_argument("--pooled-weight", type=float, default=0.25)
    parser.add_argument("--padding-weight", type=float, default=0.01)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def prompts(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = [str(row.get("prompt") or row.get("caption") or "").strip() for row in rows]
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(value for value in values if value))


def cache_index(cache_dir: Path) -> dict[str, Path]:
    rows = [json.loads(line) for line in (cache_dir / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    index = {str(row["prompt"]): Path(row["path"]) for row in rows}
    if len(index) != len(rows):
        raise ValueError("Cache index has duplicate prompt records.")
    return index


class StateDataset(Dataset):
    def __init__(self, state_paths: list[Path], slots: int) -> None:
        self.state_paths = state_paths
        self.slots = slots

    def __len__(self) -> int:
        return len(self.state_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        state = torch.load(self.state_paths[index], map_location="cpu", weights_only=True).float()
        if state.ndim != 2 or state.shape[1] != 4096 or not 1 <= state.shape[0] <= self.slots:
            raise ValueError(f"Invalid cached Wan state {self.state_paths[index]}: {tuple(state.shape)}")
        padded = torch.zeros(self.slots, 4096, dtype=torch.float32)
        padded[: state.shape[0]] = state
        return {"state": padded, "tokens": torch.tensor(state.shape[0], dtype=torch.long)}


def evaluate(model: WanConditionAutoencoder, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    totals = {"loss": 0.0, "valid_mse": 0.0, "padding_mse": 0.0, "token_cosine_loss": 0.0, "pooled_cosine_loss": 0.0}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            state, lengths = batch["state"].to(device), batch["tokens"].to(device)
            _, reconstructed = model(state, lengths)
            _, values = autoencoder_loss(reconstructed, state, lengths, args.cosine_weight, args.pooled_weight, args.padding_weight)
            size = state.shape[0]
            for key, value in values.items():
                totals[key] += value * size
            count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    index = cache_index(args.cache_dir)
    train_prompts, validation_prompts = prompts(args.train_prompts), prompts(args.validation_prompts)
    overlap = set(train_prompts) & set(validation_prompts)
    if overlap and not args.allow_prompt_overlap:
        raise ValueError(f"Train/validation prompt overlap: {next(iter(overlap))!r}")
    if overlap:
        print(
            f"[wan-ae] WARNING: allowing {len(overlap)} train/validation duplicate captions for codec pretraining; "
            "validation is not caption-held-out.",
            flush=True,
        )
    missing = [prompt for prompt in train_prompts + validation_prompts if prompt not in index]
    if missing:
        raise KeyError(f"Cached states missing for {len(missing)} prompts, e.g. {missing[0]!r}")
    train_set = StateDataset([index[prompt] for prompt in train_prompts], args.slots)
    validation_set = StateDataset([index[prompt] for prompt in validation_prompts], args.slots)
    device = torch.device(args.device)
    config = WanConditionAutoencoderConfig(
        slots=args.slots, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        heads=args.heads, dropout=args.dropout,
    )
    model = WanConditionAutoencoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best = 1, float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_config = WanConditionAutoencoderConfig(**checkpoint["config"])
        if checkpoint_config != config:
            raise ValueError("Resume configuration differs from requested configuration.")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_valid_loss", best))
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda"),
        "validation": DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda"),
    }
    history = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            state, lengths = batch["state"].to(device), batch["tokens"].to(device)
            optimizer.zero_grad(set_to_none=True)
            _, reconstructed = model(state, lengths)
            loss, _ = autoencoder_loss(reconstructed, state, lengths, args.cosine_weight, args.pooled_weight, args.padding_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        valid = evaluate(model, loaders["validation"], device, args)
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "valid": valid}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if valid["loss"] < best:
            best = valid["loss"]
        payload = {"config": asdict(config), "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best_valid_loss": best, "valid": valid, "args": vars(args)}
        torch.save(payload, args.output_dir / "last.pt")
        if valid["loss"] == best:
            torch.save(payload, args.output_dir / "best.pt")
        print(f"[wan-ae] epoch={epoch} valid={valid}", flush=True)
    print(f"[wan-ae] best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
