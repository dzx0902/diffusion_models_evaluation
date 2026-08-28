"""Train a nonlinear Tora text bottleneck using train-split videos only."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_data import load_video_partitions
from ms_video_eval.tora_conditioning import load_tora_condition, read_tora_condition_index
from ms_video_eval.tora_text_autoencoder import (
    ToraTextAutoencoder, ToraTextAutoencoderConfig, reconstruction_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--fold", default="video_6fold_1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bottleneck-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class StateDataset(Dataset):
    def __init__(self, ids: list[str], index: dict[str, dict[str, object]]) -> None:
        self.ids, self.index = ids, index
    def __len__(self) -> int:
        return len(self.ids)
    def __getitem__(self, item: int) -> torch.Tensor:
        return load_tora_condition(Path(str(self.index[self.ids[item]]["condition_path"]))).hidden_state


@torch.no_grad()
def evaluate(model: ToraTextAutoencoder, loader: DataLoader, device: torch.device, cosine: float) -> float:
    model.eval()
    total = 0.0
    for states in loader:
        states = states.to(device)
        loss, _ = reconstruction_loss(model(states)["reconstruction"], states, cosine)
        total += float(loss) * states.shape[0]
    return total / max(1, len(loader.dataset))


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    partitions = load_video_partitions(args.split_plan, args.fold)
    index = read_tora_condition_index(args.index)
    if (partitions["train"] | partitions["validation"]) - set(index):
        raise KeyError("Text cache is incomplete for train/validation")
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(StateDataset(sorted(partitions["train"]), index), batch_size=args.batch_size,
                       shuffle=True, generator=generator)
    valid = DataLoader(StateDataset(sorted(partitions["validation"]), index), batch_size=args.batch_size)
    device = torch.device(args.device)
    config = ToraTextAutoencoderConfig(hidden_dim=args.hidden_dim, bottleneck_dim=args.bottleneck_dim)
    model = ToraTextAutoencoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    training_config = {
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "cosine_weight": args.cosine_weight, "seed": args.seed,
    }
    best = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    history_path = args.output_dir / "history.jsonl"
    if last_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing autoencoder run: {last_path}")
    start_epoch = 1
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(last_path)
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        required = {"optimizer_state", "rng_state", "generator_state", "best_loss"}
        if required - set(checkpoint):
            raise ValueError("Checkpoint predates exact autoencoder resume support")
        if (checkpoint["config"] != model.config_dict() or checkpoint["fold"] != args.fold
                or checkpoint.get("training_config") != training_config):
            raise ValueError("Autoencoder resume config/fold mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        random.setstate(checkpoint["rng_state"]["python"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
        generator.set_state(checkpoint["generator_state"])
        best = float(checkpoint["best_loss"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total = 0.0
        for states in train:
            states = states.to(device); optimizer.zero_grad(set_to_none=True)
            loss, _ = reconstruction_loss(model(states)["reconstruction"], states, args.cosine_weight)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * states.shape[0]
        validation = evaluate(model, valid, device, args.cosine_weight)
        record = {"epoch": epoch, "train_loss": total / len(train.dataset), "validation_loss": validation}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {"schema_version": 1, "epoch": epoch, "config": model.config_dict(),
                   "training_config": training_config,
                   "state_dict": model.state_dict(), "split_plan": str(args.split_plan.resolve()),
                   "fold": args.fold, "fitted_partition": "train", "best_loss": min(best, validation),
                   "optimizer_state": optimizer.state_dict(), "generator_state": generator.get_state(),
                   "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                                 "torch": torch.get_rng_state(),
                                 "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        temporary = last_path.with_suffix(".pt.tmp"); torch.save(payload, temporary); os.replace(temporary, last_path)
        if validation < best:
            best = validation
            best_path = args.output_dir / "best.pt"; temporary = best_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary); os.replace(temporary, best_path)
        print(f"[tora-autoencoder] epoch={epoch}/{args.epochs} valid={validation:.6f}", flush=True)


if __name__ == "__main__":
    main()
