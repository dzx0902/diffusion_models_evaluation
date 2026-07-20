"""Train a fixed-shape latent resampler/decoder on cached Wan T5 states."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.wan_fixed_latent import (
    FixedLatentAutoencoder,
    FixedLatentConfig,
    reconstruction_stats,
    save_fixed_latent_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a nested-slot fixed Wan text latent.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fixed_latent" / "wan_128x512")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--max-slots", type=int, default=128)
    parser.add_argument("--slot-counts", type=int, nargs="+", default=[64, 68, 72, 96, 128])
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.25)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=500)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_state(record: dict[str, object], device: torch.device) -> torch.Tensor:
    state = torch.load(Path(str(record["path"])), map_location="cpu", weights_only=True)
    return state.to(device=device, dtype=torch.float32).unsqueeze(0)


def loss_for(model: FixedLatentAutoencoder, hidden: torch.Tensor, slots: int, cosine_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    reconstructed, _ = model(hidden, slots)
    mse = F.mse_loss(reconstructed.float(), hidden.float())
    cosine = F.cosine_similarity(reconstructed.float(), hidden.float(), dim=-1).mean()
    loss = mse + cosine_weight * (1.0 - cosine)
    return loss, reconstruction_stats(hidden, reconstructed)


def evaluate(
    model: FixedLatentAutoencoder,
    records: list[dict[str, object]],
    slots: list[int],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    totals = {slot: {"mse": 0.0, "cosine": 0.0} for slot in slots}
    with torch.inference_mode():
        for record in records:
            hidden = load_state(record, device)
            for slot in slots:
                reconstructed, _ = model(hidden, slot)
                stats = reconstruction_stats(hidden, reconstructed)
                totals[slot]["mse"] += stats["mse"]
                totals[slot]["cosine"] += stats["cosine"]
    count = max(len(records), 1)
    return {str(slot): {key: value / count for key, value in values.items()} for slot, values in totals.items()}


def main() -> None:
    args = parse_args()
    if any(slot < 1 or slot > args.max_slots for slot in args.slot_counts):
        raise ValueError("Every --slot-counts value must be within [1, --max-slots].")
    records = load_records(args.cache_dir / "index.jsonl")
    if len(records) < 10:
        raise ValueError("Need at least 10 cached prompts for a train/holdout split.")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    holdout_count = max(1, round(len(records) * args.holdout_fraction))
    valid_records, train_records = records[:holdout_count], records[holdout_count:]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = FixedLatentAutoencoder(
        FixedLatentConfig(latent_dim=args.latent_dim, max_slots=args.max_slots)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    best_cosine = float("-inf")

    with history_path.open("w", encoding="utf-8") as history:
        for step in range(1, args.steps + 1):
            record = train_records[rng.randrange(len(train_records))]
            slots = rng.choice(args.slot_counts)
            hidden = load_state(record, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, stats = loss_for(model, hidden, slots, args.cosine_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.eval_every != 0 and step != args.steps:
                continue
            model.eval()
            valid = evaluate(model, valid_records, args.slot_counts, device)
            summary = {"step": step, "train_slots": slots, "train_loss": float(loss.item()), **stats, "valid": valid}
            history.write(json.dumps(summary) + "\n")
            history.flush()
            selected_cosine = valid[str(max(args.slot_counts))]["cosine"]
            print(f"[fixed-latent-train] step={step} loss={loss.item():.6f} valid={valid}", flush=True)
            save_fixed_latent_checkpoint(args.output_dir / "last.pt", model, {"step": step, "valid": valid})
            if selected_cosine > best_cosine:
                best_cosine = selected_cosine
                save_fixed_latent_checkpoint(args.output_dir / "best.pt", model, {"step": step, "valid": valid})
    print(f"[fixed-latent-train] best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
