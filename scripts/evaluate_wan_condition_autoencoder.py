"""Evaluate a frozen Wan condition autoencoder on a split partition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.wan_condition_autoencoder import WanConditionAutoencoder, WanConditionAutoencoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Wan condition autoencoder on held-out captions.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--partition", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    experiment = next((row for row in plan["experiments"] if row["name"] == args.experiment), None)
    if experiment is None:
        raise KeyError(f"Unknown experiment {args.experiment!r}")
    ids = set(experiment[f"{args.partition}_video_ids"])
    manifest = {str(row["video_id"]): row for row in read_jsonl(args.video_manifest)}
    if ids - set(manifest):
        raise KeyError(f"Manifest missing split IDs: {sorted(ids - set(manifest))[:5]}")
    cache = {str(row["prompt"]): Path(row["path"]) for row in read_jsonl(args.cache_dir / "index.jsonl")}
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = WanConditionAutoencoderConfig(**checkpoint["config"])
    model = WanConditionAutoencoder(config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    rows = []
    with torch.inference_mode():
        for video_id in sorted(ids):
            record = manifest[video_id]
            state_path = cache.get(str(record["caption"]))
            if state_path is None:
                raise KeyError(f"No cached state for {video_id}")
            state = torch.load(state_path, map_location="cpu", weights_only=True).float()
            tokens = state.shape[0]
            padded = torch.zeros(1, config.slots, config.input_dim, device=device)
            padded[0, :tokens] = state.to(device)
            _, reconstructed = model(padded, torch.tensor([tokens], device=device))
            reconstructed = reconstructed[0, :tokens].cpu()
            rows.append({
                "video_id": video_id,
                "category_id": record["category_id"],
                "tokens": tokens,
                "mse": float((reconstructed - state).square().mean().item()),
                "relative_mse": float(((reconstructed - state).square().mean() / state.square().mean().clamp_min(1e-12)).item()),
                "mean_token_cosine": float(F.cosine_similarity(reconstructed, state, dim=-1).mean().item()),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_video.csv", rows)
    mean = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    ordered = sorted(rows, key=lambda row: float(row["mean_token_cosine"]))
    summary = {
        "checkpoint": str(args.checkpoint), "experiment": args.experiment, "partition": args.partition,
        "video_count": len(rows), "mean_mse": mean("mse"), "mean_relative_mse": mean("relative_mse"),
        "mean_token_cosine": mean("mean_token_cosine"), "worst_token_cosine": ordered[:10],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Wan Condition Autoencoder Held-out Reconstruction", "", f"- Experiment: `{args.experiment}`", f"- Partition: `{args.partition}`", f"- Videos: {len(rows)}", "", "| mean MSE | relative MSE | mean token cosine |", "| ---: | ---: | ---: |", f"| {summary['mean_mse']:.6f} | {summary['mean_relative_mse']:.6f} | {summary['mean_token_cosine']:.6f} |", "", "The ten lowest-cosine videos are in `summary.json`; all rows are in `per_video.csv`."]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[wan-ae-eval] wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
