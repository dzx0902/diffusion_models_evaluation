"""Evaluate train-only Tora PCA round trips on held-out captions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_data import load_video_partitions
from ms_video_eval.tora_conditioning import (
    ToraPCAProjector,
    load_tora_condition,
    read_tora_condition_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--fold", default="video_6fold_1")
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    parser.add_argument("--dims", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.projector.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing leakage-audit metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fitted_partition") != "train" or metadata.get("fold") != args.fold:
        raise ValueError("PCA projector was not fitted on this fold's training partition")
    index = read_tora_condition_index(args.index)
    ids = sorted(load_video_partitions(args.split_plan, args.fold)[args.partition])
    full_projector = ToraPCAProjector.load(args.projector)
    if any(dim < 1 or dim > full_projector.components.shape[0] for dim in args.dims):
        raise ValueError("Requested PCA dimension exceeds saved components")
    per_video = []
    accumulators = {
        dim: {"mse": 0.0, "relative_mse": 0.0, "cosine": 0.0} for dim in args.dims
    }
    for count, video_id in enumerate(ids, 1):
        state = load_tora_condition(Path(index[video_id]["condition_path"])).hidden_state
        energy = state.pow(2).mean().clamp_min(1e-12)
        for dim in args.dims:
            projector = ToraPCAProjector(
                full_projector.mean, full_projector.components[:dim]
            )
            restored = projector.decode(projector.encode(state))
            mse = F.mse_loss(restored, state)
            relative = mse / energy
            cosine = F.cosine_similarity(restored, state, dim=-1).mean()
            values = {
                "video_id": video_id,
                "dim": dim,
                "mse": float(mse),
                "relative_mse": float(relative),
                "token_cosine": float(cosine),
            }
            per_video.append(values)
            accumulators[dim]["mse"] += values["mse"]
            accumulators[dim]["relative_mse"] += values["relative_mse"]
            accumulators[dim]["cosine"] += values["token_cosine"]
        if count % 25 == 0:
            print(f"[tora-pca-holdout] {count}/{len(ids)}", flush=True)
    summary = [
        {
            "dim": dim,
            "videos": len(ids),
            "mean_mse": accumulators[dim]["mse"] / len(ids),
            "mean_relative_mse": accumulators[dim]["relative_mse"] / len(ids),
            "mean_token_cosine": accumulators[dim]["cosine"] / len(ids),
        }
        for dim in args.dims
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_video.csv", per_video), ("summary.csv", summary)):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "schema_version": 1,
        "projector": str(args.projector.resolve()),
        "projector_metadata": metadata,
        "split_plan": str(args.split_plan.resolve()),
        "fold": args.fold,
        "partition": args.partition,
        "summary": summary,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
