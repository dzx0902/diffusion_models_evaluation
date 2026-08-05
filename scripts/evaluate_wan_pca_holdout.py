"""Measure a fold-trained Wan token PCA projector on held-out captions.

This is an offline reconstruction control.  It never loads the Wan diffusion
model or EEG conditioner: it reconstructs cached native T5 states as
``(H - mean) @ W.T @ W + mean`` and reports error on the held-out videos.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wan token-PCA reconstruction on held-out captions.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Text cache created by cache_wan_text_states.py.")
    parser.add_argument("--projector", type=Path, required=True, help="Train-fold token_pca_projector.npz.")
    parser.add_argument("--video-manifest", type=Path, required=True, help="Full structured video manifest.")
    parser.add_argument("--split-plan", type=Path, required=True, help="Subject six-fold split plan.")
    parser.add_argument("--experiment", required=True, help="For example video_6fold_1.")
    parser.add_argument("--partition", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 768, 1024, 1536])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_partition_ids(plan_path: Path, experiment_name: str, partition: str) -> set[str]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [item for item in payload["experiments"] if item["name"] == experiment_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one experiment named {experiment_name!r} in {plan_path}")
    return set(matches[0][f"{partition}_video_ids"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    pca = np.load(args.projector)
    components = torch.from_numpy(pca["components"].astype(np.float32))
    center = torch.from_numpy(pca["mean"].astype(np.float32))
    if components.ndim != 2 or components.shape[1] != 4096 or center.shape != (4096,):
        raise ValueError("Projector must contain mean [4096] and components [K, 4096].")
    invalid_dims = [dim for dim in args.dims if dim < 1 or dim > components.shape[0]]
    if invalid_dims:
        raise ValueError(f"Requested dims {invalid_dims} outside projector range 1..{components.shape[0]}")

    cache_rows = read_jsonl(args.cache_dir / "index.jsonl")
    by_prompt = {str(row["prompt"]): row for row in cache_rows}
    if len(by_prompt) != len(cache_rows):
        raise ValueError("Text cache has duplicate prompt entries; cannot map held-out captions safely.")
    ids = load_partition_ids(args.split_plan, args.experiment, args.partition)
    records = [row for row in read_jsonl(args.video_manifest) if str(row["video_id"]) in ids]
    if len(records) != len(ids):
        found = {str(row["video_id"]) for row in records}
        raise ValueError(f"Split videos missing from manifest: {sorted(ids - found)[:10]}")

    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: str(row["video_id"])):
        prompt = str(record["caption"])
        cached = by_prompt.get(prompt)
        if cached is None:
            raise KeyError(f"No cached state for {record['video_id']}: {prompt!r}")
        state_path = Path(cached["path"])
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        hidden = torch.load(state_path, map_location="cpu", weights_only=True).float()
        if hidden.ndim != 2 or hidden.shape[1] != 4096:
            raise ValueError(f"Unexpected cached state shape for {record['video_id']}: {tuple(hidden.shape)}")
        centered = hidden - center
        norm = hidden.square().mean().clamp_min(1e-12)
        for dim in args.dims:
            basis = components[:dim]
            reconstructed = (centered @ basis.t()) @ basis + center
            diff = reconstructed - hidden
            cosine = torch.nn.functional.cosine_similarity(reconstructed, hidden, dim=-1).mean()
            rows.append(
                {
                    "video_id": record["video_id"],
                    "category_id": record["category_id"],
                    "tokens": hidden.shape[0],
                    "dim": dim,
                    "mse": float(diff.square().mean().item()),
                    "relative_mse": float((diff.square().mean() / norm).item()),
                    "mean_token_cosine": float(cosine.item()),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dim"]), "all")].append(row)
        grouped[(int(row["dim"]), f"category_{row['category_id']}")].append(row)
    for (dim, scope), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "dim": dim,
                "scope": scope,
                "video_count": len(values),
                "mean_mse": mean([float(value["mse"]) for value in values]),
                "mean_relative_mse": mean([float(value["relative_mse"]) for value in values]),
                "mean_token_cosine": mean([float(value["mean_token_cosine"]) for value in values]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_video.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    report = [
        "# Wan PCA Held-out Reconstruction",
        "",
        f"- Experiment: `{args.experiment}`",
        f"- Partition: `{args.partition}`",
        f"- Videos: {len(records)}",
        f"- Projector: `{args.projector}`",
        "",
        "## Overall",
        "",
        "| dim | videos | mean MSE | relative MSE | mean token cosine |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["scope"] == "all":
            report.append(
                f"| {row['dim']} | {row['video_count']} | {row['mean_mse']:.6f} | "
                f"{row['mean_relative_mse']:.6f} | {row['mean_token_cosine']:.6f} |"
            )
    report.extend(
        [
            "",
            "`relative MSE` is reconstruction MSE divided by the mean squared magnitude of the original Wan T5 state.",
            "This evaluates only the PCA round trip; it does not include EEG prediction or video generation.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[wan-pca-holdout] wrote {args.output_dir / 'report.md'} ({len(records)} videos)")


if __name__ == "__main__":
    main()
