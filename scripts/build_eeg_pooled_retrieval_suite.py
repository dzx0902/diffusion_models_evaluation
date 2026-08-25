"""Export top pooled-retrieval EEG trials and their nearest exact conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_targets(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["video_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def select_trials(
    metrics: list[dict[str, str]],
    targets: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "video_id",
        "session",
        "trial_index",
        "prompt",
        "rank",
        "reciprocal_rank",
        "cosine",
        "standardized_mse",
        "energy_ratio",
        "nearest_prompt",
        "nearest_video_id",
        "nearest_cosine",
        "retrieval_margin",
    }
    missing = required - set(metrics[0] if metrics else ())
    if missing:
        raise ValueError(f"Metrics missing fields: {sorted(missing)}")
    ordered = sorted(
        metrics,
        key=lambda row: (
            float(row["rank"]),
            -float(row["retrieval_margin"]),
            -float(row["cosine"]),
            float(row["standardized_mse"]),
            row["video_id"],
            row["session"],
        ),
    )
    selected = []
    used_videos: set[str] = set()
    for row in ordered:
        video_id = row["video_id"]
        if video_id in used_videos:
            continue
        nearest_id = row["nearest_video_id"]
        nearest_target = targets[nearest_id]
        selected.append(
            {
                "video_id": video_id,
                "category_id": video_id.split("-", 1)[0],
                "session": row["session"],
                "trial_index": int(row["trial_index"]),
                "prompt": row["prompt"],
                "rank": float(row["rank"]),
                "reciprocal_rank": float(row["reciprocal_rank"]),
                "residual_cosine": float(row["cosine"]),
                "residual_mse": float(row["standardized_mse"]),
                "energy_ratio": float(row["energy_ratio"]),
                "retrieval_margin": float(row["retrieval_margin"]),
                "nearest_exact_video_id": nearest_id,
                "nearest_exact_prompt": row["nearest_prompt"],
                "nearest_exact_cosine": float(row["nearest_cosine"]),
                "nearest_exact_latent_path": nearest_target["latent_path"],
            }
        )
        used_videos.add(video_id)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        raise ValueError(f"Requested {top_k} unique videos, found {len(selected)}")
    return selected


def main() -> None:
    args = parse_args()
    selected = select_trials(read_csv(args.metrics), read_targets(args.targets), args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    for rank, row in enumerate(selected, start=1):
        print(
            f"[pooled-suite] {rank}: {row['video_id']}/{row['session']} "
            f"retrieval_rank={row['rank']:.1f} nearest={row['nearest_exact_video_id']}",
            flush=True,
        )
    print(f"[pooled-suite] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
