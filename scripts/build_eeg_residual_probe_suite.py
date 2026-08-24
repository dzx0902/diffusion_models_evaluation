"""Select top EEG trials using centered residual retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        -int(float(row["residual_prompt_retrieval_top1"])),
        -float(row["residual_prompt_retrieval_margin"]),
        -float(row["residual_pooled_cosine"]),
        float(row["residual_mse_fraction"]),
        str(row["video_id"]),
        str(row["session"]),
        int(row["trial_index"]),
    )


def select_residual_trials(
    metrics: list[dict[str, str]],
    trials: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "video_id",
        "session",
        "trial_index",
        "residual_prompt_retrieval_top1",
        "residual_prompt_retrieval_margin",
        "residual_pooled_cosine",
        "residual_mse_fraction",
        "residual_nearest_video_id",
        "residual_nearest_prompt",
    }
    missing = required - set(metrics[0] if metrics else ())
    if missing:
        raise ValueError(f"Metrics are missing residual fields: {sorted(missing)}")

    trial_lookup = {
        (row["video_id"], row["session"], int(row["trial_index"])): row
        for row in trials
    }
    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    for metric in sorted(metrics, key=metric_sort_key):
        video_id = metric["video_id"]
        if video_id in used_videos:
            continue
        key = (video_id, metric["session"], int(metric["trial_index"]))
        if key not in trial_lookup:
            raise KeyError(f"Trial metadata missing for {key}")
        trial = trial_lookup[key]
        selected.append(
            {
                "video_id": video_id,
                "category_id": video_id.split("-", 1)[0],
                "session": metric["session"],
                "trial_index": int(metric["trial_index"]),
                "duration_sec": float(trial["duration_sec"]),
                "length_samples": int(trial["length_samples"]),
                "residual_prompt_retrieval_top1": int(
                    float(metric["residual_prompt_retrieval_top1"])
                ),
                "residual_prompt_retrieval_margin": float(
                    metric["residual_prompt_retrieval_margin"]
                ),
                "residual_pooled_cosine": float(metric["residual_pooled_cosine"]),
                "residual_mse_fraction": float(metric["residual_mse_fraction"]),
                "nearest_exact_video_id": metric["residual_nearest_video_id"],
                "nearest_exact_prompt": metric["residual_nearest_prompt"],
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
    selected = select_residual_trials(read_csv(args.metrics), read_csv(args.trials), args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    for rank, row in enumerate(selected, start=1):
        print(
            f"[residual-suite] {rank}: {row['video_id']}/{row['session']} "
            f"nearest={row['nearest_exact_video_id']} "
            f"top1={row['residual_prompt_retrieval_top1']} "
            f"margin={row['residual_prompt_retrieval_margin']:.6f}",
            flush=True,
        )
    print(f"[residual-suite] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
