"""Select a deterministic category-balanced held-out EEG generation suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--session", default="session3")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--categories", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_experiment(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment = next((row for row in payload["experiments"] if row["name"] == name), None)
    if experiment is None:
        raise KeyError(f"Unknown split experiment: {name}")
    return experiment


def select_suite(
    rows: list[dict[str, str]],
    test_ids: set[str],
    *,
    categories: list[str],
    session: str,
    duration_sec: float,
    per_category: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["video_id"] in test_ids
        and row["session"] == session
        and abs(float(row["duration_sec"]) - duration_sec) <= 1e-6
    ]
    selected = []
    for category in categories:
        candidates = sorted(
            (row for row in eligible if row["video_id"].replace("_", "-").startswith(category + "-")),
            key=lambda row: row["video_id"],
        )
        if len(candidates) < per_category:
            raise ValueError(
                f"Category {category} has {len(candidates)} eligible trials; need {per_category}"
            )
        # Evenly spaced positions avoid always selecting the lexicographically first videos.
        positions = [int((index + 0.5) * len(candidates) / per_category) for index in range(per_category)]
        for row in (candidates[min(position, len(candidates) - 1)] for position in positions):
            selected.append(
                {
                    "video_id": row["video_id"],
                    "category_id": category,
                    "session": session,
                    "trial_index": int(row["trial_index"]),
                    "duration_sec": float(row["duration_sec"]),
                    "length_samples": int(row["length_samples"]),
                }
            )
    return selected


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    experiment = read_experiment(args.split_plan, args.experiment)
    suite = select_suite(
        rows,
        set(experiment["test_video_ids"]),
        categories=args.categories,
        session=args.session,
        duration_sec=args.duration_sec,
        per_category=args.per_category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in suite:
            handle.write(json.dumps(row) + "\n")
    print(f"[eeg-generation-suite] wrote {len(suite)} trials to {args.output}")


if __name__ == "__main__":
    main()
