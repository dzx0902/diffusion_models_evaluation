"""Audit a duration-restricted EEG-to-Wan split before training or evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_protocol import filter_trial_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit duration, sessions, targets, and video-held-out partitions.")
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--sessions", nargs="+", default=["session1", "session2", "session3"])
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_target_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        str(json.loads(line)["video_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    args = parse_args()
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = filter_trial_duration(all_rows, args.duration_sec)
    if not rows:
        raise ValueError(f"No trials found for duration_sec={args.duration_sec}")

    plan = json.loads(args.split_plan.read_text(encoding="utf-8"))
    experiment = next((item for item in plan["experiments"] if item["name"] == args.experiment), None)
    if experiment is None:
        raise KeyError(f"Unknown experiment {args.experiment!r}")

    partitions = {
        name: set(experiment[f"{name}_video_ids"])
        for name in ("train", "validation", "test")
    }
    names = list(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = partitions[left] & partitions[right]
            if overlap:
                raise ValueError(f"{left}/{right} video overlap: {sorted(overlap)[:5]}")

    requested_sessions = set(args.sessions)
    trial_keys: set[tuple[str, str]] = set()
    rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["session"] not in requested_sessions:
            continue
        key = (row["video_id"], row["session"])
        if key in trial_keys:
            raise ValueError(f"Duplicate trial key after filtering: {key}")
        trial_keys.add(key)
        rows_by_video[row["video_id"]].append(row)

    duration_ids = set(rows_by_video)
    target_ids = read_target_ids(args.targets)
    reports: dict[str, dict[str, Any]] = {}
    for partition, planned_ids in partitions.items():
        video_ids = planned_ids & duration_ids
        partition_rows = [row for video_id in video_ids for row in rows_by_video[video_id]]
        for video_id in sorted(video_ids):
            found_sessions = {row["session"] for row in rows_by_video[video_id]}
            if found_sessions != requested_sessions:
                raise ValueError(
                    f"{partition}/{video_id}: sessions={sorted(found_sessions)}, "
                    f"expected={sorted(requested_sessions)}"
                )
        missing_targets = sorted(video_ids - target_ids) if target_ids is not None else []
        if missing_targets:
            raise ValueError(f"{partition}: missing targets for {missing_targets[:5]}")
        reports[partition] = {
            "planned_video_count_all_durations": len(planned_ids),
            "selected_video_count": len(video_ids),
            "selected_trial_count": len(partition_rows),
            "category_video_counts": dict(
                sorted(Counter(rows_by_video[video_id][0]["category_id"] for video_id in video_ids).items())
            ),
            "session_trial_counts": dict(sorted(Counter(row["session"] for row in partition_rows).items())),
        }

    covered_ids = set().union(*partitions.values())
    outside = duration_ids - covered_ids
    if outside:
        raise ValueError(f"Duration-filtered videos absent from the split plan: {sorted(outside)[:5]}")

    summary = {
        "subject": sorted({row["subject"] for row in rows}),
        "experiment": args.experiment,
        "duration_sec": args.duration_sec,
        "sessions": sorted(requested_sessions),
        "all_trial_count": len(all_rows),
        "selected_trial_count": len(rows),
        "excluded_trial_count": len(all_rows) - len(rows),
        "selected_video_count": len(duration_ids),
        "partitions": reports,
        "status": "PASS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EEG-to-Wan Protocol Audit",
        "",
        f"- Status: **{summary['status']}**",
        f"- Experiment: `{args.experiment}`",
        f"- Duration: **{args.duration_sec:g} seconds only**",
        f"- Trials: {len(rows)} selected / {len(all_rows)} total",
        f"- Videos: {len(duration_ids)}",
        "",
        "| partition | videos | trials | sessions | categories |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for partition, report in reports.items():
        sessions = ", ".join(f"{key}:{value}" for key, value in report["session_trial_counts"].items())
        categories = ", ".join(f"{key}:{value}" for key, value in report["category_video_counts"].items())
        lines.append(
            f"| {partition} | {report['selected_video_count']} | {report['selected_trial_count']} | "
            f"{sessions} | {categories} |"
        )
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[eeg-audit] wrote {args.output_dir / 'audit.md'}", flush=True)


if __name__ == "__main__":
    main()
