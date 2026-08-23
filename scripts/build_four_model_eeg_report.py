"""Aggregate matched YOLO scores for the four-model EEG benchmark."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


GROUPS = {
    "wan_pca512_exact": "Wan2.2 PCA-512 (exact)",
    "wan_pca512_eeg": "Wan2.2 PCA-512 (EEG)",
    "animatediff_eeg": "AnimateDiff (EEG)",
    "zeroscope_eeg": "ZeroScope (EEG)",
    "cogvideox2b_eeg": "CogVideoX-2B (EEG)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def classify_video(filename: str) -> str:
    for key in GROUPS:
        if f"_{key}_seed" in filename:
            return key
    raise ValueError(f"Unrecognized benchmark filename: {filename}")


def read_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No YOLO score rows in {path}")
    for row in rows:
        row["benchmark_group"] = classify_video(Path(row["video_file"]).name)
    return rows


def validate_matched_groups(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["benchmark_group"]].append(row)
    missing = set(GROUPS) - set(grouped)
    if missing:
        raise ValueError(f"Missing benchmark groups: {sorted(missing)}")
    expected = {row["video_id"] for row in grouped["wan_pca512_exact"]}
    if len(expected) != len(grouped["wan_pca512_exact"]):
        raise ValueError("Duplicate video_id values in Wan exact scores")
    for key in GROUPS:
        ids = {row["video_id"] for row in grouped[key]}
        if ids != expected or len(ids) != len(grouped[key]):
            raise ValueError(
                f"Group {key} is not matched; missing={sorted(expected - ids)}, extra={sorted(ids - expected)}"
            )
    return sorted(expected)


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> None:
    args = parse_args()
    rows = read_scores(args.scores)
    video_ids = validate_matched_groups(rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["benchmark_group"]].append(row)

    summary = []
    for key, label in GROUPS.items():
        values = [float(row["yolo_entity_score"]) for row in grouped[key]]
        summary.append(
            {
                "group": key,
                "model_condition": label,
                "videos": len(values),
                "mean_yolo_score": statistics.fmean(values),
                "std_yolo_score": statistics.pstdev(values),
                "mean_entity_coverage": mean(grouped[key], "entity_coverage"),
                "mean_entity_presence": mean(grouped[key], "mean_entity_presence"),
                "mean_full_entity_frame_rate": mean(grouped[key], "full_entity_frame_rate"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "model_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    detail_path = args.output_dir / "matched_video_scores.csv"
    detail_fields = ["video_id", *GROUPS]
    lookup = {(row["video_id"], row["benchmark_group"]): row for row in rows}
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        for video_id in video_ids:
            writer.writerow(
                {
                    "video_id": video_id,
                    **{
                        key: float(lookup[(video_id, key)]["yolo_entity_score"])
                        for key in GROUPS
                    },
                }
            )

    report = [
        "# Four-Model Real-EEG Video Benchmark",
        "",
        f"- Matched held-out videos: {len(video_ids)}",
        "- Subject: `chentianlin`",
        "- EEG: `session3`, 4 seconds",
        "- Seed: `0`",
        "- YOLO evaluates detectable entity presence, not action or relation correctness.",
        "",
        "| model / condition | n | YOLO score | std | coverage | presence | full-frame rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        report.append(
            f"| {row['model_condition']} | {row['videos']} | "
            f"{row['mean_yolo_score']:.3f} | {row['std_yolo_score']:.3f} | "
            f"{row['mean_entity_coverage']:.3f} | {row['mean_entity_presence']:.3f} | "
            f"{row['mean_full_entity_frame_rate']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Per-Video YOLO Score",
            "",
            "| video_id | " + " | ".join(GROUPS.values()) + " |",
            "| --- | " + " | ".join("---:" for _ in GROUPS) + " |",
        ]
    )
    for video_id in video_ids:
        report.append(
            f"| {video_id} | "
            + " | ".join(
                f"{float(lookup[(video_id, key)]['yolo_entity_score']):.3f}" for key in GROUPS
            )
            + " |"
        )
    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[four-model-report] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
