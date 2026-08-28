"""Aggregate long-form EEG ablation metrics and run matched paired tests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_statistics import aggregate_long_metrics, paired_comparisons


REQUIRED = {"variant", "generator", "subject", "fold", "seed", "generation_seed", "video_id", "metric", "value"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            values = list(csv.DictReader(handle))
        if not values or not REQUIRED.issubset(values[0]):
            raise ValueError(f"{path} must contain {sorted(REQUIRED)}")
        rows.extend(values)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.metrics)
    summary = aggregate_long_metrics(rows)
    comparisons = paired_comparisons(rows, args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_tests.csv", comparisons)
    report = [
        "# EEG Semantic Ablation Report", "",
        f"Baseline: `{args.baseline}`", "",
        "| variant | generator | metric | n | mean ± std |", "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary:
        report.append(
            f"| {row['variant']} | {row['generator']} | {row['metric']} | {row['n']} | "
            f"{row['mean']:.6f} ± {row['std']:.6f} |"
        )
    report.extend(["", "Paired tests use matched subject/fold/seed/video observations and a two-sided sign-flip randomization test."])
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[eeg-ablation-report] rows={len(rows)} comparisons={len(comparisons)}")


if __name__ == "__main__":
    main()
