"""Collect semantic and video metrics into one paired long-form table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_schema import CORE_ENTITIES_BY_CATEGORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--video-metrics", type=Path, nargs="*", default=())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def semantic_rows(job: dict[str, Any], allow_missing: bool) -> list[dict[str, Any]]:
    output = Path(job["output_dir"])
    common = {
        "variant": job["variant"], "generator": "semantic_decoder",
        "subject": job["subject"], "fold": job["fold"], "seed": job["seed"],
        "generation_seed": -1,
    }
    if job["method"] in {"coarse_template", "structured_semantic", "temporal_category"}:
        path = output / "test_semantic/predictions.json"
        if not path.is_file():
            if allow_missing:
                return []
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("video_metric_records")
        if records is None:
            records = []
            for value in payload["video_aggregation_records"]:
                truth = value["video_id"].split("-", 1)[0]
                prediction = str(value["predicted_category"]).zfill(2)
                predicted_objects = set(value.get("predicted_objects", ()))
                records.append({
                    "video_id": value["video_id"],
                    "category_correct": float(prediction == truth),
                    "object_exact": float(
                        predicted_objects == set(CORE_ENTITIES_BY_CATEGORY[truth])
                    ),
                    "caption_nonempty": float(bool(str(value.get("caption", "")).strip())),
                })
    else:
        path = output / "test_conditions/video_index.jsonl"
        if not path.is_file():
            if allow_missing:
                return []
            raise FileNotFoundError(path)
        records = read_jsonl(path)
    rows = []
    ignored = {"video_id", "condition_path", "trial_count"}
    for record in records:
        for metric, value in record.items():
            if metric in ignored or value is None or not isinstance(value, (int, float)):
                continue
            rows.append({**common, "video_id": record["video_id"], "metric": metric, "value": value})
    return rows


def main() -> None:
    args = parse_args()
    jobs = read_jsonl(args.jobs)
    rows = [row for job in jobs for row in semantic_rows(job, args.allow_missing)]
    required_video = {"variant", "generator", "subject", "fold", "seed", "generation_seed", "video_id", "metric", "value"}
    for path in args.video_metrics:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            values = list(csv.DictReader(handle))
        if values and not required_video.issubset(values[0]):
            raise ValueError(f"Video metric file lacks long-form fields: {path}")
        rows.extend(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "generator", "subject", "fold", "seed", "generation_seed", "video_id", "metric", "value"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eeg-ablation-collect] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
