"""Use the benchmark YOLO detector to score EEG/Wan probe videos.

This is an object-presence diagnostic, not a semantic action evaluator.  It
compares detector-visible central entities from the structured caption with
sampled generated frames, so exact-target and EEG-conditioned videos can be
ranked without manually opening every file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.detector import FrameDetector, load_detector_settings
from ms_video_eval.utils import ensure_dir
from ms_video_eval.video_io import extract_frames


ENTITY_ALIASES = {
    "people": "person",
    "puck": "ball",
    "truck": "car",
    "van": "car",
    "vehicle": "car",
    "suv": "car",
    "hatchback": "car",
    "pickup": "car",
    "flowers": "flower",
    "blossom": "flower",
    "blossoms": "flower",
    "bloom": "flower",
    "blooms": "flower",
}
VIDEO_ID_PATTERN = re.compile(r"(?<!\d)(\d{2}-\d{3})(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated EEG/Wan probe videos with YOLO object detection.")
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Structured V2 video manifest.")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs" / "ms_eval_settings.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(record["video_id"]): record for record in records}


def canonical_entities(record: dict[str, Any], supported: set[str]) -> list[str]:
    source = record.get("caption_entities", [])
    entities = []
    for entity in source:
        canonical = ENTITY_ALIASES.get(str(entity).strip().lower(), str(entity).strip().lower())
        if canonical in supported and canonical not in entities:
            entities.append(canonical)
    if not entities:
        raise ValueError(f"No detector-supported entities for {record['video_id']}: {source!r}")
    return entities


def video_id_from_path(path: Path) -> str:
    match = VIDEO_ID_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not infer video_id like 01-001 from {path.name}")
    return match.group(1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = read_manifest(args.manifest)
    settings = load_detector_settings(args.settings)
    supported = set((settings.class_aliases or {}).keys())
    detector = FrameDetector(settings)
    frames_root = ensure_dir(args.output_dir / "frames")
    detections_root = ensure_dir(args.output_dir / "detections")
    rows = []

    for video_path in args.videos:
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        video_id = video_id_from_path(video_path)
        if video_id not in records:
            raise KeyError(f"No manifest entry for {video_id}")
        record = records[video_id]
        expected = canonical_entities(record, supported)
        label = video_path.stem
        frames_dir = frames_root / label
        detection_path = detections_root / f"{label}.json"
        if not (args.skip_existing and any(frames_dir.glob("frame_*.jpg"))):
            extract_frames(video_path, frames_dir, sample_every=args.sample_every)
        if args.skip_existing and detection_path.exists():
            payload = json.loads(detection_path.read_text(encoding="utf-8"))
        else:
            payload = detector.detect_frames(frames_dir, detection_path)

        frame_rows = payload["detections"]
        frame_count = len(frame_rows)
        presence = {entity: 0 for entity in expected}
        full_frames = 0
        detected_all: set[str] = set()
        for frame in frame_rows:
            found = {det["class_name"] for det in frame["detections"]}
            detected_all.update(found)
            for entity in expected:
                presence[entity] += int(entity in found)
            full_frames += int(all(entity in found for entity in expected))
        rates = {entity: count / max(frame_count, 1) for entity, count in presence.items()}
        coverage = sum(rate > 0 for rate in rates.values()) / len(expected)
        mean_presence = sum(rates.values()) / len(expected)
        full_rate = full_frames / max(frame_count, 1)
        score = 0.5 * coverage + 0.3 * mean_presence + 0.2 * full_rate
        rows.append(
            {
                "video_file": str(video_path),
                "video_id": video_id,
                "caption": record["caption"],
                "expected_entities": ";".join(expected),
                "detected_entities": ";".join(sorted(detected_all)),
                "sampled_frames": frame_count,
                "entity_coverage": coverage,
                "mean_entity_presence": mean_presence,
                "min_entity_presence": min(rates.values()),
                "full_entity_frame_rate": full_rate,
                "yolo_entity_score": score,
                "per_entity_frame_rate": json.dumps(rates, sort_keys=True),
            }
        )
        print(f"[eeg-wan-yolo] {video_path.name}: score={score:.3f} coverage={coverage:.3f}", flush=True)

    rows.sort(key=lambda row: (-float(row["yolo_entity_score"]), row["video_file"]))
    ensure_dir(args.output_dir)
    write_csv(args.output_dir / "video_scores.csv", rows)
    report = ["# EEG/Wan YOLO Probe Scores", "", "| video | id | expected | coverage | mean presence | full-frame rate | score |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        report.append(
            f"| {Path(row['video_file']).name} | {row['video_id']} | {row['expected_entities']} | "
            f"{float(row['entity_coverage']):.3f} | {float(row['mean_entity_presence']):.3f} | "
            f"{float(row['full_entity_frame_rate']):.3f} | {float(row['yolo_entity_score']):.3f} |"
        )
    report.extend(["", "YOLO measures detectable object presence only; it does not verify actions or relations."])
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[eeg-wan-yolo] wrote {args.output_dir / 'video_scores.csv'}")


if __name__ == "__main__":
    main()
