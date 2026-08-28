"""Extract fixed multi-entity Tora trajectories from held-out GT videos using YOLO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.detector import FrameDetector, load_detector_settings
from ms_video_eval.semantic_data import load_video_partitions
from ms_video_eval.semantic_schema import load_semantic_records, normalize_video_id
from ms_video_eval.trajectory_extraction import interpolate_centers, tora_canvas_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--fold", default="video_6fold_1")
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/ms_eval_settings.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=49)
    parser.add_argument("--min-detection-coverage", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_video_manifest(path: Path) -> dict[str, Path]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        normalize_video_id(row["video_id"]): (
            Path(row["video_path"]) if Path(row["video_path"]).is_absolute()
            else ROOT / row["video_path"]
        )
        for row in rows
    }


def extract_uniform_frames(video: Path, output: Path, count: int) -> list[Path]:
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        capture.release(); raise ValueError(f"Video has fewer than two frames: {video}")
    indices = np.rint(np.linspace(0, frame_count - 1, count)).astype(int)
    output.mkdir(parents=True, exist_ok=True); paths = []
    for output_index, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            capture.release(); raise RuntimeError(f"Failed reading frame {frame_index} from {video}")
        path = output / f"frame_{output_index:04d}.jpg"
        if not cv2.imwrite(str(path), frame):
            capture.release(); raise RuntimeError(f"Failed writing {path}")
        paths.append(path)
    capture.release(); return paths


def detection_centers(payload: dict[str, Any], entity: str) -> list[tuple[float, float] | None]:
    result = []
    for frame in payload["detections"]:
        image = cv2.imread(str(frame["frame_path"]))
        height, width = image.shape[:2]
        matches = [item for item in frame["detections"] if item["class_name"] == entity]
        if not matches:
            result.append(None); continue
        item = max(matches, key=lambda value: float(value["confidence"]))
        x1, y1, x2, y2 = map(float, item["bbox"])
        result.append(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)))
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.num_points < 2 or not 0 <= args.min_detection_coverage <= 1:
        raise ValueError("Invalid num-points or coverage threshold")
    videos = read_video_manifest(args.video_manifest)
    semantics = {item.video_id: item for item in load_semantic_records(args.semantic_labels)}
    ids = sorted(load_video_partitions(args.split_plan, args.fold)[args.partition])
    missing = set(ids) - set(videos) | (set(ids) - set(semantics))
    if missing:
        raise KeyError(f"Missing videos/semantics: {sorted(missing)[:5]}")
    detector = FrameDetector(load_detector_settings(args.settings))
    report_rows = []; manifest_rows = []
    for count, video_id in enumerate(ids, 1):
        frame_dir = args.output_dir / "frames" / video_id
        detection_path = args.output_dir / "detections" / f"{video_id}.json"
        detection_path.parent.mkdir(parents=True, exist_ok=True)
        if not (args.skip_existing and detection_path.is_file()):
            extract_uniform_frames(videos[video_id], frame_dir, args.num_points)
            payload = detector.detect_frames(frame_dir, detection_path)
        else:
            payload = json.loads(detection_path.read_text(encoding="utf-8"))
        trajectory_paths = []; coverages = {}
        for entity_index, entity in enumerate(semantics[video_id].core_entities):
            normalized, coverage = interpolate_centers(detection_centers(payload, entity))
            points = tora_canvas_points(normalized)
            path = args.output_dir / "tracks" / video_id / f"{entity_index:02d}_{entity}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(f"{x},{y}\n" for x, y in points), encoding="utf-8")
            trajectory_paths.append(str(path.resolve())); coverages[entity] = coverage
            report_rows.append({"video_id": video_id, "entity": entity, "coverage": coverage,
                                "fallback_only": int(coverage == 0), "trajectory_path": str(path.resolve())})
        if min(coverages.values(), default=0.0) < args.min_detection_coverage:
            raise RuntimeError(f"{video_id} trajectory coverage below threshold: {coverages}")
        manifest_rows.append({"schema_version": 1, "video_id": video_id,
                              "trajectory_paths": trajectory_paths,
                              "sha256s": [sha256(Path(path)) for path in trajectory_paths],
                              "entities": list(semantics[video_id].core_entities),
                              "detection_coverage": coverages, "source_video": str(videos[video_id].resolve()),
                              "source": "gt_yolo_center_interpolated", "fold": args.fold,
                              "partition": args.partition, "canvas": [256, 256],
                              "num_points": args.num_points})
        print(f"[fixed-tora-tracks] {count}/{len(ids)} {video_id} {coverages}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
    with (args.output_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0])); writer.writeheader(); writer.writerows(report_rows)
    print(f"[fixed-tora-tracks] manifest={manifest}")


if __name__ == "__main__":
    main()
