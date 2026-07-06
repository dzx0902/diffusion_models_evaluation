"""Evaluate extracted frames with detection and benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.detector import FrameDetector, load_detector_settings
from ms_video_eval.metrics import aggregate_metrics, build_failure_summary, evaluate_video
from ms_video_eval.task_schema import load_tasks
from ms_video_eval.utils import ensure_dir, get_repo_root, safe_int, write_csv
from ms_video_eval.video_io import parse_generation_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark frames.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument(
        "--frames",
        type=Path,
        default=get_repo_root() / "outputs" / "ms_eval" / "frames",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=get_repo_root() / "outputs" / "ms_eval" / "detections",
    )
    parser.add_argument("--settings", type=Path, default=ROOT / "configs" / "ms_eval_settings.yaml")
    parser.add_argument("--output", type=Path, default=get_repo_root() / "outputs" / "ms_eval" / "metrics")
    return parser.parse_args()


def _collect_frame_dirs(frames_root: Path) -> list[Path]:
    frame_dirs = set()
    for frame_path in frames_root.rglob("frame_*.jpg"):
        frame_dirs.add(frame_path.parent)
    for frame_path in frames_root.rglob("frame_*.png"):
        frame_dirs.add(frame_path.parent)
    return sorted(frame_dirs)


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.tasks)
    task_map = {task.id: task for task in tasks}
    settings = load_detector_settings(args.settings)
    detector = FrameDetector(settings)
    ensure_dir(args.output)
    ensure_dir(args.detections)

    video_rows = []
    frame_dirs = _collect_frame_dirs(args.frames)
    for frames_dir in frame_dirs:
        try:
            model_id = frames_dir.parent.name
            stem = frames_dir.name
            if "_seed" not in stem:
                continue
            task_id = stem[: stem.rfind("_seed")]
            seed = safe_int(stem.split("_seed")[-1])
            task = task_map.get(task_id)
            if task is None:
                continue
            detection_dir = args.detections / model_id
            ensure_dir(detection_dir)
            detection_path = detection_dir / f"{task_id}_seed{seed}.json"
            if not detection_path.exists():
                detector.detect_frames(frames_dir, detection_path)
            video_path = args.frames.parent / "videos" / model_id / f"{task_id}_seed{seed}.mp4"
            row = evaluate_video(task, detection_path, video_path if video_path.exists() else None)
            row["model_id"] = model_id
            row["task_id"] = task_id
            row["seed"] = seed
            video_rows.append(row)
        except Exception as exc:
            print(f"[evaluate][error] {frames_dir}: {exc}")

    if video_rows:
        fieldnames = sorted({key for row in video_rows for key in row.keys()})
        write_csv(args.output / "video_metrics.csv", video_rows, fieldnames)
        model_rows = aggregate_metrics(video_rows, "model_id")
        task_rows = aggregate_metrics(video_rows, "task_id")
        failure_rows = build_failure_summary(video_rows)
        model_fields = ["model_id", "video_count", "spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"]
        task_fields = ["task_id", "video_count", "spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"]
        failure_fields = ["model_id", "failure_type", "count", "rate"]
        write_csv(args.output / "model_summary.csv", model_rows, model_fields)
        write_csv(args.output / "task_summary.csv", task_rows, task_fields)
        write_csv(args.output / "failure_summary.csv", failure_rows, failure_fields)
    else:
        print("[evaluate] no frame directories found")


if __name__ == "__main__":
    main()
