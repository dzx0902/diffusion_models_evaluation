"""Evaluate Wan PCA ablation videos with the existing YOLO benchmark.

The script copies or symlinks manually generated PCA videos into the benchmark
video layout, then runs frame extraction, YOLO evaluation, and report building.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wan PCA output videos.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("$MS_MODELS_ROOT/Wan2.2/outputs"),
        help="Directory containing pca_baseline_seed0.mp4 and pca_<dim>_seed0.mp4.",
    )
    parser.add_argument("--task-id", type=str, default="dog_car_walk_static")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dims", type=str, nargs="+", default=["baseline", "1536", "1024", "768", "512"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "wan_pca_eval",
        help="Evaluation root; videos/frames/detections/metrics are written below it.",
    )
    parser.add_argument("--settings", type=Path, default=ROOT / "configs" / "ms_eval_settings.wsl.yaml")
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy videos instead of symlinking. Use this if symlinks are inconvenient on mounted drives.",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only prepare the video layout.")
    return parser.parse_args()


def expand_path(path: Path) -> Path:
    text = str(path)
    for name, value in {
        "$MS_MODELS_ROOT": str(Path.home() / "workspace" / "diffusion_models_evaluation" / ".ms_video_models"),
        "$MS_BENCHMARK_ROOT": str(ROOT),
    }.items():
        text = text.replace(name, value)
    return Path(text).expanduser()


def source_name(dim: str, seed: int) -> str:
    if dim == "baseline":
        return f"pca_baseline_seed{seed}.mp4"
    return f"pca_{dim}_seed{seed}.mp4"


def model_id(dim: str) -> str:
    if dim == "baseline":
        return "wan_pca_baseline"
    return f"wan_pca_{dim}"


def link_or_copy(source: Path, target: Path, copy_file: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy_file:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def run(command: list[str]) -> None:
    print("[wan-pca-eval]", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    source_dir = expand_path(args.source_dir)
    output_root = args.output_root
    videos_root = output_root / "videos"
    frames_root = output_root / "frames"
    detections_root = output_root / "detections"
    metrics_root = output_root / "metrics"

    missing: list[Path] = []
    for dim in args.dims:
        source = source_dir / source_name(dim, args.seed)
        if not source.exists():
            missing.append(source)
            continue
        target = videos_root / model_id(dim) / f"{args.task_id}_seed{args.seed}.mp4"
        link_or_copy(source, target, args.copy)
        print(f"[wan-pca-eval] prepared {target}")

    if missing:
        raise FileNotFoundError("Missing source videos: " + ", ".join(str(path) for path in missing))
    if args.skip_run:
        return

    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ms_extract_frames.py"),
            "--videos",
            str(videos_root),
            "--output",
            str(frames_root),
            "--sample-every",
            str(args.sample_every),
            "--manifest",
            str(metrics_root / "frame_manifest.jsonl"),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ms_evaluate.py"),
            "--tasks",
            str(args.tasks),
            "--frames",
            str(frames_root),
            "--detections",
            str(detections_root),
            "--settings",
            str(args.settings),
            "--output",
            str(metrics_root),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ms_build_report.py"),
            "--metrics",
            str(metrics_root),
            "--tasks",
            str(args.tasks),
            "--output",
            str(output_root / "reports" / "wan_pca_eval_report.md"),
        ]
    )
    print(f"[wan-pca-eval] report: {output_root / 'reports' / 'wan_pca_eval_report.md'}")


if __name__ == "__main__":
    main()
