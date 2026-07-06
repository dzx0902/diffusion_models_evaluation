"""Extract frames from generated videos for benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.utils import ensure_dir, get_repo_root, timestamp_iso, write_jsonl
from ms_video_eval.video_io import extract_frames, parse_generation_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from benchmark videos.")
    parser.add_argument("--videos", type=Path, default=get_repo_root() / "outputs" / "ms_eval" / "videos")
    parser.add_argument("--output", type=Path, default=get_repo_root() / "outputs" / "ms_eval" / "frames")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=get_repo_root() / "outputs" / "ms_eval" / "metrics" / "frame_manifest.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)
    ensure_dir(args.manifest.parent)

    manifest_rows = []
    video_files = sorted(args.videos.rglob("*.mp4"))
    for video_path in video_files:
        try:
            model_id, task_id, seed = parse_generation_filename(video_path)
            frames_dir = args.output / model_id / f"{task_id}_seed{seed}"
            extracted = extract_frames(video_path, frames_dir, sample_every=args.sample_every)
            manifest_rows.append(
                {
                    "timestamp": timestamp_iso(),
                    "video_path": str(video_path),
                    "model_id": model_id,
                    "task_id": task_id,
                    "seed": seed,
                    "frames_dir": str(frames_dir),
                    "sample_every": args.sample_every,
                    "frame_count": len(extracted),
                    "status": "success",
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "timestamp": timestamp_iso(),
                    "video_path": str(video_path),
                    "model_id": "",
                    "task_id": "",
                    "seed": "",
                    "frames_dir": "",
                    "sample_every": args.sample_every,
                    "frame_count": 0,
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            print(f"[extract][error] {video_path}: {exc}")

    write_jsonl(args.manifest, manifest_rows)


if __name__ == "__main__":
    main()

