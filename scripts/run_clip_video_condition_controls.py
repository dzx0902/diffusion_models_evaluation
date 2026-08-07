"""Run matched native/exact/correct/shuffled/zero CLIP-video controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched CLIP video condition controls.")
    parser.add_argument("--backend", choices=["animatediff", "zeroscope"], required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--motion-adapter", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--session", default="session3")
    parser.add_argument("--shuffled-video-id", required=True)
    parser.add_argument("--shuffled-session", default=None)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[clip-controls] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    records = {
        str(row["video_id"]): row
        for row in (
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    for video_id in (args.video_id, args.shuffled_video_id):
        if video_id not in records:
            raise KeyError(f"Video absent from manifest: {video_id}")
    if args.video_id == args.shuffled_video_id and (args.shuffled_session or args.session) == args.session:
        raise ValueError("Shuffled control must use a different EEG video or session")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = ROOT / "scripts" / "predict_eeg_clip_condition.py"
    generator = ROOT / "scripts" / "adapters" / "clip_video_generate.py"
    common_generate = [
        sys.executable,
        str(generator),
        "--backend", args.backend,
        "--model-root", str(args.model_root),
        "--num-frames", str(args.num_frames),
        "--fps", str(args.fps),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
    ]
    if args.motion_adapter is not None:
        common_generate.extend(["--motion-adapter", str(args.motion_adapter)])
    if args.cpu_offload:
        common_generate.append("--cpu-offload")
    if args.enable_tf32:
        common_generate.append("--enable-tf32")

    native_output = args.output_dir / f"{args.video_id}_native_seed{args.seed}.mp4"
    if not (args.skip_existing and native_output.exists() and native_output.stat().st_size > 0):
        run(common_generate + [
            "--prompt", str(records[args.video_id]["caption"]),
            "--output", str(native_output),
        ])

    variants = [
        ("exact_target", "target", args.video_id, args.session),
        ("correct_eeg", "eeg", args.video_id, args.session),
        (
            "shuffled_eeg",
            "eeg",
            args.shuffled_video_id,
            args.shuffled_session or args.session,
        ),
        ("zero", "zero", args.video_id, args.session),
    ]
    for label, source, eeg_video_id, eeg_session in variants:
        condition = args.output_dir / f"{args.video_id}_{label}_seed{args.seed}.pt"
        output = args.output_dir / f"{args.video_id}_{label}_seed{args.seed}.mp4"
        if args.skip_existing and output.exists() and output.stat().st_size > 0:
            print(f"[clip-controls] skip existing {output}", flush=True)
            continue
        if not condition.exists():
            run([
                sys.executable,
                str(predictor),
                "--checkpoint", str(args.checkpoint),
                "--trials", str(args.trials),
                "--targets", str(args.targets),
                "--video-id", args.video_id,
                "--session", args.session,
                "--eeg-video-id", eeg_video_id,
                "--eeg-session", eeg_session,
                "--condition-source", source,
                "--expected-duration-sec", str(args.duration_sec),
                "--output", str(condition),
            ])
        run(common_generate + ["--condition", str(condition), "--output", str(output)])
    print(f"[clip-controls] outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
