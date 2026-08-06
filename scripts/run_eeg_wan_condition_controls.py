"""Generate matched native, exact, correct-EEG, shuffled-EEG, and zero controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched EEG-to-Wan condition controls.")
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--session", default="session3")
    parser.add_argument("--shuffled-video-id", required=True)
    parser.add_argument("--shuffled-session", default=None)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--task", default="ti2v-5B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-model", choices=["True", "False"], default="True")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[eeg-wan-controls] " + " ".join(command), flush=True)
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
    if args.video_id not in records:
        raise KeyError(f"Target video absent from manifest: {args.video_id}")
    if args.shuffled_video_id not in records:
        raise KeyError(f"Shuffled EEG video absent from manifest: {args.shuffled_video_id}")
    if args.shuffled_video_id == args.video_id and (args.shuffled_session or args.session) == args.session:
        raise ValueError("Shuffled EEG control must use a different video or session")

    def validate_manifest_trial(video_id: str, session: str) -> None:
        matches = [item for item in records[video_id].get("sessions", []) if item["session"] == session]
        if len(matches) != 1:
            raise ValueError(f"Expected one manifest trial for {video_id}/{session}; found {len(matches)}")
        duration = float(matches[0]["duration_sec"])
        if abs(duration - args.duration_sec) > 1e-6:
            raise ValueError(
                f"Manifest trial {video_id}/{session} has duration_sec={duration}; "
                f"expected {args.duration_sec}"
            )

    validate_manifest_trial(args.video_id, args.session)
    validate_manifest_trial(args.shuffled_video_id, args.shuffled_session or args.session)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.wan_repo / "Wan2.2-TI2V-5B"
    adapter = ROOT / "scripts" / "adapters" / "wan_eeg_generate.py"

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

    native_output = args.output_dir / f"{args.video_id}_native_seed{args.seed}.mp4"
    if not (args.skip_existing and native_output.exists() and native_output.stat().st_size > 0):
        native_command = [
            sys.executable,
            str(args.wan_repo / "generate.py"),
            "--task", args.task,
            "--size", args.size,
            "--ckpt_dir", str(ckpt_dir),
            "--offload_model", args.offload_model,
            "--convert_model_dtype",
            "--t5_cpu",
            "--base_seed", str(args.seed),
            "--prompt", str(records[args.video_id]["caption"]),
            "--save_file", str(native_output),
        ]
        run(native_command)

    for label, condition_source, eeg_video_id, eeg_session in variants:
        output = args.output_dir / f"{args.video_id}_{label}_seed{args.seed}.mp4"
        condition_output = args.output_dir / f"{args.video_id}_{label}_seed{args.seed}.pt"
        if args.skip_existing and output.exists() and output.stat().st_size > 0:
            print(f"[eeg-wan-controls] skip existing {output}", flush=True)
            continue
        command = [
            sys.executable,
            str(adapter),
            "--wan-repo", str(args.wan_repo),
            "--checkpoint", str(args.checkpoint),
            "--trials", str(args.trials),
            "--targets", str(args.targets),
            "--projector", str(args.projector),
            "--video-id", args.video_id,
            "--session", args.session,
            "--eeg-video-id", eeg_video_id,
            "--eeg-session", eeg_session,
            "--condition-source", condition_source,
            "--length-source", "target",
            "--expected-duration-sec", str(args.duration_sec),
            "--condition-output", str(condition_output),
        ]
        if args.enable_tf32:
            command.append("--enable-tf32")
        command.extend(
            [
                "--",
                "--task", args.task,
                "--size", args.size,
                "--ckpt_dir", str(ckpt_dir),
                "--offload_model", args.offload_model,
                "--convert_model_dtype",
                "--t5_cpu",
                "--base_seed", str(args.seed),
                "--prompt", "EEG-conditioned video.",
                "--save_file", str(output),
            ]
        )
        run(command)
    print(f"[eeg-wan-controls] outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
