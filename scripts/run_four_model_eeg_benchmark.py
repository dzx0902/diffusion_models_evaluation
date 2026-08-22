"""Generate a matched four-model benchmark from held-out Chentianlin EEG."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("wan", "animatediff", "zeroscope", "cogvideox2b"),
        default=("wan", "animatediff", "zeroscope", "cogvideox2b"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "eeg_generation_common" / "chentianlin" / "video_6fold_1",
    )
    parser.add_argument("--wan-python", type=Path, default=Path.home() / "miniconda3/envs/wan22/bin/python")
    parser.add_argument("--clip-python", type=Path, default=Path.home() / "miniconda3/envs/clip-video/bin/python")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_suite(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty benchmark suite: {path}")
    video_ids = [str(row["video_id"]) for row in rows]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("Benchmark suite contains duplicate video_id values")
    for row in rows:
        if abs(float(row["duration_sec"]) - 4.0) > 1e-6:
            raise ValueError(f"Expected a 4-second trial, got: {row}")
    return rows


def require(paths: list[Path], dry_run: bool) -> None:
    if dry_run:
        return
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(map(str, missing)))


def run(command: list[str], dry_run: bool) -> None:
    print("[four-model-eeg] " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def generate_atomic(command: list[str], output: Path, skip_existing: bool, dry_run: bool) -> None:
    if skip_existing and output.exists() and output.stat().st_size > 0:
        print(f"[four-model-eeg] skip existing {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
    if temporary.exists() and not dry_run:
        temporary.unlink()
    run([*command, "--output", str(temporary)], dry_run)
    if not dry_run:
        os.replace(temporary, output)


def predict_condition(
    python: Path,
    checkpoint: Path,
    targets: Path,
    trials: Path,
    video_id: str,
    session: str,
    output: Path,
    dry_run: bool,
) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(python),
            str(ROOT / "scripts/predict_eeg_clip_condition.py"),
            "--checkpoint", str(checkpoint),
            "--trials", str(trials),
            "--targets", str(targets),
            "--video-id", video_id,
            "--session", session,
            "--condition-source", "eeg",
            "--expected-duration-sec", "4",
            "--output", str(output),
        ],
        dry_run,
    )


def main() -> None:
    args = parse_args()
    suite = read_suite(args.suite)
    trials = ROOT / "data/manifests/chentianlin/eeg_trials.csv"
    videos_root = args.output_dir / "videos"
    conditions_root = args.output_dir / "conditions"

    wan_repo = args.models_root / "Wan2.2"
    wan_checkpoint = ROOT / "outputs/eeg_wan_structured_v2/chentianlin/video_6fold_1/best.pt"
    wan_targets = ROOT / "outputs/eeg_wan_structured_v2/video_6fold_1/wan_targets.jsonl"
    wan_projector = ROOT / (
        "outputs/eeg_wan_structured_v2/video_6fold_1/"
        "pca_decoder_validation/text_space/token_pca_projector.npz"
    )

    clip_models = {
        "animatediff": {
            "checkpoint": ROOT / "outputs/eeg_clip_video/animatediff/chentianlin/video_6fold_1/best.pt",
            "targets": ROOT / "outputs/eeg_clip_video/animatediff/condition_targets_pipeline_bf16.jsonl",
            "model_root": args.models_root / "AnimateDiff/sd-v1-5",
            "motion": args.models_root / "AnimateDiff/motion-adapter-v1-5-2",
            "dtype": "bfloat16",
            "frames": "32",
        },
        "zeroscope": {
            "checkpoint": ROOT / "outputs/eeg_clip_video/zeroscope/chentianlin/video_6fold_1/best.pt",
            "targets": ROOT / "outputs/eeg_clip_video/zeroscope/condition_targets_pipeline_fp16.jsonl",
            "model_root": args.models_root / "ZeroScope/zeroscope_v2_576w",
            "motion": None,
            "dtype": "float16",
            "frames": "32",
        },
        "cogvideox2b": {
            "checkpoint": ROOT / "outputs/eeg_clip_video/cogvideox2b/chentianlin/video_6fold_1/best.pt",
            "targets": ROOT / "outputs/eeg_clip_video/cogvideox2b/condition_targets_pipeline_fp16.jsonl",
            "model_root": args.models_root / "CogVideoX-2b",
        },
    }

    required = [args.suite, trials, args.wan_python, args.clip_python]
    if "wan" in args.models:
        required.extend([wan_repo / "generate.py", wan_checkpoint, wan_targets, wan_projector])
    for model_name in set(args.models) & set(clip_models):
        spec = clip_models[model_name]
        required.extend([spec["checkpoint"], spec["targets"], spec["model_root"]])
        if spec.get("motion") is not None:
            required.append(spec["motion"])
    require([Path(path) for path in required], args.dry_run)

    for row in suite:
        video_id = str(row["video_id"])
        session = str(row["session"])
        if "wan" in args.models:
            for label, source in (("exact", "target"), ("eeg", "eeg")):
                output = videos_root / "wan_pca512" / f"{video_id}_wan_pca512_{label}_seed{args.seed}.mp4"
                command = [
                    str(args.wan_python),
                    str(ROOT / "scripts/adapters/wan_eeg_generate.py"),
                    "--wan-repo", str(wan_repo),
                    "--checkpoint", str(wan_checkpoint),
                    "--trials", str(trials),
                    "--targets", str(wan_targets),
                    "--projector", str(wan_projector),
                    "--video-id", video_id,
                    "--session", session,
                    "--condition-source", source,
                    "--length-source", "target",
                    "--expected-duration-sec", "4",
                    "--condition-output", str(conditions_root / "wan_pca512" / f"{video_id}_{label}.pt"),
                    "--enable-tf32",
                    "--",
                    "--task", "ti2v-5B",
                    "--size", "1280*704",
                    "--frame_num", "65",
                    "--ckpt_dir", str(wan_repo / "Wan2.2-TI2V-5B"),
                    "--offload_model", "True",
                    "--convert_model_dtype",
                    "--t5_cpu",
                    "--base_seed", str(args.seed),
                    "--prompt", "EEG-conditioned video.",
                    "--save_file", str(output.with_name(f"{output.stem}.partial{output.suffix}")),
                ]
                if args.skip_existing and output.exists() and output.stat().st_size > 0:
                    print(f"[four-model-eeg] skip existing {output}", flush=True)
                else:
                    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if temporary.exists() and not args.dry_run:
                        temporary.unlink()
                    run(command, args.dry_run)
                    if not args.dry_run:
                        os.replace(temporary, output)

        for model_name in ("animatediff", "zeroscope", "cogvideox2b"):
            if model_name not in args.models:
                continue
            spec = clip_models[model_name]
            condition = conditions_root / model_name / f"{video_id}_eeg.pt"
            predict_condition(
                args.clip_python,
                Path(spec["checkpoint"]),
                Path(spec["targets"]),
                trials,
                video_id,
                session,
                condition,
                args.dry_run,
            )
            output = videos_root / model_name / f"{video_id}_{model_name}_eeg_seed{args.seed}.mp4"
            if model_name == "cogvideox2b":
                command = [
                    str(args.clip_python),
                    str(ROOT / "scripts/adapters/cogvideox_condition_generate.py"),
                    "--model-root", str(spec["model_root"]),
                    "--condition", str(condition),
                    "--dtype", "float16",
                    "--height", "480",
                    "--width", "720",
                    "--num-frames", "33",
                    "--fps", "8",
                    "--steps", "50",
                    "--guidance-scale", "6",
                    "--seed", str(args.seed),
                    "--enable-tf32",
                ]
            else:
                command = [
                    str(args.clip_python),
                    str(ROOT / "scripts/adapters/clip_video_generate.py"),
                    "--backend", model_name,
                    "--model-root", str(spec["model_root"]),
                    "--condition", str(condition),
                    "--dtype", str(spec["dtype"]),
                    "--num-frames", str(spec["frames"]),
                    "--fps", "8",
                    "--steps", "25",
                    "--guidance-scale", "7.5",
                    "--seed", str(args.seed),
                    "--enable-tf32",
                ]
                if model_name == "animatediff":
                    command.extend(["--motion-adapter", str(spec["motion"]), "--height", "512", "--width", "512"])
                else:
                    command.extend(["--height", "320", "--width", "576"])
            generate_atomic(command, output, args.skip_existing, args.dry_run)

    print(f"[four-model-eeg] outputs: {videos_root}", flush=True)


if __name__ == "__main__":
    main()
