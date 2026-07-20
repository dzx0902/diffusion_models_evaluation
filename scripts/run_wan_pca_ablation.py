"""Generate and evaluate Wan text-compression ablations.

Variants:
  baseline      no compression
  d768          feature PCA 4096 -> 768 -> 4096
  t64           token resample N -> 64 -> N
  d768_t64      feature PCA plus token resampling
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.prompt_builder import build_prompt
from ms_video_eval.task_schema import load_tasks


EXPERIMENT_PRESETS = {
    # A screening set that covers static layout, two independent motions, and human/object grounding.
    "pilot": {
        "task_ids": [
            "dog_car_walk_static",
            "dog_ball_walk_roll",
            "person_bicycle_walk_static",
        ],
        "seeds": [0, 1],
        "variants": [
            "baseline",
            "d512",
            "d256",
            "t64",
            "t48",
            "t32",
            "d512_t48",
        ],
    },
    "full": {
        "task_ids": [
            "dog_car_walk_static",
            "ball_car_roll_static",
            "dog_ball_walk_roll",
            "person_bicycle_walk_static",
            "car_flower_static_sway",
            "bird_flower_land_near",
        ],
        "seeds": [0, 1, 2],
        "variants": [
            "baseline",
            "d1536",
            "d1024",
            "d768",
            "d512",
            "d256",
            "t64",
            "t48",
            "t32",
            "d768_t64",
            "d512_t48",
            "d256_t32",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-task, multi-seed Wan text-compression ablation benchmark."
    )
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument(
        "--preset",
        choices=sorted(EXPERIMENT_PRESETS),
        default="pilot",
        help="Experiment matrix. Defaults to the 48-video screening preset.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--projector",
        type=Path,
        default=ROOT / "outputs" / "text_space" / "wan2_2_ti2v_5b" / "token_pca_projector.npz",
    )
    parser.add_argument(
        "--wan-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable for Wan generation, normally the wan22 environment.",
    )
    parser.add_argument(
        "--eval-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable for frame extraction, YOLO evaluation, and reporting.",
    )
    parser.add_argument("--wan-repo", type=Path, default=Path("$MS_MODELS_ROOT/Wan2.2"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("$MS_MODELS_ROOT/Wan2.2/Wan2.2-TI2V-5B"))
    parser.add_argument("--size", type=str, default="1280*704")
    parser.add_argument(
        "--frame-num",
        type=int,
        default=65,
        help="Wan frame count (must be 4n+1). 65 matches the benchmark's 4 seconds at 16 fps.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Override Wan's default denoising steps. Leave unset for the model default.",
    )
    parser.add_argument(
        "--gpu-resident-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep T5 and both Wan DiT submodels in GPU memory; enabled by default for 48GB GPUs.",
    )
    parser.add_argument(
        "--enable-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 acceleration for remaining FP32 CUDA operations.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "wan_pca_ablation")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs" / "ms_eval_settings.wsl.yaml")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--report-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned task/seed/variant matrix and exit without generating videos.",
    )
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    preset = EXPERIMENT_PRESETS[args.preset]
    if args.task_ids is None:
        args.task_ids = preset["task_ids"]
    if args.seeds is None:
        args.seeds = preset["seeds"]
    if args.variants is None:
        args.variants = preset["variants"]


def expand_path(path: Path) -> Path:
    text = str(path)
    models_root = ROOT / ".ms_video_models"
    replacements = {
        "$MS_BENCHMARK_ROOT": str(ROOT),
        "$MS_MODELS_ROOT": str(models_root),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return Path(text).expanduser()


def validate_python_executable(label: str, path: Path) -> None:
    """Reject an unset shell variable before subprocess reports a cryptic error."""

    if path.resolve() == Path.cwd().resolve():
        raise ValueError(
            f"--{label} resolved to the current directory ({path}). "
            "This usually means an empty shell variable was passed. "
            "Pass an absolute Python path, for example "
            "--eval-python /home/dzxy/miniconda3/envs/ms-video-eval/bin/python."
        )
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"--{label} must be an executable Python file, got: {path}")


def parse_variant(variant: str) -> tuple[int, int]:
    if variant == "baseline":
        return 0, 0
    dim = 0
    token_count = 0
    for part in variant.split("_"):
        if match := re.fullmatch(r"d(\d+)", part):
            dim = int(match.group(1))
        elif match := re.fullmatch(r"t(\d+)", part):
            token_count = int(match.group(1))
        else:
            raise ValueError(f"Unsupported variant part: {part!r} in {variant!r}")
    return dim, token_count


def model_id(variant: str) -> str:
    return f"wan_ablate_{variant}"


def run(command: list[str]) -> None:
    print("[wan-ablation]", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def selected_tasks(args: argparse.Namespace):
    by_id = {task.id: task for task in load_tasks(args.tasks)}
    missing = [task_id for task_id in args.task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(missing)}")
    return [by_id[task_id] for task_id in args.task_ids]


def write_experiment_manifest(args: argparse.Namespace, tasks) -> None:  # type: ignore[no-untyped-def]
    records = []
    for task in tasks:
        prompt = build_prompt(task)
        for seed in args.seeds:
            for variant in args.variants:
                records.append(
                    {
                        "task_id": task.id,
                        "seed": seed,
                        "variant": variant,
                        "preset": args.preset,
                        "model_id": model_id(variant),
                        "prompt": prompt,
                    }
                )
    path = args.output_root / "experiment_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[wan-ablation] planned videos={len(records)} manifest={path}", flush=True)


def generate(args: argparse.Namespace) -> None:
    tasks = selected_tasks(args)
    write_experiment_manifest(args, tasks)
    if args.dry_run:
        return
    videos_root = args.output_root / "videos"
    wan_repo = expand_path(args.wan_repo)
    ckpt_dir = expand_path(args.ckpt_dir)
    projector = expand_path(args.projector)

    for task in tasks:
        prompt = build_prompt(task)
        for seed in args.seeds:
            for variant in args.variants:
                dim, token_count = parse_variant(variant)
                output_path = videos_root / model_id(variant) / f"{task.id}_seed{seed}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if args.skip_existing and output_path.exists():
                    print(f"[wan-ablation] skip existing {output_path}")
                    continue

                wrapper_args = [
                    str(args.wan_python),
                    str(ROOT / "scripts" / "adapters" / "wan_projected_generate.py"),
                    "--wan-repo",
                    str(wan_repo),
                ]
                if args.gpu_resident_models:
                    wrapper_args.append("--gpu-resident-models")
                if args.enable_tf32:
                    wrapper_args.append("--enable-tf32")
                if dim > 0:
                    wrapper_args.extend(["--projector", str(projector), "--project-dim", str(dim)])
                else:
                    wrapper_args.append("--disable-projection")
                if token_count > 0:
                    wrapper_args.extend(["--token-count", str(token_count)])
                if args.report_error:
                    wrapper_args.append("--report-error")

                generation_args = [
                    *wrapper_args,
                    "--",
                    "--task",
                    "ti2v-5B",
                    "--size",
                    args.size,
                    "--frame_num",
                    str(args.frame_num),
                    "--ckpt_dir",
                    str(ckpt_dir),
                    "--convert_model_dtype",
                    "--base_seed",
                    str(seed),
                    "--prompt",
                    prompt,
                    "--save_file",
                    str(output_path),
                ]
                if args.sample_steps is not None:
                    generation_args.extend(["--sample_steps", str(args.sample_steps)])
                if args.gpu_resident_models:
                    command = [*generation_args, "--offload_model", "False"]
                else:
                    command = [*generation_args, "--offload_model", "True", "--t5_cpu"]
                run(command)


def evaluate(args: argparse.Namespace) -> None:
    frames_root = args.output_root / "frames"
    detections_root = args.output_root / "detections"
    metrics_root = args.output_root / "metrics"
    run(
        [
            str(args.eval_python),
            str(ROOT / "scripts" / "ms_extract_frames.py"),
            "--videos",
            str(args.output_root / "videos"),
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
            str(args.eval_python),
            str(ROOT / "scripts" / "ms_evaluate.py"),
            "--tasks",
            str(args.tasks),
            "--frames",
            str(frames_root),
            "--detections",
            str(detections_root),
            "--settings",
            str(expand_path(args.settings)),
            "--output",
            str(metrics_root),
        ]
    )
    run(
        [
            str(args.eval_python),
            str(ROOT / "scripts" / "ms_build_report.py"),
            "--metrics",
            str(metrics_root),
            "--tasks",
            str(args.tasks),
            "--output",
            str(args.output_root / "reports" / "wan_pca_ablation_report.md"),
        ]
    )


def main() -> None:
    args = parse_args()
    apply_preset(args)
    args.tasks = expand_path(args.tasks)
    args.output_root = expand_path(args.output_root)
    args.wan_python = expand_path(args.wan_python)
    args.eval_python = expand_path(args.eval_python)
    if args.generate_only and args.evaluate_only:
        raise ValueError("--generate-only and --evaluate-only cannot be used together.")
    if args.dry_run and args.evaluate_only:
        raise ValueError("--dry-run cannot be combined with --evaluate-only.")
    if not args.evaluate_only and not args.dry_run:
        validate_python_executable("wan-python", args.wan_python)
    if not args.generate_only and not args.dry_run:
        validate_python_executable("eval-python", args.eval_python)
    if not args.evaluate_only:
        generate(args)
    if args.dry_run:
        return
    if not args.generate_only:
        evaluate(args)
        print(f"[wan-ablation] report: {args.output_root / 'reports' / 'wan_pca_ablation_report.md'}")


if __name__ == "__main__":
    main()
