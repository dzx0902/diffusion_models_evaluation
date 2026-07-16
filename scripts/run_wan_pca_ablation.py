"""Generate and evaluate Wan text-compression ablations.

Variants:
  baseline      no compression
  d768          feature PCA 4096 -> 768 -> 4096
  t64           token resample N -> 64 -> N
  d768_t64      feature PCA plus token resampling
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan PCA/token-count ablation benchmark.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[
            "dog_car_walk_static",
            "dog_ball_walk_roll",
            "person_bicycle_walk_static",
            "car_flower_static_sway",
            "bird_flower_land_near",
        ],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "d1024", "d768", "d512", "d256", "t64", "t32", "d768_t64"],
    )
    parser.add_argument(
        "--projector",
        type=Path,
        default=ROOT / "outputs" / "text_space" / "wan2_2_ti2v_5b" / "token_pca_projector.npz",
    )
    parser.add_argument("--wan-repo", type=Path, default=Path("$MS_MODELS_ROOT/Wan2.2"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("$MS_MODELS_ROOT/Wan2.2/Wan2.2-TI2V-5B"))
    parser.add_argument("--size", type=str, default="1280*704")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "wan_pca_ablation")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs" / "ms_eval_settings.wsl.yaml")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--report-error", action="store_true")
    return parser.parse_args()


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


def generate(args: argparse.Namespace) -> None:
    tasks = {task.id: task for task in load_tasks(args.tasks)}
    selected_tasks = [tasks[task_id] for task_id in args.task_ids]
    videos_root = args.output_root / "videos"
    wan_repo = expand_path(args.wan_repo)
    ckpt_dir = expand_path(args.ckpt_dir)
    projector = expand_path(args.projector)

    for task in selected_tasks:
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
                    sys.executable,
                    str(ROOT / "scripts" / "adapters" / "wan_projected_generate.py"),
                    "--wan-repo",
                    str(wan_repo),
                ]
                if dim > 0:
                    wrapper_args.extend(["--projector", str(projector), "--project-dim", str(dim)])
                else:
                    wrapper_args.append("--disable-projection")
                if token_count > 0:
                    wrapper_args.extend(["--token-count", str(token_count)])
                if args.report_error:
                    wrapper_args.append("--report-error")

                command = [
                    *wrapper_args,
                    "--",
                    "--task",
                    "ti2v-5B",
                    "--size",
                    args.size,
                    "--ckpt_dir",
                    str(ckpt_dir),
                    "--offload_model",
                    "True",
                    "--convert_model_dtype",
                    "--t5_cpu",
                    "--base_seed",
                    str(seed),
                    "--prompt",
                    prompt,
                    "--save_file",
                    str(output_path),
                ]
                run(command)


def evaluate(args: argparse.Namespace) -> None:
    frames_root = args.output_root / "frames"
    detections_root = args.output_root / "detections"
    metrics_root = args.output_root / "metrics"
    run(
        [
            sys.executable,
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
            str(args.output_root / "reports" / "wan_pca_ablation_report.md"),
        ]
    )


def main() -> None:
    args = parse_args()
    if not args.evaluate_only:
        generate(args)
    if not args.generate_only:
        evaluate(args)
        print(f"[wan-ablation] report: {args.output_root / 'reports' / 'wan_pca_ablation_report.md'}")


if __name__ == "__main__":
    main()
