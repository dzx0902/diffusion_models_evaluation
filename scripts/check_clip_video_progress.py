"""Report the current AnimateDiff/EEG experiment stage without changing files."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CLIP-video baseline progress.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(os.environ.get("MS_MODELS_ROOT", ROOT / ".ms_video_models")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "eeg_clip_video",
    )
    parser.add_argument("--fold", default="video_6fold_1")
    parser.add_argument("--min-yolo-score", type=float, default=0.5)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def weight_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and item.suffix.lower() in WEIGHT_SUFFIXES
        and not item.name.endswith(".partial")
        and item.stat().st_size > 0
    )


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], None
    rows: list[dict[str, Any]] = []
    number = 0
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                rows.append(json.loads(line))
    except Exception as error:  # Report malformed partial output instead of aborting diagnostics.
        return rows, f"line {number}: {error}"
    return rows, None


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def latest_history(path: Path) -> dict[str, Any] | None:
    rows, error = read_jsonl(path)
    if error or not rows:
        return None
    return rows[-1]


def read_yolo_scores(root: Path) -> tuple[dict[str, float], dict[str, float]]:
    generated: dict[str, float] = {}
    references: dict[str, float] = {}
    if not root.is_dir():
        return generated, references
    for path in root.rglob("video_scores.csv"):
        destination = references if "yolo_reference" in path.parts else generated
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    name = Path(row["video_file"]).name
                    score = float(row["yolo_entity_score"])
                    destination[name] = max(score, destination.get(name, float("-inf")))
        except (KeyError, OSError, ValueError):
            continue
    return generated, references


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    models_root = args.models_root.expanduser()
    output_root = args.output_root.expanduser()
    base = models_root / "AnimateDiff" / "sd-v1-5"
    motion = models_root / "AnimateDiff" / "motion-adapter-v1-5-2"
    target_root = output_root / "animatediff" / "targets"
    target_index = target_root / "index.jsonl"
    condition_targets = output_root / "animatediff" / "condition_targets.jsonl"
    manifest_candidates = [
        ROOT / "data" / "manifests" / "structured_v2_video_manifest.jsonl",
        ROOT / "data" / "manifests" / "captions_simplified.jsonl",
    ]
    manifest = next((path for path in manifest_candidates if path.is_file()), None)
    manifest_rows, manifest_error = read_jsonl(manifest) if manifest else ([], None)
    expected_videos = len({str(row.get("video_id")) for row in manifest_rows})

    report: dict[str, Any] = {
        "code": {
            "head": git_value("rev-parse", "--short", "HEAD"),
            "branch": git_value("branch", "--show-current"),
        },
        "environment": {
            "python": sys.executable,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "packages": {
                name: package_version(name)
                for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")
            },
        },
        "paths": {
            "models_root": str(models_root),
            "models_root_resolved": str(models_root.resolve()) if models_root.exists() else None,
            "output_root": str(output_root),
            "manifest": str(manifest) if manifest else None,
        },
    }

    components: dict[str, Any] = {}
    for name, path in {
        "text_encoder": base / "text_encoder",
        "unet": base / "unet",
        "vae": base / "vae",
        "motion_adapter": motion,
    }.items():
        weights = weight_files(path)
        components[name] = {
            "path": str(path),
            "config": (path / "config.json").is_file(),
            "weights": [
                {"name": item.name, "size": item.stat().st_size}
                for item in weights
            ],
            "ready": (path / "config.json").is_file() and bool(weights),
        }
    partials = sorted(models_root.rglob("*.partial")) if models_root.is_dir() else []
    model_ready = (
        (base / "model_index.json").is_file()
        and (base / "scheduler" / "scheduler_config.json").is_file()
        and (base / "tokenizer").is_dir()
        and all(component["ready"] for component in components.values())
    )
    report["model"] = {
        "ready": model_ready,
        "components": components,
        "partial_files": [
            {"path": str(path), "size": path.stat().st_size} for path in partials
        ],
    }

    index_rows, index_error = read_jsonl(target_index)
    index_shapes = sorted({tuple(row.get("shape", [])) for row in index_rows})
    index_ready = bool(index_rows) and index_error is None and index_shapes == [(77, 768)]
    bound_rows, bound_error = read_jsonl(condition_targets)
    bound_ids = [str(row.get("video_id")) for row in bound_rows]
    bound_paths_exist = bool(bound_rows) and all(
        Path(str(row.get("latent_path", ""))).is_file() for row in bound_rows
    )
    binding_ready = (
        bool(bound_rows)
        and bound_error is None
        and len(bound_ids) == len(set(bound_ids))
        and (expected_videos == 0 or len(bound_rows) == expected_videos)
        and bound_paths_exist
    )
    report["targets"] = {
        "manifest_videos": expected_videos,
        "manifest_error": manifest_error,
        "index_rows": len(index_rows),
        "index_shapes": [list(shape) for shape in index_shapes],
        "index_error": index_error,
        "index_ready": index_ready,
        "condition_target_rows": len(bound_rows),
        "condition_target_error": bound_error,
        "condition_paths_exist": bound_paths_exist,
        "binding_ready": binding_ready,
    }

    exact_root = output_root / "animatediff" / "exact_check"
    exact_videos = sorted(exact_root.rglob("*.mp4")) if exact_root.is_dir() else []
    valid_videos = [path for path in exact_videos if path.stat().st_size > 0]
    exact_ready = any("exact" in path.name.lower() for path in valid_videos)
    native_ready = any("native" in path.name.lower() for path in valid_videos)
    generated_scores, reference_scores = read_yolo_scores(exact_root)
    exact_scores = [score for name, score in generated_scores.items() if "exact" in name.lower()]
    native_scores = [score for name, score in generated_scores.items() if "native" in name.lower()]
    yolo_gate_scored = bool(exact_scores) and bool(native_scores)
    yolo_gate_passed = (
        yolo_gate_scored
        and max(exact_scores) >= args.min_yolo_score
        and max(native_scores) >= args.min_yolo_score
    )
    report["generation_gate"] = {
        "exact_ready": exact_ready,
        "native_ready": native_ready,
        "min_yolo_score": args.min_yolo_score,
        "yolo_scored": yolo_gate_scored,
        "yolo_passed": yolo_gate_passed,
        "generated_yolo_scores": generated_scores,
        "reference_yolo_scores": reference_scores,
        "videos": [
            {"path": str(path), "size": path.stat().st_size} for path in valid_videos
        ],
    }

    training: dict[str, Any] = {}
    for subject in ("chentianlin", "duzhuoxuan"):
        directory = output_root / "animatediff" / subject / args.fold
        last = latest_history(directory / "history.jsonl")
        training[subject] = {
            "directory": str(directory),
            "last_epoch": int(last["epoch"]) if last else None,
            "last_valid": last.get("valid") if last else None,
            "best_checkpoint": (directory / "best.pt").is_file(),
            "last_checkpoint": (directory / "last.pt").is_file(),
        }
    report["training"] = training

    controls_root = output_root / "animatediff" / "controls"
    control_videos = sorted(controls_root.rglob("*.mp4")) if controls_root.is_dir() else []
    report["controls"] = {
        "videos": [
            {"path": str(path), "size": path.stat().st_size}
            for path in control_videos
            if path.stat().st_size > 0
        ]
    }

    required_packages = report["environment"]["packages"]
    environment_ready = all(required_packages.values())
    first_trained = training["chentianlin"]["best_checkpoint"]
    second_trained = training["duzhuoxuan"]["best_checkpoint"]
    control_names = {Path(row["path"]).name.lower() for row in report["controls"]["videos"]}
    matched_controls_ready = any("correct_eeg" in name for name in control_names)
    if not model_ready:
        next_stage = "MODEL_DOWNLOAD_OR_LAYOUT"
    elif not environment_ready:
        next_stage = "INSTALL_CLIP_VIDEO_ENV"
    elif not index_ready:
        next_stage = "EXPORT_CLIP_TARGETS"
    elif not binding_ready:
        next_stage = "BUILD_VIDEO_TARGET_BINDING"
    elif not exact_ready:
        next_stage = "GENERATE_EXACT_TARGET"
    elif not native_ready:
        next_stage = "GENERATE_NATIVE_CONTROL"
    elif not yolo_gate_scored:
        next_stage = "SCORE_GENERATION_GATE"
    elif not yolo_gate_passed:
        next_stage = "DIAGNOSE_GENERATION_GATE"
    elif not first_trained:
        next_stage = "TRAIN_CHENTIANLIN_PILOT"
    elif not matched_controls_ready:
        next_stage = "RUN_MATCHED_CONTROLS"
    elif not second_trained:
        next_stage = "TRAIN_DUZHUOXUAN"
    else:
        next_stage = "SCORE_AND_COMPARE_MODELS"
    report["next_stage"] = next_stage

    print("=== CODE ===")
    print(f"HEAD={report['code']['head']} branch={report['code']['branch']}")
    print("=== ENVIRONMENT ===")
    print(f"conda={report['environment']['conda_env']} python={report['environment']['python']}")
    for name, version in required_packages.items():
        print(f"{'OK' if version else 'MISSING':7} {name}={version}")
    print("=== MODEL ===")
    print(f"models_root={models_root}")
    print(f"model_ready={model_ready}")
    for name, component in components.items():
        weights = ", ".join(
            f"{item['name']}({format_size(item['size'])})" for item in component["weights"]
        ) or "none"
        print(f"{'OK' if component['ready'] else 'MISSING':7} {name}: {weights}")
    for partial in report["model"]["partial_files"]:
        print(f"PARTIAL {partial['path']} ({format_size(partial['size'])})")
    print("=== TARGETS ===")
    print(
        f"manifest_videos={expected_videos} index_rows={len(index_rows)} "
        f"shapes={report['targets']['index_shapes']} bound_rows={len(bound_rows)} "
        f"paths_exist={bound_paths_exist}"
    )
    if index_error or bound_error or manifest_error:
        print(f"ERROR manifest={manifest_error} index={index_error} binding={bound_error}")
    print("=== GENERATION GATE ===")
    print(f"exact={exact_ready} native={native_ready} videos={len(valid_videos)}")
    print(
        f"yolo_scored={yolo_gate_scored} yolo_passed={yolo_gate_passed} "
        f"threshold={args.min_yolo_score}"
    )
    for name, score in sorted(generated_scores.items()):
        print(f"YOLO generated {name}: {score:.3f}")
    for name, score in sorted(reference_scores.items()):
        print(f"YOLO reference {name}: {score:.3f}")
    for row in report["generation_gate"]["videos"]:
        print(f"VIDEO {row['path']} ({format_size(row['size'])})")
    print("=== TRAINING ===")
    for subject, state in training.items():
        print(
            f"{subject}: epoch={state['last_epoch']} best={state['best_checkpoint']} "
            f"last={state['last_checkpoint']}"
        )
    print(f"controls={len(report['controls']['videos'])}")
    print(f"NEXT_STAGE={next_stage}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON_REPORT={args.json_output}")


if __name__ == "__main__":
    main()
