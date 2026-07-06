"""Unified model execution layer for generation benchmarks."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompt_builder import build_prompt, export_prompt_manifest
from .task_schema import TaskSpec, load_tasks
from .utils import ensure_dir, get_repo_root, load_yaml, timestamp_iso, write_jsonl
from .video_io import create_pseudo_reference_image


@dataclass
class ModelSpec:
    """A local model command configuration."""

    id: str
    type: str = "t2v"
    enabled: bool = False
    command_template: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            id=str(data.get("id", "")).strip(),
            type=str(data.get("type", "t2v")).strip(),
            enabled=bool(data.get("enabled", False)),
            command_template=str(data.get("command_template", "")).strip(),
        )


def load_models(path: Path) -> list[ModelSpec]:
    """Load model configurations from YAML."""

    payload = load_yaml(path) or {}
    models = payload.get("models", [])
    return [ModelSpec.from_dict(item) for item in models]


def _default_output_root() -> Path:
    return get_repo_root() / "outputs" / "ms_eval"


def _default_manifest_paths(output_root: Path) -> tuple[Path, Path]:
    return (
        output_root / "metrics" / "generation_manifest.jsonl",
        output_root / "metrics" / "prompts.jsonl",
    )


def _format_command(template: str, variables: dict[str, Any]) -> str:
    return template.format(**variables)


def _create_input_image_if_needed(
    task: TaskSpec,
    model_type: str,
    output_root: Path,
    model_id: str,
) -> str:
    if model_type.lower() not in {"ti2v", "i2v"}:
        return ""
    pseudo_dir = output_root / "pseudo_refs" / model_id
    ensure_dir(pseudo_dir)
    pseudo_path = pseudo_dir / f"{task.id}.png"
    create_pseudo_reference_image(task, pseudo_path)
    return str(pseudo_path)


def run_generation_benchmark(
    tasks: list[TaskSpec],
    models: list[ModelSpec],
    seeds: list[int],
    mode: str = "t2v",
    output_root: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    """Run generation commands for enabled models and benchmark tasks."""

    output_root = output_root or _default_output_root()
    video_root = output_root / "videos"
    ensure_dir(video_root)
    manifest_path, prompt_manifest_path = _default_manifest_paths(output_root)

    selected_tasks = tasks[:limit] if limit is not None else tasks
    prompt_records = [dict(record) for record in export_prompt_manifest(selected_tasks, prompt_manifest_path)]
    prompt_map = {record["task_id"]: record for record in prompt_records}
    manifest_rows: list[dict[str, Any]] = []
    ran_any = False

    for model in models:
        if not model.enabled:
            continue
        model_type = model.type.lower()
        if mode.lower() == "t2v" and model_type != "t2v":
            print(f"[generate] skip model={model.id}: model type is {model.type}, requested mode is {mode}")
            continue
        if mode.lower() in {"ti2v", "i2v"} and model_type not in {"ti2v", "i2v"}:
            print(f"[generate] skip model={model.id}: model type is {model.type}, requested mode is {mode}")
            continue
        ran_any = True
        model_video_dir = video_root / model.id
        ensure_dir(model_video_dir)
        for task in selected_tasks:
            prompt = prompt_map[task.id]["prompt"] if task.id in prompt_map else build_prompt(task)
            input_image = _create_input_image_if_needed(task, model.type, output_root, model.id)
            for seed in seeds:
                output_path = model_video_dir / f"{task.id}_seed{seed}.mp4"
                variables = {
                    "prompt": prompt,
                    "output_path": str(output_path),
                    "duration_sec": task.duration_sec,
                    "fps": task.fps,
                    "seed": seed,
                    "task_id": task.id,
                    "model_id": model.id,
                    "input_image": input_image,
                }
                try:
                    command = _format_command(model.command_template, variables)
                except Exception as exc:
                    row = {
                        "model_id": model.id,
                        "task_id": task.id,
                        "seed": seed,
                        "prompt": prompt,
                        "output_path": str(output_path),
                        "command": model.command_template,
                        "status": "failed",
                        "error_message": f"command_template format failed: {exc}",
                        "timestamp": timestamp_iso(),
                        "input_image": input_image,
                    }
                    manifest_rows.append(row)
                    print(f"[generate][error] model={model.id} task={task.id} seed={seed}: {row['error_message']}")
                    continue
                print(
                    f"[generate] model={model.id} task={task.id} seed={seed} output={output_path}"
                )
                if skip_existing and output_path.exists():
                    row = {
                        "model_id": model.id,
                        "task_id": task.id,
                        "seed": seed,
                        "prompt": prompt,
                        "output_path": str(output_path),
                        "command": command,
                        "status": "skipped_existing",
                        "error_message": "",
                        "timestamp": timestamp_iso(),
                        "input_image": input_image,
                    }
                    manifest_rows.append(row)
                    continue
                if dry_run:
                    row = {
                        "model_id": model.id,
                        "task_id": task.id,
                        "seed": seed,
                        "prompt": prompt,
                        "output_path": str(output_path),
                        "command": command,
                        "status": "dry_run",
                        "error_message": "",
                        "timestamp": timestamp_iso(),
                        "input_image": input_image,
                    }
                    manifest_rows.append(row)
                    continue
                ensure_dir(output_path.parent)
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )
                status = "success" if result.returncode == 0 and output_path.exists() else "failed"
                error_message = ""
                if status != "success":
                    error_message = (result.stderr or result.stdout or "").strip()
                    print(
                        f"[generate][error] model={model.id} task={task.id} seed={seed} "
                        f"returncode={result.returncode}"
                    )
                    if error_message:
                        print(error_message)
                row = {
                    "model_id": model.id,
                    "task_id": task.id,
                    "seed": seed,
                    "prompt": prompt,
                    "output_path": str(output_path),
                    "command": command,
                    "status": status,
                    "error_message": error_message,
                    "timestamp": timestamp_iso(),
                    "input_image": input_image,
                }
                manifest_rows.append(row)

    write_jsonl(manifest_path, manifest_rows)
    if not ran_any:
        print(
            "[generate] no enabled models matched the requested mode. "
            "Set enabled: true in configs/ms_eval_models.yaml and check --mode."
        )
    return manifest_rows
