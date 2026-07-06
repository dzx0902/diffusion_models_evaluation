"""Prompt construction utilities for multi-subject benchmark tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .task_schema import TaskSpec
from .utils import ensure_dir, write_jsonl


def build_prompt(task: TaskSpec) -> str:
    """Build an English prompt from a task definition."""

    if len(task.subjects) < 2:
        raise ValueError(f"Task {task.id} requires at least two subjects")

    subject_a = task.subjects[0]
    subject_b = task.subjects[1]
    lines = [
        f"A realistic {task.duration_sec}-second video with two main subjects: {subject_a.name} and {subject_b.name}.",
        f"{subject_a.name} is located on the {subject_a.initial_position} side of the frame.",
        f"{subject_b.name} is located on the {subject_b.initial_position} side of the frame.",
        f"From 0 to {task.duration_sec} seconds, {subject_a.name} {subject_a.motion}.",
        f"{subject_b.name} {subject_b.motion}.",
        f"Scene: {task.scene}.",
        f"Camera: {task.camera}.",
        "Both subjects remain fully visible throughout the video.",
        f"There is exactly one {subject_a.name} and exactly one {subject_b.name}.",
        f"No extra {subject_a.name}s, no extra {subject_b.name}s, no unrelated main objects.",
        "Realistic video, natural lighting, stable motion, no visual artifacts.",
    ]
    if len(task.subjects) > 2:
        extra_subjects = ", ".join(subject.name for subject in task.subjects[2:])
        lines.insert(
            1,
            f"Additional subjects may appear only as background context: {extra_subjects}.",
        )
    return " ".join(lines)


def build_prompt_record(task: TaskSpec) -> dict[str, Any]:
    """Create a prompt export record for a task."""

    return {
        "task_id": task.id,
        "prompt": build_prompt(task),
        "subjects": [subject.__dict__ for subject in task.subjects],
        "duration_sec": task.duration_sec,
        "fps": task.fps,
    }


def export_prompt_manifest(tasks: list[TaskSpec], output_path: Path) -> list[dict[str, Any]]:
    """Export prompts as JSONL."""

    records = [build_prompt_record(task) for task in tasks]
    ensure_dir(output_path.parent)
    write_jsonl(output_path, records)
    return records

