"""Task schema and YAML loading for multi-subject evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import load_yaml


@dataclass
class SubjectSpec:
    """A single subject in a benchmark task."""

    name: str
    count: int = 1
    initial_position: str = ""
    motion: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectSpec":
        return cls(
            name=str(data.get("name", "")).strip(),
            count=int(data.get("count", 1)),
            initial_position=str(data.get("initial_position", "")).strip(),
            motion=str(data.get("motion", "")).strip(),
        )


@dataclass
class TaskSpec:
    """A benchmark task definition."""

    id: str
    subjects: list[SubjectSpec] = field(default_factory=list)
    scene: str = ""
    camera: str = ""
    duration_sec: int = 4
    fps: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        subjects = [SubjectSpec.from_dict(item) for item in data.get("subjects", [])]
        return cls(
            id=str(data.get("id", "")).strip(),
            subjects=subjects,
            scene=str(data.get("scene", "")).strip(),
            camera=str(data.get("camera", "")).strip(),
            duration_sec=int(data.get("duration_sec", 4)),
            fps=int(data.get("fps", 16)),
        )

    @property
    def subject_a(self) -> SubjectSpec:
        return self.subjects[0]

    @property
    def subject_b(self) -> SubjectSpec:
        return self.subjects[1]

    def prompt_subjects(self) -> list[str]:
        return [subject.name for subject in self.subjects]


def load_tasks(path: Path) -> list[TaskSpec]:
    """Load a list of tasks from a YAML file."""

    payload = load_yaml(path) or {}
    tasks = payload.get("tasks", [])
    return [TaskSpec.from_dict(item) for item in tasks]

