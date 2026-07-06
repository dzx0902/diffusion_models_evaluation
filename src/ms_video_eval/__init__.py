"""Multi-subject video evaluation framework."""

from .task_schema import SubjectSpec, TaskSpec, load_tasks
from .prompt_builder import build_prompt, export_prompt_manifest
from .model_runner import ModelSpec, load_models, run_generation_benchmark

__all__ = [
    "SubjectSpec",
    "TaskSpec",
    "ModelSpec",
    "load_tasks",
    "load_models",
    "build_prompt",
    "export_prompt_manifest",
    "run_generation_benchmark",
]

