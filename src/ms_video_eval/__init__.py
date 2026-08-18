"""Multi-subject video evaluation framework."""

from __future__ import annotations

from importlib import import_module
from typing import Any

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


_LAZY_IMPORTS = {
    "SubjectSpec": (".task_schema", "SubjectSpec"),
    "TaskSpec": (".task_schema", "TaskSpec"),
    "load_tasks": (".task_schema", "load_tasks"),
    "build_prompt": (".prompt_builder", "build_prompt"),
    "export_prompt_manifest": (".prompt_builder", "export_prompt_manifest"),
    "ModelSpec": (".model_runner", "ModelSpec"),
    "load_models": (".model_runner", "load_models"),
    "run_generation_benchmark": (".model_runner", "run_generation_benchmark"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
