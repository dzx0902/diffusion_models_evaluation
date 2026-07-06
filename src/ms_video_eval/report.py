"""Markdown report generation for the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .task_schema import load_tasks
from .utils import format_value, read_csv, read_jsonl


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---" if idx == 0 else "---:" for idx, _ in enumerate(columns)]) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            values.append(format_value(row.get(column, "")))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _sorted_rows(rows: list[dict[str, Any]], key: str = "ms_vgs") -> list[dict[str, Any]]:
    def _score(row: dict[str, Any]) -> float:
        try:
            return float(row.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return sorted(rows, key=_score, reverse=True)


def build_report(metrics_dir: Path, tasks_path: Path | None = None, output_path: Path | None = None) -> str:
    """Build a markdown report from generated metrics files."""

    video_rows = read_csv(metrics_dir / "video_metrics.csv")
    model_rows = read_csv(metrics_dir / "model_summary.csv")
    task_rows = read_csv(metrics_dir / "task_summary.csv")
    failure_rows = read_csv(metrics_dir / "failure_summary.csv")
    prompt_rows = read_jsonl(metrics_dir / "prompts.jsonl")

    task_objects = load_tasks(tasks_path) if tasks_path and tasks_path.exists() else []
    task_lookup = {task.id: task for task in task_objects}

    lines: list[str] = []
    lines.append("# Multi-Subject Video Evaluation Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Videos evaluated: {len(video_rows)}")
    lines.append(f"- Tasks evaluated: {len(task_rows)}")
    lines.append(f"- Models evaluated: {len(model_rows)}")
    lines.append("- VQ strategy: missing VQ defaults to `0.5` in MS-VGS.")
    lines.append("- Detection reliability note: YOLO aliases are heuristic for open-vocabulary classes such as `flower`.")
    lines.append("")

    if model_rows:
        lines.append("## Model Ranking")
        columns = ["model_id", "video_count", "spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"]
        lines.append(_markdown_table(_sorted_rows(model_rows), columns))
        lines.append("")

        lines.append("## Model Sub-metrics")
        for row in _sorted_rows(model_rows):
            lines.append(f"### {row.get('model_id', '')}")
            lines.append(
                _markdown_table(
                    [row],
                    ["model_id", "video_count", "spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"],
                )
            )
            lines.append("")

    if task_rows:
        lines.append("## Task Summary")
        columns = ["task_id", "video_count", "spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"]
        lines.append(_markdown_table(_sorted_rows(task_rows), columns))
        lines.append("")

    if task_lookup and video_rows:
        lines.append("## Per-task Model Performance")
        for task_id, task in task_lookup.items():
            lines.append(f"### {task_id}")
            task_video_rows = [row for row in video_rows if row.get("task_id") == task_id]
            if not task_video_rows:
                continue
            task_video_rows = _sorted_rows(task_video_rows)
            lines.append(
                _markdown_table(
                    task_video_rows,
                    ["model_id", "seed", "spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"],
                )
            )
            lines.append("")
            subject_text = ", ".join(task.prompt_subjects())
            lines.append(f"- Subjects: {subject_text}")
            lines.append(f"- Scene: {task.scene}")
            lines.append(f"- Camera: {task.camera}")
            lines.append("")

    if failure_rows:
        lines.append("## Failure Summary")
        lines.append(_markdown_table(failure_rows, ["model_id", "failure_type", "count", "rate"]))
        lines.append("")

    if prompt_rows:
        lines.append("## Prompt Export")
        preview = prompt_rows[: min(3, len(prompt_rows))]
        lines.append(_markdown_table(preview, ["task_id", "duration_sec", "fps"]))
        lines.append("")

    lines.append("## Reliability and Limitations")
    lines.append("- `flower` is approximated through generic detector aliases such as `potted plant` or `vase`.")
    lines.append("- Fusion detection is a heuristic based on repeated IoU overlap.")
    lines.append("- Motion compliance uses bounding-box displacement, not semantic tracking.")
    lines.append("- The benchmark is designed for local GPU servers and assumes model generation happens outside the evaluation process.")
    lines.append("")

    lines.append("## Recommendation Template")
    lines.append("- Prefer models with higher MS-VGS and strong SPA/SCA balance.")
    lines.append("- If a model has low `wrong_class` but good `spa_all`, consider improving prompt grounding before retraining or switching models.")
    lines.append("- If `possible_fusion` is frequent, prioritize models with better multi-object separation or add first-frame anchoring.")

    report = "\n".join(lines).strip() + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report

