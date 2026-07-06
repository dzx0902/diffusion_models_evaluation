"""Metric computation for multi-subject video generation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2

from .task_schema import TaskSpec
from .utils import ensure_dir, read_json, read_jsonl, safe_float, safe_int, timestamp_iso


@dataclass
class SubjectMetrics:
    presence_ratio: float = 0.0
    count_accuracy: float = 0.0
    tp: float = 0.0


def _load_frame_detections(detection_path: Path) -> list[dict[str, Any]]:
    data = read_json(detection_path)
    return data.get("detections", [])


def _frame_subject_detections(frame_entry: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for det in frame_entry.get("detections", []):
        grouped.setdefault(det.get("class_name", ""), []).append(det)
    return grouped


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _parse_position_tokens(position: str) -> list[str]:
    tokens = position.lower().replace("-", " ").split()
    return tokens


def _position_matches(position: str, bbox: list[float], width: int, height: int) -> bool:
    if not position:
        return True
    x, y = _bbox_center(bbox)
    pos = position.lower()
    tokens = _parse_position_tokens(position)
    checks: list[bool] = []
    if "left" in tokens:
        checks.append(x < 0.45 * width)
    if "right" in tokens:
        checks.append(x > 0.55 * width)
    if "upper" in tokens:
        checks.append(y < 0.45 * height)
    if "lower" in tokens:
        checks.append(y > 0.55 * height)
    if "center" in tokens:
        checks.append(0.35 * width <= x <= 0.65 * width and 0.35 * height <= y <= 0.65 * height)
    return all(checks) if checks else True


def _subject_visibility_series(
    detections: list[dict[str, Any]],
    subject_name: str,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for frame_entry in detections:
        grouped = _frame_subject_detections(frame_entry)
        items = grouped.get(subject_name, [])
        if items:
            best = max(items, key=lambda item: safe_float(item.get("confidence", 0.0)))
            series.append({"present": True, "bbox": best.get("bbox", []), "confidence": best.get("confidence", 0.0)})
        else:
            series.append({"present": False, "bbox": None, "confidence": 0.0})
    return series


def _longest_true_segment(flags: list[bool]) -> int:
    longest = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _motion_compliance(subject_motion: str, subject_series: list[dict[str, Any]], width: int, height: int) -> float:
    visible = [entry for entry in subject_series if entry["present"] and entry["bbox"]]
    if len(visible) < 2:
        return 0.0
    start = visible[0]["bbox"]
    end = visible[-1]["bbox"]
    sx, sy = _bbox_center(start)
    ex, ey = _bbox_center(end)
    dx = ex - sx
    dy = ey - sy
    diag = math.sqrt(width * width + height * height)
    norm_disp = math.sqrt(dx * dx + dy * dy) / diag
    motion = subject_motion.lower()

    if "stationary" in motion or "remains stationary" in motion:
        return 1.0 if norm_disp <= 0.08 else 0.0
    if "toward the center" in motion:
        center_x = width / 2.0
        center_y = height / 2.0
        start_dist = math.hypot(sx - center_x, sy - center_y)
        end_dist = math.hypot(ex - center_x, ey - center_y)
        return 1.0 if end_dist < start_dist else 0.0
    if "to the right" in motion:
        return 1.0 if ex > sx + 0.05 * width else 0.0
    if "to the left" in motion:
        return 1.0 if ex < sx - 0.05 * width else 0.0
    if "sways" in motion:
        y_positions = [(_bbox_center(entry["bbox"])[1]) for entry in visible]
        y_range = max(y_positions) - min(y_positions) if y_positions else 0.0
        return 1.0 if y_range >= 0.02 * height and norm_disp <= 0.20 else 0.0
    return 1.0 if norm_disp >= 0.05 else 0.0


def _subject_presence_metrics(
    subject: dict[str, Any],
    subject_series: list[dict[str, Any]],
    total_frames: int,
) -> SubjectMetrics:
    presence_flags = [entry["present"] for entry in subject_series]
    presence_ratio = sum(presence_flags) / total_frames if total_frames else 0.0
    count_accuracy = presence_ratio
    tp = _longest_true_segment(presence_flags) / total_frames if total_frames else 0.0
    return SubjectMetrics(presence_ratio=presence_ratio, count_accuracy=count_accuracy, tp=tp)


def _compute_spatial_relation(
    task: TaskSpec,
    detections: list[dict[str, Any]],
    width: int,
    height: int,
) -> float:
    if not detections:
        return 0.0
    checked_frames = detections[: max(1, len(detections) // 4)]
    hits = 0
    total = 0
    for frame_entry in checked_frames:
        grouped = _frame_subject_detections(frame_entry)
        frame_ok = True
        for subject in task.subjects:
            items = grouped.get(subject.name, [])
            if not items:
                frame_ok = False
                break
            best = max(items, key=lambda item: safe_float(item.get("confidence", 0.0)))
            if not _position_matches(subject.initial_position, best.get("bbox", []), width, height):
                frame_ok = False
                break
        hits += int(frame_ok)
        total += 1
    return hits / total if total else 0.0


def _compute_class_correctness(subject_metrics: list[SubjectMetrics]) -> float:
    reliable = sum(1 for item in subject_metrics if item.presence_ratio >= 0.6)
    if not subject_metrics:
        return 0.0
    return reliable / len(subject_metrics)


def _compute_subject_count_accuracy(
    task: TaskSpec,
    detections: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    total = len(detections)
    subject_ratios: dict[str, float] = {}
    correct_frames = 0
    for subject in task.subjects:
        subject_ratios[subject.name] = 0.0
    for frame_entry in detections:
        grouped = _frame_subject_detections(frame_entry)
        frame_ok = True
        for subject in task.subjects:
            count = len(grouped.get(subject.name, []))
            if count == subject.count:
                subject_ratios[subject.name] += 1.0
            else:
                frame_ok = False
        if frame_ok:
            correct_frames += 1
    for subject in task.subjects:
        subject_ratios[subject.name] = subject_ratios[subject.name] / total if total else 0.0
    return (correct_frames / total if total else 0.0, subject_ratios)


def evaluate_video(
    task: TaskSpec,
    detection_path: Path,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """Compute all benchmark metrics for a single video."""

    detection_data = read_json(detection_path)
    detections = detection_data.get("detections", [])
    total_frames = len(detections)
    if total_frames == 0 and video_path is not None and video_path.exists():
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

    width = 1
    height = 1
    if detections:
        first_frame_path = detections[0].get("frame_path")
        if first_frame_path:
            frame = cv2.imread(str(first_frame_path))
            if frame is not None:
                height, width = frame.shape[:2]

    subject_results: dict[str, SubjectMetrics] = {}
    subject_series_map: dict[str, list[dict[str, Any]]] = {}
    for subject in task.subjects:
        series = _subject_visibility_series(detections, subject.name)
        subject_series_map[subject.name] = series
        subject_results[subject.name] = _subject_presence_metrics(subject.__dict__, series, total_frames)

    spa_all = 0.0
    sca, subject_count_ratios = _compute_subject_count_accuracy(task, detections)
    if detections:
        all_presence_hits = 0
        for idx in range(total_frames):
            frame_ok = True
            for subject in task.subjects:
                if not subject_series_map[subject.name][idx]["present"]:
                    frame_ok = False
                    break
            all_presence_hits += int(frame_ok)
        spa_all = all_presence_hits / total_frames if total_frames else 0.0

    spa_subject = {f"spa_subject_{subject.name}": result.presence_ratio for subject, result in zip(task.subjects, subject_results.values())}
    tp_subject = {f"tp_subject_{subject.name}": result.tp for subject, result in zip(task.subjects, subject_results.values())}
    tp_all = min((item.tp for item in subject_results.values()), default=0.0)
    cc = _compute_class_correctness(list(subject_results.values()))
    sra = _compute_spatial_relation(task, detections, width, height)
    motion_scores = [
        _motion_compliance(subject.motion, subject_series_map[subject.name], width, height)
        for subject in task.subjects
    ]
    mc = mean(motion_scores) if motion_scores else 0.0
    fusion_hits = 0
    fusion_total = 0
    for frame_entry in detections:
        grouped = _frame_subject_detections(frame_entry)
        boxes = []
        for subject in task.subjects:
            items = grouped.get(subject.name, [])
            if items:
                best = max(items, key=lambda item: safe_float(item.get("confidence", 0.0)))
                boxes.append(best.get("bbox", []))
        if len(boxes) >= 2:
            fusion_total += 1
            overlap = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if _bbox_iou(boxes[i], boxes[j]) > 0.35:
                        overlap = True
                        break
                if overlap:
                    break
            fusion_hits += int(overlap)
    sfr = fusion_hits / fusion_total if fusion_total else 0.0
    vq = None
    ms_vgs = (
        0.20 * spa_all
        + 0.15 * sca
        + 0.15 * cc
        + 0.15 * tp_all
        + 0.10 * sra
        + 0.10 * mc
        + 0.10 * (1.0 - sfr)
        + 0.05 * (0.5 if vq is None else vq)
    )

    failure_flags = []
    if any(result.presence_ratio < 0.5 for result in subject_results.values()):
        failure_flags.append("missing_subject")
    if sca < 0.5:
        failure_flags.append("wrong_count")
    if cc < 0.5:
        failure_flags.append("wrong_class")
    if tp_all < 0.5:
        failure_flags.append("low_persistence")
    if sra < 0.5:
        failure_flags.append("wrong_spatial_relation")
    if mc < 0.5:
        failure_flags.append("wrong_motion")
    if sfr > 0.1:
        failure_flags.append("possible_fusion")
    if not detections or all(not frame.get("detections") for frame in detections):
        failure_flags.append("detection_unreliable")

    row: dict[str, Any] = {
        "timestamp": timestamp_iso(),
        "task_id": task.id,
        "model_id": detection_path.parent.name,
        "seed": detection_path.stem.split("_seed")[-1] if "_seed" in detection_path.stem else "",
        "total_sampled_frames": total_frames,
        "spa_all": spa_all,
        "sca": sca,
        "cc": cc,
        "tp_all": tp_all,
        "sra": sra,
        "mc": mc,
        "sfr": sfr,
        "vq": vq,
        "ms_vgs": ms_vgs,
        "failure_flags": "|".join(failure_flags),
        "detection_unreliable": int("detection_unreliable" in failure_flags),
    }
    row.update(spa_subject)
    row.update(tp_subject)
    for subject_name, ratio in subject_count_ratios.items():
        row[f"count_accuracy_subject_{subject_name}"] = ratio
    return row


def evaluate_videos(
    tasks: list[TaskSpec],
    frames_root: Path,
    detections_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Evaluate all generated videos for a set of tasks."""

    task_map = {task.id: task for task in tasks}
    rows: list[dict[str, Any]] = []
    for model_dir in sorted([path for path in detections_root.iterdir() if path.is_dir()]):
        for detection_file in sorted(model_dir.glob("*.json")):
            stem = detection_file.stem
            if "_seed" not in stem:
                continue
            task_id = stem[: stem.rfind("_seed")]
            task = task_map.get(task_id)
            if task is None:
                continue
            seed = stem.split("_seed")[-1]
            video_path = Path(frames_root.parent / "videos" / model_dir.name / f"{task_id}_seed{seed}.mp4")
            row = evaluate_video(task, detection_file, video_path if video_path.exists() else None)
            row["model_id"] = model_dir.name
            row["task_id"] = task_id
            row["seed"] = safe_int(seed)
            rows.append(row)
    ensure_dir(output_root)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (output_root / "video_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows


def aggregate_metrics(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    """Aggregate per-video metrics by a key such as model_id or task_id."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key, "")), []).append(row)
    aggregated: list[dict[str, Any]] = []
    metric_keys = ["spa_all", "sca", "cc", "tp_all", "sra", "mc", "sfr", "ms_vgs"]
    for key, items in grouped.items():
        record = {group_key: key, "video_count": len(items)}
        for metric in metric_keys:
            values = [safe_float(item.get(metric, 0.0)) for item in items]
            record[metric] = sum(values) / len(values) if values else 0.0
        aggregated.append(record)
    aggregated.sort(key=lambda item: item.get("ms_vgs", 0.0), reverse=True)
    return aggregated


def build_failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize failure flags across videos."""

    counts: dict[tuple[str, str], int] = {}
    total_by_model: dict[str, int] = {}
    for row in rows:
        model_id = str(row.get("model_id", ""))
        total_by_model[model_id] = total_by_model.get(model_id, 0) + 1
        flags = [flag for flag in str(row.get("failure_flags", "")).split("|") if flag]
        for flag in flags:
            counts[(model_id, flag)] = counts.get((model_id, flag), 0) + 1
    summary: list[dict[str, Any]] = []
    for (model_id, flag), count in sorted(counts.items()):
        total = total_by_model.get(model_id, 1)
        summary.append(
            {
                "model_id": model_id,
                "failure_type": flag,
                "count": count,
                "rate": count / total if total else 0.0,
            }
        )
    return summary

