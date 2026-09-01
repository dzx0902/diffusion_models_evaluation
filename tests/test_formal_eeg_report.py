from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_formal_eeg_ablation_report import phase1_sections, phase2_sections


def test_phase1_report_separates_primary_and_exploratory_rows() -> None:
    payload = {
        "subject": "chentianlin", "fold": "video_6fold_1", "training_seed": 42,
        "rows": [
            {
                "scope": "first6_468", "comparable_group": "first6_temporal",
                "variant": "a_4s_first6", "decoder": "mean_logit",
                "test_videos": 78, "primary_metric": "category_accuracy",
                "primary_value": 0.2821, "chance": 1 / 6, "object_exact": 0.2821,
                "invalid_pair_rate": 0.0, "primary_analysis": True,
            },
            {
                "scope": "first6_468", "comparable_group": "first6_temporal",
                "variant": "a_4s_first6", "decoder": "object_top2",
                "test_videos": 78, "primary_metric": "category_accuracy",
                "primary_value": 0.1410, "chance": 1 / 6, "object_exact": 0.1410,
                "invalid_pair_rate": 0.1923, "primary_analysis": False,
            },
        ],
    }
    lines, rows = phase1_sections(payload)
    text = "\n".join(lines)
    assert len(rows) == 2
    assert "Primary results" in text
    assert "Exploratory decoder audit" in text
    assert "object_top2" in text
    assert "a_4s_first6 / mean_logit" in text


def test_phase2_pending_without_video_metrics(tmp_path: Path) -> None:
    lines, status = phase2_sections(None, [], tmp_path)
    assert status["status"] == "PENDING"
    assert "PENDING" in "\n".join(lines)


def test_phase2_writes_summary_and_paired_tests(tmp_path: Path) -> None:
    path = tmp_path / "video.csv"
    fields = [
        "variant", "generator", "subject", "fold", "seed",
        "generation_seed", "video_id", "metric", "value",
    ]
    rows = []
    for variant, offset in (("a_base", 0.0), ("a_enhanced", 0.1)):
        for video in ("01-001", "01-002"):
            rows.append({
                "variant": variant, "generator": "animatediff",
                "subject": "chentianlin", "fold": "video_6fold_1", "seed": 42,
                "generation_seed": 0, "video_id": video,
                "metric": "semantic_clip_score", "value": 0.5 + offset,
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines, status = phase2_sections(path, [{"status": "PASS", "jobs": 4}], tmp_path)
    assert status["status"] == "COMPLETE"
    assert status["paired_comparisons"] == 1
    assert (tmp_path / "video_summary.csv").is_file()
    assert (tmp_path / "video_paired_tests.csv").is_file()
    assert "a_enhanced" in "\n".join(lines)


def test_phase2_compares_main_three_methods_in_normalized_tora_family(tmp_path: Path) -> None:
    path = tmp_path / "video.csv"
    fields = [
        "variant", "generator", "subject", "fold", "seed",
        "generation_seed", "video_id", "metric", "value",
    ]
    rows = []
    for variant, offset in (("a_enhanced", 0.0), ("b_enhanced", 0.1), ("c2_full", 0.2)):
        for video in ("01-001", "01-002"):
            rows.append({
                "variant": variant, "generator": "tora",
                "subject": "chentianlin", "fold": "video_6fold_1", "seed": 42,
                "generation_seed": 0, "video_id": video,
                "metric": "semantic_clip_score", "value": 0.5 + offset,
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _, status = phase2_sections(path, [{"status": "PASS", "jobs": 6}], tmp_path)
    assert status["paired_comparisons"] == 2
