from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_eeg_ablation_metrics import semantic_rows as collect_semantic_rows
from scripts.build_standardized_eeg_ablation_results import semantic_rows, temporal_rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_semantic_summary_preserves_full8_protocol_scope(tmp_path: Path) -> None:
    path = tmp_path / "runs_eeg2caption/a_base/chentianlin/video_6fold_1/seed42/test_semantic/predictions.json"
    write_json(path, {
        "video_aggregation_metrics": {
            "fused": {
                "category_accuracy": 0.25,
                "predicted_cardinality_exact_set": 0.2,
                "macro_ap": 0.4,
            },
            "structured": {},
        },
        "collapse_audit": {
            "video_count": 104, "unique_captions": 5, "passed": True,
        },
    })
    rows = semantic_rows(tmp_path, allow_missing=True)
    assert len(rows) == 1
    assert rows[0]["scope"] == "full8_624"
    assert rows[0]["test_videos"] == 104
    assert rows[0]["chance"] == 1 / 8


def test_temporal_posthoc_replaces_duplicate_base_row(tmp_path: Path) -> None:
    base = tmp_path / "runs_eeg2caption_temporal/a_4s_first6/chentianlin/video_6fold_1/seed42"
    write_json(base / "test_semantic/predictions.json", {
        "metrics": {
            "video_count": 78, "video_category_accuracy": 0.28,
            "category_derived_object_exact": 0.28, "object_macro_ap": 0.43,
        },
        "collapse_audit": {"unique_captions": 6, "passed": True},
    })
    decoder_metrics = {
        name: {
            "video_count": 78, "category_accuracy": 0.2,
            "object_exact": 0.2, "invalid_pair_rate": 0.0,
            "unique_predicted_categories": 6,
        }
        for name in (
            "mean_logit", "mean_probability", "majority_vote",
            "object_top2", "valid_pair_object", "hybrid",
        )
    }
    write_json(base / "temporal_decoding/decoding_report.json", {
        "test_metrics": decoder_metrics, "selected_decoder": "valid_pair_object",
        "selected_hybrid_alpha": 0.5,
        "collapse_audit": {"unique_captions": 6, "passed": True},
    })
    rows = temporal_rows(tmp_path, allow_missing=True)
    assert len(rows) == 6
    assert {row["decoder"] for row in rows} == set(decoder_metrics)
    selected = next(row for row in rows if row["decoder"] == "valid_pair_object")
    assert "selected on validation" in selected["selection"]
    mean_logit = next(row for row in rows if row["decoder"] == "mean_logit")
    assert mean_logit["object_macro_ap"] == 0.43


def test_long_form_collector_supports_new_video_aggregation_schema(tmp_path: Path) -> None:
    output = tmp_path / "run"
    write_json(output / "test_semantic/predictions.json", {
        "video_aggregation_records": [{
            "video_id": "01-001", "predicted_category": "01",
            "predicted_objects": ["person", "ball"], "caption": "A caption.",
        }],
    })
    rows = collect_semantic_rows({
        "output_dir": str(output), "method": "temporal_category",
        "variant": "a_1s4_first6", "subject": "chentianlin",
        "fold": "video_6fold_1", "seed": 42,
    }, allow_missing=False)
    values = {row["metric"]: row["value"] for row in rows}
    assert values == {
        "category_correct": 1.0, "object_exact": 1.0, "caption_nonempty": 1.0,
    }
