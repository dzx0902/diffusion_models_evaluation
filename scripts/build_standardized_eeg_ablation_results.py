"""Build protocol-separated JSON/CSV/Markdown summaries for the EEG ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TEMPORAL_VARIANTS = ("a_4s_first6", "a_2s2_first6", "a_1s4_first6")
SEMANTIC_VARIANTS = ("a_base", "a_enhanced", "b_base", "b_enhanced")
TORA_VARIANTS = ("c2_mse", "c2_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eeg_semantic"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/eeg_semantic/reports/standardized_fold1_seed42"),
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def read(path: Path, allow_missing: bool) -> dict[str, Any] | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if allow_missing:
        return None
    raise FileNotFoundError(path)


def semantic_rows(root: Path, allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for variant in SEMANTIC_VARIANTS:
        path = root / "runs_eeg2caption" / variant / (
            "chentianlin/video_6fold_1/seed42/test_semantic/predictions.json"
        )
        data = read(path, allow_missing)
        if data is None:
            continue
        fused = data["video_aggregation_metrics"]["fused"]
        structured = data["video_aggregation_metrics"].get("structured", {}).get(
            "aggregate", {}
        )
        audit = data["collapse_audit"]
        rows.append({
            "scope": "full8_624", "comparable_group": "full8_semantic_caption",
            "family": "EEG2Caption semantic", "variant": variant,
            "decoder": "mean_session_logits", "test_videos": audit["video_count"],
            "primary_metric": "category_accuracy",
            "primary_value": fused["category_accuracy"], "chance": 1 / 8,
            "object_exact": fused["predicted_cardinality_exact_set"],
            "object_macro_ap": fused["macro_ap"],
            "structured_macro_f1": structured.get("macro_f1"),
            "retrieval_top1": None, "retrieval_top5": None,
            "invalid_pair_rate": 0.0, "unique_captions": audit["unique_captions"],
            "collapse_passed": audit["passed"], "selection": "validation checkpoint",
            "primary_analysis": True,
            "source": str(path),
        })
    return rows


def tora_rows(root: Path, allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for variant in TORA_VARIANTS:
        path = root / "runs_compact" / variant / (
            "chentianlin/video_6fold_1/seed42/test_conditions/report.json"
        )
        data = read(path, allow_missing)
        if data is None:
            continue
        rows.append({
            "scope": "full8_624", "comparable_group": "full8_tora_alignment",
            "family": "Compact EEG to Tora PCA", "variant": variant,
            "decoder": "continuous_condition", "test_videos": data["video_count"],
            "primary_metric": "retrieval_top5",
            "primary_value": data["video_retrieval_top5"], "chance": data["chance_top5"],
            "object_exact": None, "object_macro_ap": data.get("object_macro_ap"),
            "structured_macro_f1": None,
            "retrieval_top1": data["video_retrieval_top1"],
            "retrieval_top5": data["video_retrieval_top5"],
            "invalid_pair_rate": None, "unique_captions": None,
            "collapse_passed": None, "selection": "validation checkpoint",
            "primary_analysis": True,
            "source": str(path),
        })
    return rows


def temporal_rows(root: Path, allow_missing: bool) -> list[dict[str, Any]]:
    rows = []
    for variant in TEMPORAL_VARIANTS:
        base = root / "runs_eeg2caption_temporal" / variant / (
            "chentianlin/video_6fold_1/seed42"
        )
        prediction_path = base / "test_semantic/predictions.json"
        data = read(prediction_path, allow_missing)
        decoding_path = base / "temporal_decoding/decoding_report.json"
        report = read(decoding_path, True)
        if data is not None and report is None:
            metrics = data["metrics"]
            audit = data["collapse_audit"]
            rows.append({
                "scope": "first6_468", "comparable_group": "first6_temporal",
                "family": "Temporal EEG2Caption", "variant": variant,
                "decoder": "mean_logit", "test_videos": metrics["video_count"],
                "primary_metric": "category_accuracy",
                "primary_value": metrics["video_category_accuracy"], "chance": 1 / 6,
                "object_exact": metrics["category_derived_object_exact"],
                "object_macro_ap": metrics["object_macro_ap"],
                "structured_macro_f1": None, "retrieval_top1": None,
                "retrieval_top5": None, "invalid_pair_rate": 0.0,
                "unique_captions": audit["unique_captions"],
                "collapse_passed": audit["passed"],
                "selection": "validation checkpoint", "source": str(prediction_path),
                "primary_analysis": True,
            })
        if report is None:
            if not allow_missing:
                raise FileNotFoundError(decoding_path)
            continue
        for decoder, metrics in report["test_metrics"].items():
            rows.append({
                "scope": "first6_468", "comparable_group": "first6_temporal_posthoc",
                "family": "Temporal evidence decoding", "variant": variant,
                "decoder": decoder, "test_videos": metrics["video_count"],
                "primary_metric": "category_accuracy",
                "primary_value": metrics["category_accuracy"], "chance": 1 / 6,
                "object_exact": metrics["object_exact"],
                "object_macro_ap": (
                    data["metrics"]["object_macro_ap"]
                    if data is not None and decoder == "mean_logit" else None
                ),
                "structured_macro_f1": None, "retrieval_top1": None,
                "retrieval_top5": None,
                "invalid_pair_rate": metrics["invalid_pair_rate"],
                "unique_captions": (
                    report["collapse_audit"]["unique_captions"]
                    if decoder == report["selected_decoder"] else None
                ),
                "collapse_passed": (
                    report["collapse_audit"]["passed"]
                    if decoder == report["selected_decoder"] else None
                ),
                "selection": (
                    "decoder selected on validation; "
                    f"hybrid alpha={report['selected_hybrid_alpha']} selected on validation"
                    if decoder == report["selected_decoder"] and decoder == "hybrid"
                    else "decoder selected on validation"
                    if decoder == report["selected_decoder"]
                    else f"hybrid alpha={report['selected_hybrid_alpha']} selected on validation"
                    if decoder == "hybrid"
                    else "fixed decoder"
                ),
                "primary_analysis": decoder == report["selected_decoder"],
                "source": str(decoding_path),
            })
    return rows


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    rows = (
        semantic_rows(args.root, args.allow_missing)
        + tora_rows(args.root, args.allow_missing)
        + temporal_rows(args.root, args.allow_missing)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps({
            "schema_version": 1, "subject": "chentianlin",
            "fold": "video_6fold_1", "training_seed": 42,
            "status": "provisional_single_fold_single_seed",
            "rows": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    report = [
        "# EEG Caption Ablation — Standardized Fold-1 Results", "",
        "Status: **provisional, single subject / fold / seed**. Results from different "
        "protocol scopes are not ranked against each other.", "",
        "| scope | comparable group | variant | decoder | n | primary | value | chance | object exact | invalid pair |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['scope']} | {row['comparable_group']} | {row['variant']} | "
            f"{row['decoder']} | {row['test_videos']} | {row['primary_metric']} | "
            f"{fmt(row['primary_value'])} | {fmt(row['chance'])} | "
            f"{fmt(row['object_exact'])} | {fmt(row['invalid_pair_rate'])} |"
        )
    leaders = {}
    for row in rows:
        if not row["primary_analysis"]:
            continue
        group = row["comparable_group"]
        if group not in leaders or row["primary_value"] > leaders[group]["primary_value"]:
            leaders[group] = row
    report.extend(["", "## Within-scope leaders", ""])
    for group, row in leaders.items():
        report.append(
            f"- `{group}`: `{row['variant']} / {row['decoder']}` = "
            f"{fmt(row['primary_value'])} ({row['primary_metric']}; chance {fmt(row['chance'])})."
        )
    report.extend([
        "", "## Protocol notes", "",
        "- `full8_624`: categories 01--08; 104 held-out test videos in fold 1.",
        "- `first6_468`: categories 01--06 only; 78 held-out test videos in fold 1.",
        "- Temporal decoder and hybrid parameters are selected only on validation; test is evaluation-only.",
        "- C2 retrieval metrics and discrete caption classification metrics have different endpoints and are not directly comparable.",
        "- Formal paper claims require all six folds, multiple seeds, paired tests, and completion of the semantic-label audit.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[standardized-eeg-results] rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
