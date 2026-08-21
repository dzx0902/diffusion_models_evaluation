"""Run and summarize a common-protocol screen of classic EEG classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("mlp", "eegnet", "shallownet", "deepnet", "tsconv", "conformer", "multiscale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def run_models(args: argparse.Namespace) -> None:
    for model in args.models:
        output = args.output_root / model
        report = output / "report.json"
        if args.skip_existing and report.exists():
            print(f"[eeg-screen] skip {model}: {report}", flush=True)
            continue
        command = [
            args.python,
            str(ROOT / "scripts/train_eeg_category_probe.py"),
            "--trials", str(args.trials),
            "--split-plan", str(args.split_plan),
            "--experiment", args.experiment,
            "--duration-sec", str(args.duration_sec),
            "--model", model,
            "--output-dir", str(output),
            "--epochs", str(args.epochs),
            "--min-epochs", str(args.min_epochs),
            "--early-stop-patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
        ]
        last_checkpoint = output / "last.pt"
        if last_checkpoint.exists():
            command.extend(("--resume", str(last_checkpoint)))
        print("[eeg-screen] " + " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


def summarize(args: argparse.Namespace) -> None:
    rows = []
    for model in args.models:
        path = args.output_root / model / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": model,
            "checkpoint_epoch": report["checkpoint_epoch"],
            "parameters": report.get("parameter_count", ""),
            "validation_accuracy": report["validation"]["accuracy"],
            "validation_macro_accuracy": report["validation"]["macro_accuracy"],
            "test_accuracy": report["test"]["accuracy"],
            "test_macro_accuracy": report["test"]["macro_accuracy"],
        })
    rows.sort(key=lambda row: row["test_macro_accuracy"], reverse=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model"]
    with (args.output_root / "model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# EEG Category Model Screen",
        "",
        "| model | epoch | parameters | validation macro | test accuracy | test macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['checkpoint_epoch']} | {row['parameters']} | "
            f"{row['validation_macro_accuracy']:.4f} | {row['test_accuracy']:.4f} | "
            f"{row['test_macro_accuracy']:.4f} |"
        )
    (args.output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[eeg-screen] wrote {args.output_root / 'report.md'}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        run_models(args)
    summarize(args)


if __name__ == "__main__":
    main()
