"""Run the full benchmark pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full multi-subject benchmark pipeline.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "configs" / "ms_eval_models.yaml")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs" / "ms_eval_settings.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mode", type=str, default="t2v", choices=["t2v", "ti2v", "i2v"])
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-generate", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    gen_script = ROOT / "scripts" / "ms_generate.py"
    ext_script = ROOT / "scripts" / "ms_extract_frames.py"
    eval_script = ROOT / "scripts" / "ms_evaluate.py"
    report_script = ROOT / "scripts" / "ms_build_report.py"

    if not args.no_generate:
        command = [
            python,
            str(gen_script),
            "--tasks",
            str(args.tasks),
            "--models",
            str(args.models),
            "--mode",
            args.mode,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.skip_existing:
            command.append("--skip-existing")
        _run(command)

    _run(
        [
            python,
            str(ext_script),
            "--sample-every",
            str(args.sample_every),
        ]
    )
    _run(
        [
            python,
            str(eval_script),
            "--tasks",
            str(args.tasks),
            "--settings",
            str(args.settings),
        ]
    )
    _run([python, str(report_script), "--tasks", str(args.tasks)])


if __name__ == "__main__":
    main()

