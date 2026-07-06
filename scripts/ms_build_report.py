"""Build a markdown benchmark report from metrics files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.report import build_report
from ms_video_eval.utils import get_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark report.")
    parser.add_argument("--metrics", type=Path, default=get_repo_root() / "outputs" / "ms_eval" / "metrics")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "configs" / "ms_eval_tasks.yaml",
        help="Optional task YAML for richer report context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_repo_root() / "outputs" / "ms_eval" / "reports" / "ms_eval_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.metrics, tasks_path=args.tasks, output_path=args.output)
    print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()

