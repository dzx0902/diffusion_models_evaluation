"""Generate videos for multi-subject benchmark tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.model_runner import load_models, run_generation_benchmark
from ms_video_eval.task_schema import load_tasks
from ms_video_eval.utils import get_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-subject video generation benchmark.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "configs" / "ms_eval_models.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mode", type=str, default="t2v", choices=["t2v", "ti2v", "i2v"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-root", type=Path, default=get_repo_root() / "outputs" / "ms_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.tasks)
    models = load_models(args.models)
    run_generation_benchmark(
        tasks=tasks,
        models=models,
        seeds=args.seeds,
        mode=args.mode,
        output_root=args.output_root,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()

