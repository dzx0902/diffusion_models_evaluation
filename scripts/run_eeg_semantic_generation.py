"""Run a matched multi-generator EEG semantic ablation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_generation import (
    load_generators,
    read_conditions,
    read_trajectories,
    run_generation_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--condition-kind", choices=("caption", "tora_state"), required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--generators", nargs="+", required=True)
    parser.add_argument("--trajectory-manifest", type=Path, default=None)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_conditions(args.input, args.condition_kind)
    generators = load_generators(args.generator_config)
    trajectories = read_trajectories(args.trajectory_manifest)
    records = run_generation_matrix(
        rows,
        args.condition_kind,
        generators,
        args.generators,
        args.seeds,
        args.output_dir,
        {
            "repo_root": str(ROOT.resolve()),
            "models_root": str(args.models_root.resolve()),
            "home": str(Path.home()),
            "variant": args.variant,
            "subject": args.subject,
            "fold": args.fold,
            "training_seed": args.training_seed,
        },
        trajectories,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "generation_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"[eeg-semantic-generation] jobs={len(records)} manifest={manifest}")


if __name__ == "__main__":
    main()
