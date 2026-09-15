"""Materialize and run warm-started C2 training stages in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_matrix import deep_merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline", type=Path,
        default=ROOT / "configs/eeg_semantic/c2_multistage.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    pipeline = yaml.safe_load(args.pipeline.read_text(encoding="utf-8"))
    base = yaml.safe_load(resolve(pipeline["base_config"]).read_text(encoding="utf-8"))
    config_root = resolve(pipeline["config_output_dir"])
    config_root.mkdir(parents=True, exist_ok=True)
    stages = pipeline["stages"]
    known = {str(stage["id"]) for stage in stages}
    selected = known if args.stages is None else set(args.stages)
    unknown = selected - known
    if unknown:
        raise KeyError(f"Unknown C2 stages: {sorted(unknown)}")
    outputs = {
        str(stage["id"]): resolve(
            deep_merge(base, stage.get("overrides", {}))["experiment"]["output_dir"]
        )
        for stage in stages
    }
    for stage in stages:
        stage_id = str(stage["id"])
        if stage_id not in selected:
            continue
        config = deep_merge(base, stage.get("overrides", {}))
        config_path = config_root / f"{stage_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output_dir = outputs[stage_id]
        if args.skip_existing and (output_dir / "completed.json").is_file():
            print(f"[c2-multistage] skip completed {stage_id}", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts/train_compact_tora_alignment.py"),
            "--config", str(config_path),
            "--device", args.device,
        ]
        init_from = stage.get("init_from")
        if init_from:
            checkpoint = outputs[str(init_from)] / "best.pt"
            if not args.dry_run and not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Stage {stage_id} requires completed {init_from}: {checkpoint}"
                )
            command.extend(["--init-checkpoint", str(checkpoint)])
        print(f"[c2-multistage] {stage_id}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    print("[c2-multistage] completed", flush=True)


if __name__ == "__main__":
    main()
