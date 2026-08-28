"""Build and validate the immutable video-to-trajectory map used by all methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_data import load_video_partitions


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--fold", default="video_6fold_1")
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--pattern", default="{video_id}.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = sorted(load_video_partitions(args.split_plan, args.fold)[args.partition])
    rows = []
    for video_id in ids:
        pattern = args.pattern.format(video_id=video_id)
        paths = sorted(path for path in args.trajectory_root.glob(pattern) if path.is_file() and path.stat().st_size)
        if not paths:
            raise FileNotFoundError(args.trajectory_root / pattern)
        rows.append({"schema_version": 1, "video_id": video_id,
                     "trajectory_paths": [str(path.resolve()) for path in paths],
                     "sha256s": [sha256(path) for path in paths],
                     "source": "fixed", "partition": args.partition, "fold": args.fold})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"[fixed-trajectories] videos={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
