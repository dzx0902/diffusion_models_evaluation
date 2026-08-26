"""Export exact GT Tora→PCA→Tora conditions for generation controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.tora_conditioning import (
    ToraPCAProjector,
    load_tora_condition,
    read_tora_condition_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = read_tora_condition_index(args.index)
    projector = ToraPCAProjector.load(args.projector, args.dim)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id in args.video_ids:
        condition = load_tora_condition(Path(index[video_id]["condition_path"]))
        restored = projector.decode(projector.encode(condition.hidden_state))
        path = args.output_dir / f"{video_id}_pca{args.dim}.pt"
        torch.save(
            {
                "schema_version": 1,
                "video_id": video_id,
                "caption": condition.caption,
                "hidden_state": restored,
                "control": "gt_tora_pca_roundtrip",
                "pca_dim": args.dim,
                "projector": str(args.projector.resolve()),
            },
            path,
        )
        rows.append({"video_id": video_id, "condition_path": str(path.resolve())})
    (args.output_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"[tora-pca-roundtrip] wrote {len(rows)} conditions to {args.output_dir}")


if __name__ == "__main__":
    main()
