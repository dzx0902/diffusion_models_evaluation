"""Average multiple offline Tora caption states into per-video semantic centroids."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.tora_conditioning import load_tora_condition, read_tora_condition_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", choices=("raw_mean", "unit_rescaled"), default="unit_rescaled")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def semantic_centroid(states: list[torch.Tensor], normalization: str) -> torch.Tensor:
    stacked = torch.stack([state.float() for state in states])
    if normalization == "raw_mean":
        return stacked.mean(dim=0)
    norms = stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    direction = F.normalize((stacked / norms).mean(dim=0), dim=-1)
    return direction * norms.mean(dim=0)


def main() -> None:
    args = parse_args()
    indices = [read_tora_condition_index(path) for path in args.indices]
    ids = set(indices[0])
    if any(set(index) != ids for index in indices[1:]):
        raise ValueError("Every paraphrase cache must contain the same video IDs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    states_dir = args.output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id in sorted(ids):
        conditions = [load_tora_condition(Path(index[video_id]["condition_path"])) for index in indices]
        state = semantic_centroid([condition.hidden_state for condition in conditions], args.normalization)
        output = states_dir / f"{video_id}.pt"
        temporary = output.with_suffix(".pt.tmp")
        torch.save(
            {"schema_version": 1, "video_id": video_id, "caption": conditions[0].caption,
             "hidden_state": state, "centroid_sources": [str(path.resolve()) for path in args.indices],
             "normalization": args.normalization},
            temporary,
        )
        os.replace(temporary, output)
        rows.append({"video_id": video_id, "condition_path": str(output.resolve())})
    (args.output_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "record_count": len(rows), "normalization": args.normalization,
                    "source_indices": [str(path.resolve()) for path in args.indices],
                    "cross_video_statistics_used": False}, indent=2),
        encoding="utf-8",
    )
    print(f"[tora-caption-centroids] videos={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
