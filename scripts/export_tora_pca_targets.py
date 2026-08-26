"""Project cached Tora text states into a fixed PCA semantic bottleneck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.tora_conditioning import load_tora_condition, read_tora_condition_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = np.load(args.projector, allow_pickle=False)
    mean = torch.from_numpy(package["mean"]).float()
    components = torch.from_numpy(package["components"]).float()
    if not 1 <= args.dim <= components.shape[0]:
        raise ValueError(f"dim must be in [1, {components.shape[0]}]")
    components = components[: args.dim]
    index = read_tora_condition_index(args.index)
    output_index = args.output_dir / "index.jsonl"
    if output_index.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for count, (video_id, source) in enumerate(sorted(index.items()), 1):
        condition = load_tora_condition(Path(source["condition_path"]))
        latent = (condition.hidden_state - mean) @ components.t()
        output = args.output_dir / f"{video_id}.pt"
        torch.save(
            {
                "schema_version": 1,
                "video_id": video_id,
                "caption": condition.caption,
                "latent": latent,
                "tokens": latent.shape[0],
                "projector": str(args.projector.resolve()),
                "pca_dim": args.dim,
            },
            output,
        )
        rows.append(
            {
                "video_id": video_id,
                "caption": condition.caption,
                "latent_path": str(output.resolve()),
                "tokens": int(latent.shape[0]),
                "shape": list(latent.shape),
                "projector": str(args.projector.resolve()),
                "pca_dim": args.dim,
            }
        )
        if count % 25 == 0:
            print(f"[tora-pca-export] {count}/{len(index)}", flush=True)
    output_index.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"[tora-pca-export] index: {output_index}")


if __name__ == "__main__":
    main()
