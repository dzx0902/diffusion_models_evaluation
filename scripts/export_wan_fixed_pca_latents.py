"""Export fixed [slots, dim] PCA-padded latents from cached Wan T5 states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stable fixed-shape Wan PCA latents.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.projector)
    components = torch.from_numpy(payload["components"][: args.dim].astype(np.float32))
    mean = torch.from_numpy(payload["mean"].astype(np.float32))
    records = [json.loads(line) for line in (args.cache_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"

    with index_path.open("w", encoding="utf-8") as index:
        for record in records:
            state = torch.load(Path(record["path"]), map_location="cpu", weights_only=True).float()
            if state.shape[0] > args.slots:
                raise ValueError(f"Prompt {record['id']} has {state.shape[0]} tokens, exceeding slots={args.slots}")
            output_path = args.output_dir / f"{int(record['id']):06d}.pt"
            if not (args.skip_existing and output_path.exists()):
                latent = torch.zeros(args.slots, args.dim, dtype=torch.float16)
                latent[: state.shape[0]] = ((state - mean) @ components.t()).to(torch.float16)
                torch.save({"latent": latent, "tokens": int(state.shape[0]), "prompt": record["prompt"]}, output_path)
            index.write(
                json.dumps(
                    {"id": record["id"], "tokens": int(state.shape[0]), "path": str(output_path), "prompt": record["prompt"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[fixed-pca-export] wrote {len(records)} latents to {args.output_dir}")


if __name__ == "__main__":
    main()
