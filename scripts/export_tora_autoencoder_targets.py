"""Encode all Tora text states with a frozen train-only nonlinear bottleneck."""

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

from ms_video_eval.tora_conditioning import load_tora_condition, read_tora_condition_index
from ms_video_eval.tora_text_autoencoder import ToraTextAutoencoder, ToraTextAutoencoderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get("fitted_partition") != "train":
        raise ValueError("Autoencoder checkpoint does not declare train-only fitting")
    model = ToraTextAutoencoder(ToraTextAutoencoderConfig(**checkpoint["config"])).to(args.device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    output_index = args.output_dir / "index.jsonl"
    if output_index.exists() and not args.overwrite:
        raise FileExistsError(output_index)
    args.output_dir.mkdir(parents=True, exist_ok=True); rows = []
    with torch.inference_mode():
        for video_id, row in sorted(read_tora_condition_index(args.index).items()):
            condition = load_tora_condition(Path(row["condition_path"]))
            latent = model.encode(condition.hidden_state.to(args.device)).cpu()
            path = args.output_dir / f"{video_id}.pt"
            torch.save({"schema_version": 1, "video_id": video_id, "caption": condition.caption,
                        "latent": latent, "autoencoder_checkpoint": str(args.checkpoint.resolve())}, path)
            rows.append({"video_id": video_id, "caption": condition.caption,
                         "latent_path": str(path.resolve()), "shape": list(latent.shape),
                         "autoencoder_checkpoint": str(args.checkpoint.resolve())})
    output_index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"[tora-autoencoder-export] videos={len(rows)} output={output_index}")


if __name__ == "__main__":
    main()
