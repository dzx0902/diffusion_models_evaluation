"""Encode cached Wan states into fixed autoencoder targets for EEG training."""

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

from ms_video_eval.wan_condition_autoencoder import WanConditionAutoencoder, WanConditionAutoencoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed [slots, latent_dim] Wan autoencoder latents.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = WanConditionAutoencoderConfig(**checkpoint["config"])
    model = WanConditionAutoencoder(config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    rows = [json.loads(line) for line in (args.cache_dir / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as index:
        with torch.inference_mode():
            for row in rows:
                state = torch.load(Path(row["path"]), map_location="cpu", weights_only=True).float()
                tokens = int(state.shape[0])
                if state.shape != (tokens, config.input_dim) or not 1 <= tokens <= config.slots:
                    raise ValueError(f"Invalid cached state for {row['id']}: {tuple(state.shape)}")
                output = args.output_dir / f"{int(row['id']):06d}.pt"
                if not (args.skip_existing and output.exists()):
                    padded = torch.zeros(1, config.slots, config.input_dim, device=device)
                    padded[0, :tokens] = state.to(device)
                    latent = model.encode(padded, torch.tensor([tokens], device=device))[0].cpu().to(torch.float16)
                    torch.save({"latent": latent, "tokens": tokens, "prompt": row["prompt"], "space": "wan_condition_autoencoder"}, output)
                index.write(json.dumps({"id": row["id"], "tokens": tokens, "path": str(output), "prompt": row["prompt"]}, ensure_ascii=False) + "\n")
    print(f"[wan-ae-export] wrote {len(rows)} latents to {args.output_dir}")


if __name__ == "__main__":
    main()
