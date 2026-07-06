"""Adapter for running ByteDance ContentV with benchmark-style CLI args.

The official ContentV demo keeps prompt and output path in the script body.
This adapter exposes them as command-line arguments so `ms_generate.py` can
call ContentV through `command_template` like the other models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a ContentV video.")
    parser.add_argument("--repo", type=Path, required=True, help="Path to the cloned bytedance/ContentV repo.")
    parser.add_argument("--model-id", type=str, default="ByteDance/ContentV-8B")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=125)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="overexposed, low quality, deformation, poor composition, visual artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"ContentV repo does not exist: {repo}")
    sys.path.insert(0, str(repo))

    from contentv_pipeline import ContentVPipeline
    from contentv_transformer import SD3Transformer3DModel

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = SD3Transformer3DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = ContentVPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    video = pipe(
        num_frames=args.num_frames,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    export_to_video(video, str(args.output), fps=args.fps)


if __name__ == "__main__":
    main()

