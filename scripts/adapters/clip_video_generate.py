"""Generate AnimateDiff or ZeroScope video from native text or fixed CLIP states."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.clip_video_pipeline import (
    encode_prompt_with_pipeline,
    load_clip_video_pipeline,
    pipeline_execution_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CLIP-conditioned baseline video.")
    parser.add_argument("--backend", choices=["animatediff", "zeroscope"], required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--motion-adapter", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--condition", type=Path, default=None, help="A .pt payload containing latent [77,D].")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-prompt", default="bad quality, worse quality, distorted, deformed")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Pipeline compute dtype. AnimateDiff may require bfloat16 to avoid dark-frame collapse.",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.prompt is None) == (args.condition is None):
        raise ValueError("Supply exactly one of --prompt or --condition")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype)
    pipe = load_clip_video_pipeline(
        backend=args.backend,
        model_root=args.model_root,
        motion_adapter=args.motion_adapter,
        dtype=dtype,
    )
    pipe.enable_vae_slicing()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    call_args: dict[str, Any] = {
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
    }
    if args.height is not None:
        call_args["height"] = args.height
    if args.width is not None:
        call_args["width"] = args.width
    if args.prompt is not None:
        call_args["prompt"] = args.prompt
        call_args["negative_prompt"] = args.negative_prompt
    else:
        payload = torch.load(args.condition, map_location="cpu", weights_only=True)
        latent = payload["latent"].float()
        expected_dim = int(pipe.text_encoder.config.hidden_size)
        expected_tokens = int(pipe.tokenizer.model_max_length)
        if latent.shape != (expected_tokens, expected_dim):
            raise ValueError(
                f"Condition shape={tuple(latent.shape)}, expected={(expected_tokens, expected_dim)}"
            )
        execution_device = pipeline_execution_device(pipe, device)
        prompt_embeds = latent.unsqueeze(0).to(device=execution_device, dtype=dtype)
        prompt_embeds, negative_prompt_embeds = encode_prompt_with_pipeline(
            pipe,
            prompt=None,
            device=execution_device,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            prompt_embeds=prompt_embeds,
        )
        call_args["prompt_embeds"] = prompt_embeds
        if negative_prompt_embeds is not None:
            call_args["negative_prompt_embeds"] = negative_prompt_embeds

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = pipe(**call_args).frames[0]
    from diffusers.utils import export_to_video

    export_to_video(frames, str(args.output), fps=args.fps)
    print(f"[clip-video] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
