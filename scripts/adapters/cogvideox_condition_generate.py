"""Generate CogVideoX-2B video from native text or a serialized condition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.cogvideox_pipeline import (
    encode_cogvideox_prompt,
    load_cogvideox_pipeline,
    resolve_cogvideox_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--condition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--max-sequence-length", type=int, default=226)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    dtype = resolve_cogvideox_dtype(args.dtype)
    pipe = load_cogvideox_pipeline(args.model_root, dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    call_args = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
    }
    if args.prompt is not None:
        call_args["prompt"] = args.prompt
        call_args["negative_prompt"] = args.negative_prompt
    else:
        payload = torch.load(args.condition, map_location="cpu", weights_only=True)
        latent = payload["latent"].float()
        expected = (args.max_sequence_length, int(pipe.text_encoder.config.d_model))
        if tuple(latent.shape) != expected:
            raise ValueError(f"Condition shape={tuple(latent.shape)}, expected={expected}")
        execution_device = torch.device(getattr(pipe, "_execution_device", device))
        positive, negative = encode_cogvideox_prompt(
            pipe,
            prompt=None,
            device=execution_device,
            dtype=dtype,
            guidance_scale=args.guidance_scale,
            prompt_embeds=latent.unsqueeze(0).to(device=execution_device, dtype=dtype),
            negative_prompt=args.negative_prompt,
            max_sequence_length=args.max_sequence_length,
        )
        call_args["prompt_embeds"] = positive
        if negative is not None:
            call_args["negative_prompt_embeds"] = negative

    frames = pipe(**call_args).frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    export_to_video(frames, str(args.output), fps=args.fps)
    print(f"[cogvideox-condition] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
