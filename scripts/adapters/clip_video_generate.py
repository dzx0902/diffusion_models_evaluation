"""Generate AnimateDiff or ZeroScope video from native text or fixed CLIP states."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


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
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    return parser.parse_args()


def encode_negative(pipe: Any, prompt: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = None
    if bool(getattr(pipe.text_encoder.config, "use_attention_mask", False)):
        attention_mask = inputs.attention_mask.to(device)
    with torch.inference_mode():
        return pipe.text_encoder(
            inputs.input_ids.to(device),
            attention_mask=attention_mask,
        )[0].to(dtype=dtype)


def load_pipeline(args: argparse.Namespace, dtype: torch.dtype) -> Any:
    if args.backend == "animatediff":
        if args.motion_adapter is None:
            raise ValueError("--motion-adapter is required for AnimateDiff")
        from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

        adapter = MotionAdapter.from_pretrained(
            args.motion_adapter,
            torch_dtype=dtype,
            variant="fp16",
            local_files_only=True,
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            args.model_root,
            motion_adapter=adapter,
            torch_dtype=dtype,
            variant="fp16",
            safety_checker=None,
            feature_extractor=None,
            local_files_only=True,
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        return pipe

    from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_root,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


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
    dtype = torch.float16
    pipe = load_pipeline(args, dtype)
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
        execution_device = getattr(pipe, "_execution_device", device)
        prompt_embeds = latent.unsqueeze(0).to(device=execution_device, dtype=dtype)
        call_args["prompt_embeds"] = prompt_embeds
        call_args["negative_prompt_embeds"] = encode_negative(
            pipe,
            args.negative_prompt,
            execution_device,
            dtype,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = pipe(**call_args).frames[0]
    from diffusers.utils import export_to_video

    export_to_video(frames, str(args.output), fps=args.fps)
    print(f"[clip-video] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
