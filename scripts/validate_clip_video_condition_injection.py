"""Validate native text and injected prompt embeddings in one Diffusers pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
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
    parser = argparse.ArgumentParser(
        description="Compare native prompt encoding with injection of the exact same tensor."
    )
    parser.add_argument("--backend", choices=["animatediff", "zeroscope"], required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--motion-adapter", type=Path, default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="condition_injection")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    return parser.parse_args()


def frame_array(frames: list[Any]) -> np.ndarray:
    return np.stack([np.asarray(frame.convert("RGB"), dtype=np.float32) for frame in frames])


def main() -> None:
    args = parse_args()
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
    execution_device = pipeline_execution_device(pipe, device)

    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = encode_prompt_with_pipeline(
            pipe,
            prompt=args.prompt,
            device=execution_device,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )

    common: dict[str, Any] = {
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
    }
    native_frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        **common,
    ).frames[0]

    injected_args: dict[str, Any] = {
        "prompt_embeds": prompt_embeds.clone(),
        "generator": torch.Generator(device=device).manual_seed(args.seed),
        **common,
    }
    if negative_prompt_embeds is not None:
        injected_args["negative_prompt_embeds"] = negative_prompt_embeds.clone()
    injected_frames = pipe(**injected_args).frames[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_path = args.output_dir / f"{args.name}_native.mp4"
    injected_path = args.output_dir / f"{args.name}_injected.mp4"
    condition_path = args.output_dir / f"{args.name}_condition.pt"
    from diffusers.utils import export_to_video

    export_to_video(native_frames, str(native_path), fps=args.fps)
    export_to_video(injected_frames, str(injected_path), fps=args.fps)
    torch.save(
        {
            "latent": prompt_embeds.squeeze(0).detach().cpu().float(),
            "tokens": int(prompt_embeds.shape[1]),
            "prompt": args.prompt,
            "backend": args.backend,
            "model_root": str(args.model_root.resolve()),
            "embedding_contract": "diffusers.pipeline.encode_prompt.v1",
            "compute_dtype": args.dtype,
            "pipeline_class": type(pipe).__name__,
        },
        condition_path,
    )

    native = frame_array(native_frames)
    injected = frame_array(injected_frames)
    difference = native - injected
    mse = float(np.mean(np.square(difference)))
    metrics = {
        "backend": args.backend,
        "pipeline_class": type(pipe).__name__,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "shape": list(native.shape),
        "native_mean": float(native.mean()),
        "native_std": float(native.std()),
        "injected_mean": float(injected.mean()),
        "injected_std": float(injected.std()),
        "pixel_mae": float(np.mean(np.abs(difference))),
        "pixel_rmse": math.sqrt(mse),
        "pixel_max_abs": float(np.max(np.abs(difference))),
        "psnr_db": None if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse)),
        "native_video": str(native_path),
        "injected_video": str(injected_path),
        "condition": str(condition_path),
    }
    report_path = args.output_dir / f"{args.name}_report.json"
    report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("[clip-injection] " + json.dumps(metrics), flush=True)
    print(f"[clip-injection] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
