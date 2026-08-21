"""Generate a small native-text capability screen for CLIP video backends."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.clip_video_pipeline import load_clip_video_pipeline, resolve_dtype


DEFAULT_VIDEO_IDS = (
    "01-001",
    "02-040",
    "03-002",
    "04-041",
    "05-064",
    "06-069",
    "07-031",
    "08-060",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("animatediff", "zeroscope"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--motion-adapter", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", default=list(DEFAULT_VIDEO_IDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--negative-prompt", default="bad quality, worse quality, distorted, deformed")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_captions(path: Path, wanted: list[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row["video_id"])
        if video_id in wanted:
            if video_id in rows:
                raise ValueError(f"Duplicate video_id in manifest: {video_id}")
            rows[video_id] = str(row["caption"])
    missing = sorted(set(wanted) - set(rows))
    if missing:
        raise ValueError(f"Missing requested video IDs: {missing}")
    return rows


def backend_defaults(backend: str) -> tuple[str, int, int]:
    if backend == "animatediff":
        return "bfloat16", 512, 512
    return "float16", 320, 576


def frames_to_uint8(frames: list[Any]) -> np.ndarray:
    """Normalize PIL, uint8, or Diffusers float frames for diagnostics."""
    converted = []
    for frame in frames:
        pixels = np.asarray(frame)
        if np.issubdtype(pixels.dtype, np.floating):
            if pixels.size and float(pixels.max()) <= 1.5:
                pixels = pixels * 255.0
            pixels = np.rint(pixels)
        converted.append(np.clip(pixels, 0, 255).astype(np.uint8))
    return np.stack(converted)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.backend == "animatediff" and args.motion_adapter is None:
        raise ValueError("--motion-adapter is required for AnimateDiff")
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    default_dtype, default_height, default_width = backend_defaults(args.backend)
    dtype_name = args.dtype or default_dtype
    height = args.height or default_height
    width = args.width or default_width
    captions = load_captions(args.manifest, args.video_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "generation_metrics.jsonl"

    pipe = load_clip_video_pipeline(
        backend=args.backend,
        model_root=args.model_root,
        motion_adapter=args.motion_adapter,
        dtype=resolve_dtype(dtype_name),
    )
    pipe.enable_vae_slicing()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    from diffusers.utils import export_to_video

    for video_id in args.video_ids:
        for seed in args.seeds:
            output = args.output_dir / f"{video_id}_native_seed{seed}.mp4"
            if args.skip_existing and output.is_file() and output.stat().st_size > 0:
                print(f"[native-screen] skip {output}", flush=True)
                continue
            print(f"[native-screen] {args.backend} {video_id} seed={seed}: {captions[video_id]}", flush=True)
            started = time.perf_counter()
            frames = pipe(
                prompt=captions[video_id],
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).frames[0]
            elapsed = time.perf_counter() - started
            export_to_video(frames, str(output), fps=args.fps)
            pixels = frames_to_uint8(frames)
            record: dict[str, Any] = {
                "backend": args.backend,
                "video_id": video_id,
                "caption": captions[video_id],
                "seed": seed,
                "dtype": dtype_name,
                "height": height,
                "width": width,
                "num_frames": args.num_frames,
                "fps": args.fps,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "elapsed_sec": elapsed,
                "pixel_min": int(pixels.min()),
                "pixel_max": int(pixels.max()),
                "pixel_mean": float(pixels.mean()),
                "pixel_std": float(pixels.std()),
                "output": str(output.resolve()),
            }
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[native-screen] wrote {output} elapsed={elapsed:.2f}s "
                f"mean={record['pixel_mean']:.2f} std={record['pixel_std']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
