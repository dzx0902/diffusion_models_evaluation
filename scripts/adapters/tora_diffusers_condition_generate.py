"""Generate Tora Diffusers controls from native text or injected T5 states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.tora_conditioning import load_tora_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tora-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--condition", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--conditioning", choices=("native", "injected"), required=True)
    parser.add_argument("--point-path", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--offload", choices=("model", "sequential", "none"), default="model")
    parser.add_argument("--disable-slicing", action="store_true")
    parser.add_argument("--disable-tiling", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> Path:
    diffusers_root = args.tora_repo.resolve() / "diffusers-version"
    conditioning = getattr(args, "conditioning", "injected")
    prompt = getattr(args, "prompt", None)
    required = [
        diffusers_root / "tora" / "t2v_pipeline.py",
        diffusers_root / "tora" / "traj_utils.py",
        args.model_root.resolve() / "model_index.json",
        *[path.resolve() for path in args.point_path],
    ]
    if conditioning == "injected":
        if args.condition is None:
            raise ValueError("Injected conditioning requires --condition")
        required.append(args.condition.resolve())
    elif not prompt and args.condition is None:
        raise ValueError("Native conditioning requires --prompt or a caption-bearing --condition")
    elif args.condition is not None:
        required.append(args.condition.resolve())
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Tora Diffusers inputs:\n" + "\n".join(map(str, missing)))
    if args.num_frames < 2 or (args.num_frames - 1) % 4:
        raise ValueError("num-frames must satisfy (num_frames - 1) % 4 == 0")
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by 8")
    return diffusers_root


def load_control(path: Path) -> tuple[str, torch.Tensor, dict[str, Any]]:
    condition = load_tora_condition(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = {
        key: payload[key]
        for key in ("video_id", "control", "pca_dim", "projector")
        if key in payload
    }
    return condition.caption, condition.hidden_state, metadata


def export_video(frames: list[Any], output: Path, fps: int) -> None:
    import imageio.v2 as imageio
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Tora Diffusers generation requires CUDA")
    diffusers_root = validate_paths(args)
    sys.path.insert(0, str(diffusers_root))

    from diffusers import CogVideoXDPMScheduler
    from tora.t2v_pipeline import ToraPipeline
    from tora.traj_utils import process_traj
    from torchvision.utils import flow_to_image

    if args.condition is not None:
        caption, hidden_state, condition_metadata = load_control(args.condition.resolve())
    else:
        caption = str(args.prompt)
        hidden_state = None
        condition_metadata = {}
    dtype = getattr(torch, args.dtype)
    pipeline = ToraPipeline.from_pretrained(args.model_root.resolve(), torch_dtype=dtype)
    pipeline.scheduler = CogVideoXDPMScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    if args.offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif args.offload == "model":
        pipeline.enable_model_cpu_offload()
    else:
        assert hidden_state is not None
        pipeline.to("cuda")
    if not args.disable_slicing:
        pipeline.vae.enable_slicing()
    if not args.disable_tiling:
        pipeline.vae.enable_tiling()

    raw_flow, _ = process_traj(
        [str(path.resolve()) for path in args.point_path],
        args.num_frames,
        (args.height, args.width),
        device="cpu",
    )
    flow_channels_first = raw_flow.permute(0, 3, 1, 2)
    video_flow = flow_to_image(flow_channels_first).unsqueeze(0).to("cuda", dtype)
    video_flow = video_flow / (255.0 / 2.0) - 1.0

    generation: dict[str, Any] = {
        "video_flow": video_flow,
        "num_videos_per_prompt": 1,
        "num_inference_steps": args.steps,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "use_dynamic_cfg": True,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "max_sequence_length": 226,
    }
    if args.conditioning == "native":
        generation.update(prompt=caption, negative_prompt=args.negative_prompt)
    else:
        device = pipeline._execution_device
        prompt_embeds = hidden_state.unsqueeze(0).to(device=device, dtype=dtype)
        negative_embeds = pipeline._get_t5_prompt_embeds(
            prompt=args.negative_prompt,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device=device,
            dtype=dtype,
        )
        generation.update(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
        )

    result = pipeline(**generation)
    export_video(result.frames[0], args.output.resolve(), args.fps)
    report = {
        "schema_version": 1,
        "backend": "official_tora_diffusers",
        "conditioning": args.conditioning,
        "condition": None if args.condition is None else str(args.condition.resolve()),
        "condition_shape": None if hidden_state is None else list(hidden_state.shape),
        "caption": caption,
        "condition_metadata": condition_metadata,
        "point_paths": [str(path.resolve()) for path in args.point_path],
        "model_root": str(args.model_root.resolve()),
        "output": str(args.output.resolve()),
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "dtype": args.dtype,
        "offload": args.offload,
    }
    report_path = args.report.resolve() if args.report else args.output.resolve().with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tora-diffusers] video: {args.output.resolve()}")
    print(f"[tora-diffusers] report: {report_path}")


if __name__ == "__main__":
    main()
