"""Generate native and exact autoencoder-decoded Wan controls from cached T5 states."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.wan_condition_autoencoder import WanConditionAutoencoder, WanConditionAutoencoderConfig
from ms_video_eval.wan_t5_injection import first_call_context_injector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native and autoencoder-decoded Wan controls for selected videos.")
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--task", default="ti2v-5B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-model", choices=["True", "False"], default="True")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_wan(repo: Path, wan_args: list[str], context: torch.Tensor | None) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from wan.modules.t5 import T5EncoderModel

    original_call = T5EncoderModel.__call__
    if context is not None:
        T5EncoderModel.__call__ = first_call_context_injector(original_call, context)
    old_argv = sys.argv
    try:
        sys.argv = [str(repo / "generate.py"), *wan_args]
        runpy.run_path(str(repo / "generate.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        T5EncoderModel.__call__ = original_call


def main() -> None:
    args = parse_args()
    repo = args.wan_repo.resolve()
    if not (repo / "generate.py").exists():
        raise FileNotFoundError(repo / "generate.py")
    device = torch.device(args.device)
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device, weights_only=False)
    config = WanConditionAutoencoderConfig(**checkpoint["config"])
    autoencoder = WanConditionAutoencoder(config).to(device).eval()
    autoencoder.load_state_dict(checkpoint["state_dict"])
    manifest = {str(row["video_id"]): row for row in read_jsonl(args.manifest)}
    cache = {str(row["prompt"]): Path(row["path"]) for row in read_jsonl(args.cache_dir / "index.jsonl")}
    missing = [video_id for video_id in args.video_ids if video_id not in manifest]
    if missing:
        raise KeyError(f"video_id values absent from manifest: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_id in args.video_ids:
        prompt = str(manifest[video_id]["caption"])
        state_path = cache.get(prompt)
        if state_path is None:
            raise KeyError(f"No cached T5 state for {video_id}: {prompt!r}")
        state = torch.load(state_path, map_location="cpu", weights_only=True).float()
        tokens = state.shape[0]
        padded = torch.zeros(1, config.slots, config.input_dim, device=device)
        padded[0, :tokens] = state.to(device)
        lengths = torch.tensor([tokens], device=device)
        with torch.inference_mode():
            latent, reconstructed = autoencoder(padded, lengths)
        context = reconstructed[0, :tokens]
        mse = float((context.cpu() - state).square().mean().item())
        cosine = float(F.cosine_similarity(context.cpu(), state, dim=-1).mean().item())
        variants = [] if args.skip_native else [("native_text", None)]
        variants.append((f"ae_k{config.latent_dim}", context))
        for label, condition in variants:
            output = args.output_dir / f"{video_id}_{label}_seed{args.seed}.mp4"
            if args.skip_existing and output.exists() and output.stat().st_size > 0:
                print(f"[wan-ae-control] skip existing {output}", flush=True)
                continue
            print(f"[wan-ae-control] video={video_id} variant={label} tokens={tokens} mse={mse:.6e} cosine={cosine:.6f}", flush=True)
            run_wan(repo, ["--task", args.task, "--size", args.size, "--ckpt_dir", str(repo / "Wan2.2-TI2V-5B"), "--offload_model", args.offload_model, "--convert_model_dtype", "--t5_cpu", "--base_seed", str(args.seed), "--prompt", prompt, "--save_file", str(output)], condition)
    print(f"[wan-ae-control] outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
