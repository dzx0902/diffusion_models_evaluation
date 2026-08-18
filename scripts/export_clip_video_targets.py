"""Export fixed native CLIP text states for CLIP-conditioned video models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

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
    parser = argparse.ArgumentParser(description="Export fixed 77-token CLIP video conditions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["animatediff", "zeroscope"], required=True)
    parser.add_argument("--motion-adapter", type=Path, default=None)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def semantic_token_count(pipe: Any, prompt: str) -> int:
    inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return int(inputs.attention_mask.sum().item())


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    pipe = load_clip_video_pipeline(
        backend=args.backend,
        model_root=args.model_root,
        motion_adapter=args.motion_adapter,
        dtype=dtype,
    )
    pipe.to(device)
    execution_device = pipeline_execution_device(pipe, device)
    records = read_jsonl(args.manifest)
    prompts = list(dict.fromkeys(str(record["caption"]) for record in records))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = args.output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    if index_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {index_path}")

    index_rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        with torch.inference_mode():
            prompt_embeds, _ = encode_prompt_with_pipeline(
                pipe,
                prompt=prompt,
                device=execution_device,
                guidance_scale=1.0,
            )
        latent = prompt_embeds.squeeze(0).detach().cpu().float()
        semantic_tokens = semantic_token_count(pipe, prompt)
        path = (latent_dir / f"{index:04d}.pt").resolve()
        payload = {
            "latent": latent,
            "tokens": int(latent.shape[0]),
            "semantic_tokens": semantic_tokens,
            "prompt": prompt,
            "backend": args.backend,
            "model_root": str(args.model_root.resolve()),
            "embedding_contract": "diffusers.pipeline.encode_prompt.v1",
            "compute_dtype": args.dtype,
            "pipeline_class": type(pipe).__name__,
        }
        torch.save(payload, path)
        index_rows.append(
            {
                "prompt": prompt,
                "path": str(path),
                "tokens": int(latent.shape[0]),
                "semantic_tokens": semantic_tokens,
                "shape": list(latent.shape),
                "backend": args.backend,
                "embedding_contract": "diffusers.pipeline.encode_prompt.v1",
                "compute_dtype": args.dtype,
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(prompts):
            print(f"[clip-targets] {index + 1}/{len(prompts)}", flush=True)

    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "backend": args.backend,
        "model_root": str(args.model_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "video_count": len(records),
        "unique_prompt_count": len(prompts),
        "condition_shape": index_rows[0]["shape"] if index_rows else None,
        "fixed_tokens": int(pipe.tokenizer.model_max_length),
        "embedding_contract": "diffusers.pipeline.encode_prompt.v1",
        "compute_dtype": args.dtype,
        "pipeline_class": type(pipe).__name__,
        "diffusers_version": importlib.metadata.version("diffusers"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[clip-targets] wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
