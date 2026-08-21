"""Export fixed native CogVideoX prompt states through the official pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-sequence-length", type=int, default=226)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    dtype = resolve_cogvideox_dtype(args.dtype)
    pipe = load_cogvideox_pipeline(args.model_root, dtype)
    pipe.text_encoder.to(device)

    records = read_jsonl(args.manifest)
    prompts = list(dict.fromkeys(str(record["caption"]) for record in records))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = args.output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    if index_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {index_path}")

    rows = []
    for index, prompt in enumerate(prompts):
        with torch.inference_mode():
            embeds, _ = encode_cogvideox_prompt(
                pipe,
                prompt=prompt,
                device=device,
                dtype=dtype,
                guidance_scale=1.0,
                max_sequence_length=args.max_sequence_length,
            )
        latent = embeds.squeeze(0).detach().cpu().float()
        path = (latent_dir / f"{index:04d}.pt").resolve()
        torch.save(
            {
                "latent": latent,
                "tokens": int(latent.shape[0]),
                "prompt": prompt,
                "backend": "cogvideox-2b",
                "embedding_contract": "diffusers.CogVideoXPipeline.encode_prompt.v1",
                "compute_dtype": args.dtype,
            },
            path,
        )
        rows.append(
            {
                "prompt": prompt,
                "path": str(path),
                "tokens": int(latent.shape[0]),
                "shape": list(latent.shape),
                "backend": "cogvideox-2b",
                "embedding_contract": "diffusers.CogVideoXPipeline.encode_prompt.v1",
                "compute_dtype": args.dtype,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(prompts):
            print(f"[cogvideox-targets] {index + 1}/{len(prompts)}", flush=True)

    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "backend": "cogvideox-2b",
        "model_root": str(args.model_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "video_count": len(records),
        "unique_prompt_count": len(prompts),
        "condition_shape": rows[0]["shape"] if rows else None,
        "embedding_contract": "diffusers.CogVideoXPipeline.encode_prompt.v1",
        "compute_dtype": args.dtype,
        "pipeline_class": type(pipe).__name__,
        "diffusers_version": importlib.metadata.version("diffusers"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[cogvideox-targets] wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
