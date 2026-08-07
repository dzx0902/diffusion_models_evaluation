"""Export fixed native CLIP text states for CLIP-conditioned video models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed 77-token CLIP video conditions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["animatediff", "zeroscope"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_prompt(tokenizer: Any, text_encoder: Any, prompt: str) -> tuple[torch.Tensor, int]:
    max_length = int(tokenizer.model_max_length)
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = None
    if bool(getattr(text_encoder.config, "use_attention_mask", False)):
        attention_mask = inputs.attention_mask
    with torch.inference_mode():
        latent = text_encoder(inputs.input_ids, attention_mask=attention_mask)[0].squeeze(0).float()
    return latent, int(inputs.attention_mask.sum().item())


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer, CLIPTextModel

    text_encoder_dir = args.model_root / "text_encoder"
    tokenizer_dir = args.model_root / "tokenizer"
    if not text_encoder_dir.is_dir() or not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Expected text_encoder/ and tokenizer/ under {args.model_root}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    load_options: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch.float32}
    if args.backend == "animatediff":
        load_options["variant"] = "fp16"
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir, **load_options).eval()
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
        latent, semantic_tokens = encode_prompt(tokenizer, text_encoder, prompt)
        path = (latent_dir / f"{index:04d}.pt").resolve()
        payload = {
            "latent": latent,
            "tokens": int(latent.shape[0]),
            "semantic_tokens": semantic_tokens,
            "prompt": prompt,
            "backend": args.backend,
            "model_root": str(args.model_root.resolve()),
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
        "fixed_tokens": int(tokenizer.model_max_length),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[clip-targets] wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
