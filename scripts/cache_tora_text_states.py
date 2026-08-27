"""Cache official Tora-compatible T5 cross-attention states on the GPU server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import T5EncoderModel, T5Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_schema import normalize_video_id
from ms_video_eval.tora_conditioning import TORA_TEXT_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions",
        type=Path,
        default=ROOT / "outputs" / "semantic_labels" / "eeg_semantic_labels_v1.jsonl",
    )
    parser.add_argument("--t5-model", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "tora" / "text_cache"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="T5 inference dtype; bfloat16 keeps T5-XXL within a 48 GB GPU.",
    )
    parser.add_argument("--save-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_t5_model(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    tora_root = os.environ.get("TORA_ROOT")
    if not tora_root:
        raise ValueError("Pass --t5-model or set TORA_ROOT to the official Tora repository")
    return (Path(tora_root) / "sat" / "ckpts" / "t5-v1_1-xxl").resolve()


def load_captions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source: dict[str, Any] = json.loads(line)
        video_id = normalize_video_id(source["video_id"])
        if video_id in seen:
            raise ValueError(f"Duplicate caption video_id: {video_id}")
        seen.add(video_id)
        rows.append({"video_id": video_id, "caption": str(source["caption"]).strip()})
    if not rows:
        raise ValueError("Caption source is empty")
    return rows


def batches(values: list[dict[str, str]], size: int):  # type: ignore[no-untyped-def]
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model_path = resolve_t5_model(args.t5_model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    rows = load_captions(args.captions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    if (index_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("Tora cache exists; pass --overwrite or choose a new output directory")
    device = torch.device(args.device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model_dtype = getattr(torch, args.model_dtype)
    model = T5EncoderModel.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    save_dtype = getattr(torch, args.save_dtype)
    index_rows = []
    with torch.inference_mode():
        for group in batches(rows, args.batch_size):
            captions = [row["caption"] for row in group]
            encoded = tokenizer(
                captions,
                truncation=True,
                max_length=TORA_TEXT_TOKENS,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            # Matches official Tora FrozenT5Embedder: no attention_mask is passed.
            hidden = model(input_ids=input_ids).last_hidden_state
            for index, row in enumerate(group):
                output_path = args.output_dir / f"{row['video_id']}.pt"
                torch.save(
                    {
                        "schema_version": 1,
                        "video_id": row["video_id"],
                        "caption": row["caption"],
                        "hidden_state": hidden[index].to(save_dtype).cpu(),
                        "input_ids": encoded["input_ids"][index].cpu(),
                        "attention_mask": encoded["attention_mask"][index].cpu(),
                        "attention_mask_consumed_by_tora_reference": False,
                    },
                    output_path,
                )
                index_rows.append(
                    {
                        "video_id": row["video_id"],
                        "caption": row["caption"],
                        "condition_path": str(output_path.resolve()),
                        "shape": list(hidden[index].shape),
                        "dtype": args.save_dtype,
                    }
                )
            print(f"[tora-text] cached {len(index_rows)}/{len(rows)}", flush=True)
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "source": str(args.captions.resolve()),
        "t5_model": str(model_path),
        "max_length": TORA_TEXT_TOKENS,
        "hidden_dim": int(index_rows[0]["shape"][1]),
        "model_dtype": args.model_dtype,
        "save_dtype": args.save_dtype,
        "attention_mask_consumed_by_tora_reference": False,
        "reference": "alibaba/Tora sat/sgm/modules/encoders/modules.py FrozenT5Embedder",
        "record_count": len(index_rows),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tora-text] index: {index_path}")


if __name__ == "__main__":
    main()
