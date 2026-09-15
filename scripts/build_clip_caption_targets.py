"""Encode every audited caption into one normalized CLIP text vector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_schema import load_semantic_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output_dir = args.output_dir.resolve()
    index_path = output_dir / "index.jsonl"
    if index_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {index_path}")
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(args.clip_model, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=True)
    records = load_semantic_records(args.semantic_labels)
    rows = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        inputs = processor(
            text=[record.caption for record in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            vectors = model.get_text_features(**inputs)
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1).cpu()
        for record, vector in zip(batch, vectors):
            path = vectors_dir / f"{record.video_id}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "latent": vector.unsqueeze(0),
                    "tokens": 1,
                    "space": "clip_text_normalized",
                },
                path,
            )
            rows.append({
                "schema_version": 1,
                "video_id": record.video_id,
                "category_id": record.category_id,
                "prompt": record.caption,
                "latent_path": str(path),
            })
        print(f"[clip-caption-targets] {min(start + len(batch), len(records))}/{len(records)}", flush=True)

    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "record_count": len(rows),
        "semantic_labels": str(args.semantic_labels.resolve()),
        "semantic_labels_sha256": sha256(args.semantic_labels),
        "clip_model": str(Path(args.clip_model).resolve()),
        "normalization": "l2",
        "dimension": int(vectors.shape[-1]),
        "index": str(index_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[clip-caption-targets] index={index_path}", flush=True)


if __name__ == "__main__":
    main()
