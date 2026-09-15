"""Remove train-only coarse-category centroids from caption embeddings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_vector(row: dict[str, Any]) -> torch.Tensor:
    payload = torch.load(row["latent_path"], map_location="cpu", weights_only=True)
    latent = payload["latent"].float()
    tokens = int(payload.get("tokens", latent.shape[0]))
    return latent[:tokens].mean(dim=0)


def load_fold(path: Path, fold: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = next((row for row in payload["experiments"] if row["name"] == fold), None)
    if result is None:
        raise KeyError(f"Unknown fold {fold!r}")
    return result


def build_residuals(
    rows: list[dict[str, Any]], train_ids: set[str]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    vectors = {str(row["video_id"]): load_vector(row) for row in rows}
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in train_ids:
            category = str(row.get("category_id") or video_id.split("-", 1)[0])
            grouped[category].append(vectors[video_id])
    categories = {str(row.get("category_id") or str(row["video_id"]).split("-", 1)[0]) for row in rows}
    missing = categories - set(grouped)
    if missing:
        raise ValueError(f"Training split lacks categories: {sorted(missing)}")
    centroids = {
        category: F.normalize(torch.stack(values).mean(dim=0), dim=0)
        for category, values in grouped.items()
    }
    residuals = {}
    for row in rows:
        video_id = str(row["video_id"])
        category = str(row.get("category_id") or video_id.split("-", 1)[0])
        residuals[video_id] = F.normalize(vectors[video_id] - centroids[category], dim=0)
    return residuals, centroids


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    index_path = output_dir / "index.jsonl"
    if index_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {index_path}")
    rows = read_jsonl(args.targets)
    fold = load_fold(args.split_plan, args.fold)
    train_ids = set(map(str, fold["train_video_ids"]))
    known = {str(row["video_id"]) for row in rows}
    if not train_ids <= known:
        raise KeyError(f"Targets miss train videos: {sorted(train_ids - known)[:5]}")
    residuals, centroids = build_residuals(rows, train_ids)
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in rows:
        video_id = str(row["video_id"])
        path = vectors_dir / f"{video_id}.pt"
        torch.save(
            {
                "schema_version": 1,
                "latent": residuals[video_id].unsqueeze(0),
                "tokens": 1,
                "space": "clip_text_category_residual",
            },
            path,
        )
        output_rows.append({
            **row,
            "latent_path": str(path),
            "source_latent_path": str(row["latent_path"]),
            "residual_fold": args.fold,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    torch.save(
        {
            "schema_version": 1,
            "fold": args.fold,
            "fit_partition": "train",
            "train_video_ids": sorted(train_ids),
            "category_centroids": centroids,
        },
        output_dir / "category_centroids.pt",
    )
    metadata = {
        "schema_version": 1,
        "fold": args.fold,
        "fit_partition": "train",
        "train_video_count": len(train_ids),
        "record_count": len(output_rows),
        "categories": sorted(centroids),
        "source_targets": str(args.targets.resolve()),
        "split_plan": str(args.split_plan.resolve()),
        "residual_normalization": "l2",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[category-residual] index={index_path} records={len(output_rows)}", flush=True)


if __name__ == "__main__":
    main()
