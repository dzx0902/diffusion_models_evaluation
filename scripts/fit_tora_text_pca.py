"""Fit a train-only streaming PCA projector for official Tora text states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_data import load_video_partitions
from ms_video_eval.tora_conditioning import load_tora_condition, read_tora_condition_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--fold", default="video_6fold_1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--batch-vectors", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ids_digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.max_dim < 1 or args.batch_vectors < args.max_dim:
        raise ValueError("batch-vectors must be at least max-dim")
    index = read_tora_condition_index(args.index)
    train_ids = sorted(load_video_partitions(args.split_plan, args.fold)["train"])
    missing = set(train_ids) - set(index)
    if missing:
        raise KeyError(f"Tora cache missing train IDs: {sorted(missing)[:5]}")
    projector = IncrementalPCA(n_components=args.max_dim, batch_size=args.batch_vectors)
    pending: list[np.ndarray] = []
    pending_rows = 0
    fitted_rows = 0

    def fit_values(values: np.ndarray) -> None:
        nonlocal fitted_rows
        projector.partial_fit(values)
        fitted_rows += values.shape[0]
        print(f"[tora-pca] fitted token vectors={fitted_rows}", flush=True)

    for video_id in train_ids:
        state = load_tora_condition(Path(index[video_id]["condition_path"])).hidden_state.numpy()
        pending.append(state)
        pending_rows += state.shape[0]
        # Keep at least one full batch pending, so the final partial_fit always
        # receives >= n_components rows and no tail vectors are discarded.
        while pending_rows >= args.batch_vectors * 2:
            matrix = np.concatenate(pending, axis=0)
            fit_values(matrix[: args.batch_vectors].astype(np.float32, copy=False))
            remainder = matrix[args.batch_vectors:]
            pending = [remainder]
            pending_rows = remainder.shape[0]
    tail = np.concatenate(pending, axis=0).astype(np.float32, copy=False)
    if tail.shape[0] < args.max_dim:
        raise ValueError("Final PCA batch has fewer rows than max-dim")
    fit_values(tail)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mean=projector.mean_.astype(np.float32),
        components=projector.components_.astype(np.float32),
        explained_variance=projector.explained_variance_.astype(np.float32),
        explained_variance_ratio=projector.explained_variance_ratio_.astype(np.float32),
        singular_values=projector.singular_values_.astype(np.float32),
        fitted_token_vectors=np.array([fitted_rows], dtype=np.int64),
    )
    metadata = {
        "schema_version": 1,
        "projector": str(args.output.resolve()),
        "source_index": str(args.index.resolve()),
        "split_plan": str(args.split_plan.resolve()),
        "fold": args.fold,
        "fitted_partition": "train",
        "train_video_count": len(train_ids),
        "train_video_ids_sha256": ids_digest(train_ids),
        "fitted_token_vectors": fitted_rows,
        "raw_dim": int(projector.components_.shape[1]),
        "components": int(projector.components_.shape[0]),
        "explained_variance_ratio_sum": float(projector.explained_variance_ratio_.sum()),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[tora-pca] projector: {args.output}")


if __name__ == "__main__":
    main()
