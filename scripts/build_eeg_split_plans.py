"""Create subject-dependent, category-stratified six-fold EEG video splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 6-fold grouped EEG/video split plans.")
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "eeg_wan" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def captions(records: list[dict], ids: set[str]) -> list[dict]:
    return [{"video_id": row["video_id"], "caption": row["caption"]} for row in records if row["video_id"] in ids]


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.video_manifest)
    if len(records) != len({row["video_id"] for row in records}):
        raise ValueError("Video manifest has duplicate video_id records")
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_category[row["category_id"]].append(row["video_id"])
    if not by_category or any(len(ids) != 78 for ids in by_category.values()):
        raise ValueError("Six-fold protocol requires exactly 78 videos in every category")
    sessions = sorted({item["session"] for row in records for item in row["sessions"]})
    if sessions != ["session1", "session2", "session3"]:
        raise ValueError(f"Expected raw session1..3 only, got {sessions}")

    rng = random.Random(args.seed)
    category_folds: dict[str, list[list[str]]] = {}
    for category, ids in sorted(by_category.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        category_folds[category] = [ids[index * 13 : (index + 1) * 13] for index in range(6)]

    experiments: list[dict] = []
    for fold in range(6):
        validation_fold = (fold + 1) % 6
        test_ids = {video_id for groups in category_folds.values() for video_id in groups[fold]}
        validation_ids = {video_id for groups in category_folds.values() for video_id in groups[validation_fold]}
        train_ids = {
            video_id
            for groups in category_folds.values()
            for group_index, group in enumerate(groups)
            if group_index not in {fold, validation_fold}
            for video_id in group
        }
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise RuntimeError("Video split overlap")
        experiments.append(
            {
                "name": f"video_6fold_{fold + 1}",
                "protocol": "subject_dependent_video_held_out",
                "fold": fold + 1,
                "train_sessions": sessions,
                "validation_sessions": sessions,
                "test_sessions": sessions,
                "train_video_ids": sorted(train_ids),
                "validation_video_ids": sorted(validation_ids),
                "test_video_ids": sorted(test_ids),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / f"{args.subject}_video_6fold_plan.json"
    if plan_path.exists() and not args.overwrite:
        raise FileExistsError(f"Pass --overwrite to replace {plan_path}")
    plan_path.write_text(
        json.dumps(
            {
                "subject": args.subject,
                "seed": args.seed,
                "folds": 6,
                "videos_per_category_per_partition": 13,
                "source_manifest": str(args.video_manifest),
                "experiments": experiments,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    captions_dir = args.output_dir / f"{args.subject}_video_6fold_captions"
    captions_dir.mkdir(exist_ok=True)
    for experiment in experiments:
        for partition in ("train", "validation", "test"):
            ids = set(experiment[f"{partition}_video_ids"])
            write_jsonl(captions_dir / f"{experiment['name']}_{partition}.jsonl", captions(records, ids))
    print(f"[eeg-splits] wrote {plan_path}")
    print("[eeg-splits] 6 folds; each fold train=416, validation=104, test=104 videos")


if __name__ == "__main__":
    main()
