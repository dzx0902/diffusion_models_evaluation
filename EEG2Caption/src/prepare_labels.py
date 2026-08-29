"""Build ordered six-dimensional labels for the 468 two-object videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/userhome2/zhoutianyi/Dataset/Multi-Object")
OBJECT_NAMES = ("person", "dog", "car", "ball", "flower", "bird")
PAIR_OBJECTS = {
    "01": ("person", "ball"),
    "02": ("dog", "ball"),
    "03": ("dog", "car"),
    "04": ("car", "flower"),
    "05": ("flower", "bird"),
    "06": ("person", "bird"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


def normalize_id(value: object) -> str:
    return Path(str(value).strip()).stem.replace("-", "_")


def expected_ids() -> List[str]:
    return [
        "{}_{:03d}".format(prefix, trial)
        for prefix in PAIR_OBJECTS
        for trial in range(1, 79)
    ]


def load_caption_rows(path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"id", "caption"}.issubset(rows[0]):
        raise ValueError("Caption CSV must contain id and caption columns.")
    captions = {normalize_id(row["id"]): row["caption"].strip() for row in rows}
    missing = [video_id for video_id in expected_ids() if video_id not in captions]
    if missing:
        raise ValueError("Caption CSV is missing IDs such as {}.".format(missing[:5]))
    return captions, rows


def main() -> None:
    args = parse_args()
    captions, _ = load_caption_rows(
        args.dataset_root / "Videos" / "structured_v2_captions.csv"
    )
    ids = expected_ids()
    labels = torch.zeros(len(ids), len(OBJECT_NAMES), dtype=torch.float32)
    pair_indices = torch.empty(len(ids), dtype=torch.long)
    cardinalities = torch.full((len(ids),), 2, dtype=torch.long)
    records = []
    prefixes = list(PAIR_OBJECTS)
    for row_index, video_id in enumerate(ids):
        prefix = video_id.split("_", 1)[0]
        objects = PAIR_OBJECTS[prefix]
        object_indices = [OBJECT_NAMES.index(name) for name in objects]
        labels[row_index, object_indices] = 1.0
        pair_indices[row_index] = prefixes.index(prefix)
        records.append({
            "row_index": row_index,
            "video_id": video_id,
            "pair_index": prefixes.index(prefix),
            "pair_prefix": prefix,
            "pair_name": "_".join(objects),
            "label": [int(value) for value in labels[row_index].tolist()],
            "label_names": list(objects),
            "ground_truth_caption": captions[video_id],
        })
    if len(ids) != 468 or not torch.equal(labels.sum(1).long(), cardinalities):
        raise RuntimeError("Generated label validation failed.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = {
        "schema_version": 1,
        "ids": ids,
        "labels": labels,
        "pair_indices": pair_indices,
        "cardinalities": cardinalities,
        "object_names": list(OBJECT_NAMES),
        "pair_prefixes": prefixes,
        "pair_names": ["_".join(PAIR_OBJECTS[prefix]) for prefix in prefixes],
        "pair_objects": {prefix: list(objects) for prefix, objects in PAIR_OBJECTS.items()},
        "records": records,
        "ordering": "01_001..06_078",
        "caption_path": str(
            (args.dataset_root / "Videos" / "structured_v2_captions.csv").resolve()
        ),
    }
    pt_path = args.output_dir / "video_multilabels_2object.pt"
    temporary = pt_path.with_suffix(".pt.tmp")
    torch.save(package, temporary)
    temporary.replace(pt_path)

    json_path = args.output_dir / "video_multilabels_2object.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output_dir / "video_multilabels_2object.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row_index", "video_id", "pair_index", "pair_prefix", "pair_name",
                *OBJECT_NAMES, "label_names", "ground_truth_caption",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "row_index": record["row_index"],
                "video_id": record["video_id"],
                "pair_index": record["pair_index"],
                "pair_prefix": record["pair_prefix"],
                "pair_name": record["pair_name"],
                **{
                    name: record["label"][index]
                    for index, name in enumerate(OBJECT_NAMES)
                },
                "label_names": " ".join(record["label_names"]),
                "ground_truth_caption": record["ground_truth_caption"],
            })
    print("saved 468 ordered labels [person,dog,car,ball,flower,bird]")
    print("PT:", pt_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
