"""Build deterministic, auditable semantic labels from structured captions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_schema import build_semantic_records, semantic_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "manifests" / "structured_v2_video_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "eeg_semantic_labels_v1.jsonl",
    )
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--expected-videos", type=int, default=624)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    for path in (args.output, audit_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    records = build_semantic_records(args.input)
    if len(records) != args.expected_videos:
        raise ValueError(f"Expected {args.expected_videos} videos, found {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item.to_json(), ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    audit = semantic_audit(records)
    audit["source"] = str(args.input.resolve())
    audit["output"] = str(args.output.resolve())
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[semantic-labels] wrote {len(records)} records to {args.output}")
    print(f"[semantic-labels] audit: {audit_path}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
