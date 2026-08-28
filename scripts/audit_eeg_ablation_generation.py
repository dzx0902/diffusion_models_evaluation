"""Fail closed on unmatched videos/seeds or changing trajectories across variants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({str(row["variant"]) for row in rows})
    grouped: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates = []
    for row in rows:
        key = (
            str(row["generator"]), str(row["video_id"]), int(row["seed"]),
            int(row.get("generation_seed", row["seed"])),
        )
        variant = str(row["variant"])
        if variant in grouped[key]:
            duplicates.append((*key, variant))
        grouped[key][variant] = row
    unmatched = [key for key, values in grouped.items() if set(values) != set(variants)]
    trajectory_mismatches = []
    for key, values in grouped.items():
        hashes = {
            tuple(row.get("trajectory_sha256s") or [row.get("trajectory_sha256")])
            for row in values.values() if row.get("trajectory_sha256")
        }
        if len(hashes) > 1:
            trajectory_mismatches.append(key)
    failed = bool(duplicates or unmatched or trajectory_mismatches)
    return {
        "schema_version": 1,
        "variants": variants,
        "jobs": len(rows),
        "matched_cells": len(grouped),
        "duplicates": duplicates,
        "unmatched_cells": unmatched,
        "trajectory_mismatches": trajectory_mismatches,
        "status": "FAIL" if failed else "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(read(args.manifests))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
