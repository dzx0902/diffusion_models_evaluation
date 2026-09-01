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


def audit(
    rows: list[dict[str, Any]],
    generators: set[str] | None = None,
    generator_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    generator_aliases = generator_aliases or {}
    if generators is not None:
        rows = [row for row in rows if str(row["generator"]) in generators]
    rows = [
        {**row, "generator": generator_aliases.get(str(row["generator"]), str(row["generator"]))}
        for row in rows
    ]
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
    failed = bool(not rows or duplicates or unmatched or trajectory_mismatches)
    return {
        "schema_version": 1,
        "variants": variants,
        "jobs": len(rows),
        "matched_cells": len(grouped),
        "duplicates": duplicates,
        "unmatched_cells": unmatched,
        "trajectory_mismatches": trajectory_mismatches,
        "generator_filter": sorted(generators) if generators is not None else None,
        "generator_aliases": generator_aliases,
        "status": "FAIL" if failed else "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--generators", nargs="+", default=None)
    parser.add_argument(
        "--generator-alias", action="append", default=[], metavar="SOURCE=TARGET",
        help="Normalize routes into one paired comparison family; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aliases = {}
    for value in args.generator_alias:
        if "=" not in value:
            parser.error(f"Invalid --generator-alias {value!r}; expected SOURCE=TARGET")
        source, target = value.split("=", 1)
        aliases[source] = target
    result = audit(
        read(args.manifests),
        set(args.generators) if args.generators is not None else None,
        aliases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
