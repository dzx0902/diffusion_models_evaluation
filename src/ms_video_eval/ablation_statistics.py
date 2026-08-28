"""Matched aggregation and dependency-free paired tests for ablations."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


IDENTIFIERS = ("variant", "generator", "subject", "fold", "seed", "generation_seed", "video_id")


def aggregate_long_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["generator"]), str(row["metric"]))].append(
            float(row["value"])
        )
    result = []
    for (variant, generator, metric), values in sorted(grouped.items()):
        result.append(
            {
                "variant": variant,
                "generator": generator,
                "metric": metric,
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "sem": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
            }
        )
    return result


def paired_sign_flip_test(
    differences: Sequence[float],
    permutations: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Two-sided paired randomization test; exact for at most 20 pairs."""

    values = [float(value) for value in differences]
    if len(values) < 2:
        raise ValueError("A paired test requires at least two matched observations")
    observed = abs(statistics.fmean(values))
    if len(values) <= 20:
        signs: Iterable[tuple[int, ...]] = itertools.product((-1, 1), repeat=len(values))
        total = 2 ** len(values)
    else:
        rng = random.Random(seed)
        signs = (tuple(rng.choice((-1, 1)) for _ in values) for _ in range(permutations))
        total = permutations
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(pattern, values)) / len(values)) >= observed - 1e-15
        for pattern in signs
    )
    std = statistics.stdev(values)
    return {
        "n": len(values),
        "mean_difference": statistics.fmean(values),
        "std_difference": std,
        "cohen_dz": statistics.fmean(values) / std if std else 0.0,
        "p_value": (extreme + (0 if len(values) <= 20 else 1)) / (total + (0 if len(values) <= 20 else 1)),
        "permutations": total,
    }


def paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
) -> list[dict[str, Any]]:
    keys = ("generator", "metric")
    index: dict[tuple[str, str, str, str, str, int, int, str], float] = {}
    variants = set()
    for row in rows:
        variant = str(row["variant"])
        variants.add(variant)
        key = (
            variant,
            str(row["generator"]),
            str(row["metric"]),
            str(row["subject"]),
            str(row["fold"]),
            int(row["seed"]),
            int(row.get("generation_seed", -1)),
            str(row["video_id"]),
        )
        if key in index:
            raise ValueError(f"Duplicate matched metric row: {key}")
        index[key] = float(row["value"])
    if baseline not in variants:
        raise KeyError(f"Unknown baseline variant: {baseline}")
    result = []
    for variant in sorted(variants - {baseline}):
        combinations = sorted({(key[1], key[2]) for key in index if key[0] == variant})
        for generator, metric in combinations:
            variant_keys = {
                key[3:]: value for key, value in index.items()
                if key[:3] == (variant, generator, metric)
            }
            baseline_keys = {
                key[3:]: value for key, value in index.items()
                if key[:3] == (baseline, generator, metric)
            }
            if set(variant_keys) != set(baseline_keys):
                raise ValueError(
                    f"Unmatched {variant} vs {baseline} for {generator}/{metric}: "
                    f"variant={len(variant_keys)} baseline={len(baseline_keys)}"
                )
            if len(variant_keys) < 2:
                continue
            differences = [variant_keys[key] - baseline_keys[key] for key in sorted(variant_keys)]
            result.append(
                {
                    "baseline": baseline,
                    "variant": variant,
                    "generator": generator,
                    "metric": metric,
                    **paired_sign_flip_test(differences),
                }
            )
    ordered = sorted(range(len(result)), key=lambda index: float(result[index]["p_value"]))
    running = 0.0
    count = len(result)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(result[index]["p_value"]))
        running = max(running, adjusted)
        result[index]["p_holm"] = running
    return result
