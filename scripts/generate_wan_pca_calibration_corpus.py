"""Create a prompt-only calibration corpus for a global Wan token PCA basis.

The corpus uses short, structured multi-subject descriptions resembling the
EEG experiment, but excludes every caption in an optional experiment manifest.
It can therefore fit a fixed global PCA basis without test-caption leakage.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCENES = ["a park", "a street", "a garden", "a sidewalk", "an open field", "a playground"]
PERSON_BALL_ACTIONS = ["kicks", "throws", "holds", "rolls", "carries", "reaches for"]
DOG_BALL_ACTIONS = ["sniffs", "nudges", "chases", "catches", "watches", "runs after"]
PERSON_BIRD_ACTIONS = ["watches", "walks toward", "stands near"]
BIRD_FLOWER_ACTIONS = ["hovers beside", "lands near", "feeds from", "flies past"]
VEHICLE_ACTIONS = ["passes", "stops beside", "moves toward", "waits near"]
PERSON_BICYCLE_ACTIONS = ["walks beside", "stands beside", "moves toward"]
EXTRA_SUBJECTS = ["a cat", "a horse", "a bus", "a kite", "a skateboard", "a potted plant"]
EXTRA_ACTIONS = ["moves beside", "stops near", "passes", "watches", "approaches"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a non-overlapping short-caption corpus for global Wan PCA.")
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def excluded_prompts(path: Path | None) -> set[str]:
    if path is None:
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row.get("caption", "")).strip() for row in rows if str(row.get("caption", "")).strip()}


def build_prompt(rng: random.Random) -> tuple[str, str]:
    scene = rng.choice(SCENES)
    family = rng.choice(["person_ball", "dog_ball", "person_bicycle", "bird_flower", "triple_ball", "triple_flower", "vehicle", "extra"])
    if family == "person_ball":
        return f"A person {rng.choice(PERSON_BALL_ACTIONS)} a ball in {scene}.", family
    if family == "dog_ball":
        return f"A dog {rng.choice(DOG_BALL_ACTIONS)} a ball in {scene}.", family
    if family == "person_bicycle":
        return f"A person {rng.choice(PERSON_BICYCLE_ACTIONS)} a bicycle in {scene}.", family
    if family == "bird_flower":
        return f"A bird {rng.choice(BIRD_FLOWER_ACTIONS)} a flower in {scene}.", family
    if family == "triple_ball":
        return (f"A person {rng.choice(PERSON_BALL_ACTIONS)} a ball. A dog {rng.choice(DOG_BALL_ACTIONS)} the ball in {scene}.", family)
    if family == "triple_flower":
        return (f"A person {rng.choice(PERSON_BIRD_ACTIONS)} a bird. The bird {rng.choice(BIRD_FLOWER_ACTIONS)} a flower in {scene}.", family)
    if family == "vehicle":
        subject = rng.choice(["A car", "A bicycle", "A bus"])
        object_ = rng.choice(["a ball", "a flower", "a person", "a dog"])
        return f"{subject} {rng.choice(VEHICLE_ACTIONS)} {object_} in {scene}.", family
    subject = rng.choice(EXTRA_SUBJECTS)
    object_ = rng.choice(EXTRA_SUBJECTS + ["a person", "a dog", "a ball"])
    return f"{subject.capitalize()} {rng.choice(EXTRA_ACTIONS)} {object_} in {scene}.", family


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    blocked = excluded_prompts(args.exclude_manifest)
    rng = random.Random(args.seed)
    rows = []
    seen = set(blocked)
    attempts = 0
    while len(rows) < args.count:
        attempts += 1
        if attempts > args.count * 100:
            raise RuntimeError("Could not create enough unique calibration prompts; reduce --count.")
        prompt, family = build_prompt(rng)
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append({"id": f"calibration_{len(rows):05d}", "prompt": prompt, "family": family})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[wan-pca-calibration] wrote {len(rows)} prompts to {args.output}")
    if blocked:
        print(f"[wan-pca-calibration] excluded {len(blocked)} experiment captions", flush=True)


if __name__ == "__main__":
    main()
