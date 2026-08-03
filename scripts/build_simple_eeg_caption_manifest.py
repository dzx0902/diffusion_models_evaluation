"""Create a concise, category-consistent caption manifest for EEG-to-Wan training.

The source captions are intentionally preserved in ``source_caption``.  The
output ``caption`` keeps only the category subjects and their primary action or
relation, so it can be used as a separate supervision target without changing
the original annotations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build concise category-consistent EEG caption manifest.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "manifests" / "video_manifest.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "simple_v1_video_manifest.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def contains(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def person_ball(text: str) -> str:
    if contains(text, "tosses and catches", "throws and catches", "catches and tosses"):
        action = "tosses and catches"
    elif contains(text, "dribbles"):
        action = "dribbles"
    elif contains(text, "kicks"):
        action = "kicks"
    elif contains(text, "rolls", "rolling"):
        action = "rolls"
    elif contains(text, "pushes", "pushing"):
        action = "pushes"
    elif contains(text, "passes"):
        action = "passes"
    elif contains(text, "bats", "bumps", "strikes", "putts", "hits"):
        action = "hits"
    elif contains(text, "tosses", "throws"):
        action = "throws"
    elif contains(text, "releases", "drops"):
        action = "releases"
    elif contains(text, "spins"):
        action = "spins"
    else:
        action = "moves"
    return f"A person {action} a ball."


def dog_ball(text: str) -> str:
    if contains(text, "chases"):
        action = "chases"
    elif contains(text, "catches", "picks up", "grips"):
        action = "catches"
    elif contains(text, "mouths", "bites"):
        action = "mouths"
    elif contains(text, "paws", "bats", "taps"):
        action = "paws"
    elif contains(text, "sniffs", "noses", "nudges", "investigates", "inspects"):
        action = "sniffs"
    elif contains(text, "watches"):
        action = "watches"
    else:
        action = "is near"
    return f"A dog {action} a ball."


def dog_car(text: str) -> str:
    if contains(text, "car drives", "vehicle drives", "van drives", "suv drives", "hatchback drives", "truck drives"):
        return "A car drives past a dog."
    if contains(text, "dog walks", "dog runs", "retriever walks", "husky walks", "corgi walks", "terrier walks"):
        return "A dog walks near a car."
    if contains(text, "watches", "facing"):
        return "A dog watches a car."
    return "A dog is near a car."


def car_flower(text: str) -> str:
    if contains(text, "drives", "moving", "approaching", "passes"):
        return "A car drives past flowers."
    return "A car is parked near flowers."


def bird_flower(text: str) -> str:
    if contains(text, "hovers", "flying"):
        action = "hovers"
    elif contains(text, "pecks", "feeds", "probes"):
        action = "pecks"
    elif contains(text, "lands"):
        action = "lands"
    elif contains(text, "walks", "hops"):
        action = "moves"
    else:
        action = "perches"
    return f"A bird {action} near flowers."


def person_bird(text: str) -> str:
    if contains(text, "feeds", "food", "treat"):
        return "A person feeds a bird."
    if contains(text, "releases"):
        return "A person releases a bird."
    if contains(text, "holds", "offers", "presents"):
        return "A person interacts with a bird."
    return "A person watches a bird."


def person_dog_ball(text: str) -> str:
    if contains(text, "throws", "tosses", "launches"):
        return "A person throws a ball and a dog chases it."
    if contains(text, "kicks"):
        return "A person kicks a ball and a dog follows it."
    if contains(text, "rolls"):
        return "A person rolls a ball and a dog follows it."
    return "A person and a dog play with a ball."


def person_bird_flower(text: str) -> str:
    if contains(text, "feeds", "probes", "pecks"):
        return "A person watches a bird feed near flowers."
    if contains(text, "flies", "hovers", "flutter"):
        return "A person watches a bird move near flowers."
    return "A person watches a bird near flowers."


SIMPLIFIERS: dict[str, Callable[[str], str]] = {
    "01": person_ball,
    "02": dog_ball,
    "03": dog_car,
    "04": car_flower,
    "05": bird_flower,
    "06": person_bird,
    "07": person_dog_ball,
    "08": person_bird_flower,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    records = read_jsonl(args.input)
    seen: set[str] = set()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    output: list[dict] = []
    for row in records:
        video_id = str(row["video_id"])
        category = str(row["category_id"])
        if video_id in seen:
            raise ValueError(f"Duplicate video_id: {video_id}")
        if category not in SIMPLIFIERS:
            raise ValueError(f"No simplifier registered for category {category!r}")
        seen.add(video_id)
        source_caption = str(row["caption"]).strip()
        caption = SIMPLIFIERS[category](source_caption.lower())
        if not caption:
            raise ValueError(f"Empty simplified caption for {video_id}")
        simplified = dict(row)
        simplified["source_caption"] = source_caption
        simplified["caption"] = caption
        simplified["caption_scheme"] = "simple_v1"
        output.append(simplified)
        by_category[category][caption] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[simple-captions] wrote {args.output} ({len(output)} videos)")
    for category in sorted(by_category):
        print(f"[simple-captions] category={category} templates={len(by_category[category])} {dict(by_category[category])}")


if __name__ == "__main__":
    main()
