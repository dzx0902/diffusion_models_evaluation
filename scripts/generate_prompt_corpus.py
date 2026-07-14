"""Generate a diverse prompt corpus for text-space analysis.

The output JSONL can be passed directly to scripts/analyze_wan_text_space.py via
--prompt-file. A plain-text file with one prompt per line is also written.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


SUBJECTS = [
    "dog",
    "cat",
    "person",
    "bicycle",
    "car",
    "bus",
    "motorcycle",
    "ball",
    "bird",
    "horse",
    "chair",
    "backpack",
    "umbrella",
    "kite",
    "skateboard",
    "traffic cone",
    "robot toy",
    "potted plant",
]

MOTIONS = [
    "walks slowly toward the center",
    "moves from left to right",
    "moves from right to left",
    "circles around the other subject",
    "approaches the other subject and then stops",
    "remains stationary",
    "turns slightly in place",
    "rolls slowly forward",
    "moves diagonally across the frame",
    "briefly pauses and then continues moving",
    "sways gently",
    "jumps once and lands near the center",
]

SCENES = [
    "outdoor street",
    "city sidewalk",
    "quiet park",
    "garden path",
    "school playground",
    "parking lot",
    "beach walkway",
    "indoor studio",
    "living room",
    "suburban driveway",
    "open plaza",
    "forest trail",
]

CAMERAS = [
    "static camera",
    "slow dolly shot",
    "slight handheld camera",
    "wide-angle fixed camera",
    "low-angle camera",
    "eye-level camera",
]

LIGHTING = [
    "natural daylight",
    "soft overcast light",
    "warm sunset light",
    "bright indoor lighting",
    "cinematic soft lighting",
]

SPATIAL_RELATIONS = [
    ("left", "right"),
    ("right", "left"),
    ("front", "back"),
    ("back", "front"),
    ("upper left", "lower right"),
    ("lower right", "upper left"),
    ("center", "right"),
    ("left", "center"),
]

STYLES = [
    "realistic video",
    "documentary-style video",
    "natural handheld video",
    "clean studio video",
    "cinematic realistic video",
]

TEMPLATES = [
    (
        "A {style} with two main subjects: one {subject_a} and one {subject_b}. "
        "The {subject_a} starts on the {pos_a} side of the frame and {motion_a}. "
        "The {subject_b} starts on the {pos_b} side of the frame and {motion_b}. "
        "Scene: {scene}. Camera: {camera}. Lighting: {lighting}. "
        "Both subjects remain visible, separated, and easy to count."
    ),
    (
        "Create a realistic 4-second video in a {scene}. "
        "There is exactly one {subject_a} on the {pos_a} and exactly one {subject_b} on the {pos_b}. "
        "During the shot, the {subject_a} {motion_a}, while the {subject_b} {motion_b}. "
        "Use a {camera} with {lighting}. No extra main objects."
    ),
    (
        "A multi-subject video showing a {subject_a} and a {subject_b}. "
        "At the beginning, the {subject_a} is {pos_a} and the {subject_b} is {pos_b}. "
        "The {subject_a} {motion_a}; the {subject_b} {motion_b}. "
        "The scene is a {scene}, filmed with a {camera}. "
        "Keep the composition stable and preserve the spatial relationship."
    ),
    (
        "In a {scene}, show exactly two prominent subjects: a {subject_a} and a {subject_b}. "
        "The {subject_a} appears on the {pos_a}; the {subject_b} appears on the {pos_b}. "
        "Motion: the {subject_a} {motion_a}; the {subject_b} {motion_b}. "
        "Use {lighting} and a {camera}. The video should look natural and artifact-free."
    ),
]


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    subject_a: str
    subject_b: str
    motion_a: str
    motion_b: str
    position_a: str
    position_b: str
    scene: str
    camera: str
    lighting: str
    style: str
    template_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diverse prompts for text-space analysis.")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("outputs/text_space/prompt_corpus/wan_prompt_corpus.jsonl"),
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("outputs/text_space/prompt_corpus/wan_prompt_corpus.txt"),
    )
    parser.add_argument(
        "--max-pair-repeats",
        type=int,
        default=12,
        help="Soft cap for repeated unordered subject pairs before reshuffling attempts continue.",
    )
    return parser.parse_args()


def choose_subject_pair(rng: random.Random, pair_counts: dict[tuple[str, str], int], max_repeats: int):
    for _ in range(100):
        subject_a, subject_b = rng.sample(SUBJECTS, 2)
        key = tuple(sorted((subject_a, subject_b)))
        if pair_counts.get(key, 0) < max_repeats:
            pair_counts[key] = pair_counts.get(key, 0) + 1
            return subject_a, subject_b
    subject_a, subject_b = rng.sample(SUBJECTS, 2)
    key = tuple(sorted((subject_a, subject_b)))
    pair_counts[key] = pair_counts.get(key, 0) + 1
    return subject_a, subject_b


def build_record(index: int, rng: random.Random, pair_counts: dict[tuple[str, str], int], max_repeats: int):
    subject_a, subject_b = choose_subject_pair(rng, pair_counts, max_repeats)
    pos_a, pos_b = rng.choice(SPATIAL_RELATIONS)
    motion_a = rng.choice(MOTIONS)
    motion_b = rng.choice(MOTIONS)
    if motion_a == "remains stationary" and motion_b == "remains stationary":
        motion_a = rng.choice([motion for motion in MOTIONS if motion != "remains stationary"])

    fields = {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "pos_a": pos_a,
        "pos_b": pos_b,
        "motion_a": motion_a,
        "motion_b": motion_b,
        "scene": rng.choice(SCENES),
        "camera": rng.choice(CAMERAS),
        "lighting": rng.choice(LIGHTING),
        "style": rng.choice(STYLES),
    }
    template_id = rng.randrange(len(TEMPLATES))
    prompt = TEMPLATES[template_id].format(**fields)
    return PromptRecord(
        id=f"prompt_{index:05d}",
        prompt=prompt,
        subject_a=subject_a,
        subject_b=subject_b,
        motion_a=motion_a,
        motion_b=motion_b,
        position_a=pos_a,
        position_b=pos_b,
        scene=fields["scene"],
        camera=fields["camera"],
        lighting=fields["lighting"],
        style=fields["style"],
        template_id=template_id,
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pair_counts: dict[tuple[str, str], int] = {}
    records: list[PromptRecord] = []
    seen_prompts: set[str] = set()

    attempts = 0
    while len(records) < args.count:
        attempts += 1
        if attempts > args.count * 50:
            raise RuntimeError("Could not generate enough unique prompts; lower count or max-pair-repeats.")
        record = build_record(len(records), rng, pair_counts, args.max_pair_repeats)
        if record.prompt in seen_prompts:
            continue
        seen_prompts.add(record.prompt)
        records.append(record)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True))
            handle.write("\n")
    with args.output_txt.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.prompt)
            handle.write("\n")

    print(f"[prompt-corpus] wrote {len(records)} prompts")
    print(f"[prompt-corpus] jsonl: {args.output_jsonl}")
    print(f"[prompt-corpus] txt:   {args.output_txt}")


if __name__ == "__main__":
    main()
