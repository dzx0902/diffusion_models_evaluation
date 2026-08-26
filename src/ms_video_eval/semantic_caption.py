"""Deterministic, confidence-aware captions for semantic EEG predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


ARTICLES = {
    "person": "a person",
    "dog": "a dog",
    "bird": "a bird",
    "ball": "a ball",
    "car": "a car",
    "flower": "a flower",
}

COARSE_ACTION_VERBALIZATIONS = {
    "observe": "observes",
    "transfer": "interacts with",
    "contact": "interacts with",
    "carry": "holds",
    "approach": "moves toward",
    "locomotion": "moves near",
    "stationary": "remains near",
    "inspect": "interacts with",
    "manipulate": "manipulates",
    "other": "is shown with",
}


@dataclass(frozen=True)
class SlotPrediction:
    value: str
    confidence: float


def retain_confident(
    predictions: Sequence[SlotPrediction],
    threshold: float,
) -> list[SlotPrediction]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    return [item for item in predictions if item.confidence >= threshold]


def _join_natural(values: Sequence[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


def _plural_verb_phrase(value: str) -> str:
    replacements = {
        "observes": "observe",
        "interacts with": "interact with",
        "holds": "hold",
        "moves toward": "move toward",
        "moves near": "move near",
        "remains near": "remain near",
        "manipulates": "manipulate",
        "is shown with": "are shown with",
        "is shown": "are shown",
    }
    return replacements.get(value, value)


def verbalize_semantics(
    slots: Mapping[str, Sequence[SlotPrediction]],
    thresholds: Mapping[str, float] | None = None,
) -> str:
    """Create a conservative caption and omit uncertain semantic slots.

    The verbalizer never adds style, trajectory, scene, or motion direction.
    Fine actions are expected in base/present-tense phrases such as ``kicks`` or
    ``runs after``; when no action is reliable, only entity presence is stated.
    """

    thresholds = thresholds or {}

    def selected(name: str) -> list[str]:
        threshold = float(thresholds.get(name, 0.5))
        return [item.value for item in retain_confident(slots.get(name, ()), threshold)]

    subjects = selected("subject")
    objects = selected("object")
    actions = selected("action")
    scenes = selected("scene")
    counts = selected("count")

    subject_text = _join_natural([ARTICLES.get(value, value) for value in subjects])
    object_text = _join_natural([ARTICLES.get(value, value) for value in objects])
    if counts and len(subjects) == 1 and subjects[0] == "person":
        number = counts[0]
        subject_text = f"{number.capitalize()} people" if number != "one" else "A person"

    if subject_text and actions:
        action = _plural_verb_phrase(actions[0]) if len(subjects) > 1 else actions[0]
        sentence = f"{subject_text.capitalize()} {action}"
        if object_text:
            sentence += f" {object_text}"
    elif subject_text and object_text:
        verb = "are shown with" if len(subjects) > 1 else "is shown with"
        sentence = f"{subject_text.capitalize()} {verb} {object_text}"
    elif subject_text:
        sentence = f"{subject_text.capitalize()} is shown"
    elif object_text:
        sentence = f"{object_text.capitalize()} is shown"
    else:
        sentence = "A video"
    if scenes:
        sentence += f" in {scenes[0]}"
    return sentence.rstrip(". ") + "."


def verbalize_relations(relations: Sequence[SlotPrediction], threshold: float = 0.5) -> str:
    """Turn source-style ``subject verb object`` relations into clauses."""

    retained = retain_confident(relations, threshold)
    sentences: list[str] = []
    for prediction in retained:
        words = prediction.value.split()
        if not words:
            continue
        rendered = [ARTICLES.get(word, word) for word in words]
        clause = " ".join(rendered)
        sentences.append(clause[:1].upper() + clause[1:].rstrip(". ") + ".")
    return " ".join(sentences) if sentences else "A video."
