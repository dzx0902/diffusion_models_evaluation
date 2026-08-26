"""Canonical semantic annotations for the EEG caption ablation.

The source ``structured_v2`` annotations are useful but intentionally free
form: entities contain aliases and incidental objects, while relations contain
hundreds of surface forms.  This module preserves those source fields for
audit, and derives a small, deterministic core ontology for supervised EEG
decoding.  It does not learn vocabulary or statistics from validation/test
records.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CORE_ENTITIES_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "01": ("person", "ball"),
    "02": ("dog", "ball"),
    "03": ("dog", "car"),
    "04": ("car", "flower"),
    "05": ("bird", "flower"),
    "06": ("person", "bird"),
    "07": ("person", "dog", "ball"),
    "08": ("person", "bird", "flower"),
}

ANIMATE_ENTITIES = frozenset({"person", "dog", "bird"})
ENTITY_ALIASES = {
    "people": "person",
    "persons": "person",
    "dogs": "dog",
    "birds": "bird",
    "balls": "ball",
    "cars": "car",
    "flowers": "flower",
    "van": "car",
    "truck": "car",
}

# The order is significant: more specific groups are checked first.
COARSE_ACTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("observe", ("watch", "look", "stare", "observe")),
    ("transfer", ("throw", "toss", "pass", "release", "drop")),
    ("contact", ("kick", "hit", "bat", "tap", "paw", "peck", "nudge", "push", "pull")),
    ("carry", ("hold", "carry", "pick up", "grab", "grip", "mouth", "catch")),
    (
        "approach",
        (
            "approach", "chase", "follow", "run after", "runs after", "ran after",
            "walk toward", "walks toward", "move toward", "moves toward",
        ),
    ),
    ("locomotion", ("run", "walk", "drive", "roll", "fly", "flutter", "hop", "move")),
    ("stationary", ("stand", "sit", "lie", "park", "stop", "perch", "hover")),
    ("inspect", ("sniff", "feed", "eat", "scratch")),
    ("manipulate", ("dribble", "spin", "bounce", "putt")),
)

COUNT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bthree (?:people|persons|men|women)\b", "three"),
    (r"\btwo (?:people|persons|men|women)\b", "two"),
    (r"\b(?:a|one) (?:person|man|woman)\b", "one"),
)


def normalize_video_id(value: object) -> str:
    """Normalize both ``01_001`` and ``01-001`` to the project convention."""

    match = re.fullmatch(r"\s*(\d{2})[-_](\d{3})(?:\.mp4)?\s*", str(value))
    if match is None:
        raise ValueError(f"Invalid video_id: {value!r}")
    return f"{match.group(1)}-{match.group(2)}"


def normalize_entity(value: str) -> str:
    entity = re.sub(r"\s+", " ", value.strip().lower())
    return ENTITY_ALIASES.get(entity, entity)


def category_from_video_id(video_id: str) -> str:
    return normalize_video_id(video_id).split("-", 1)[0]


def subject_count_from_caption(caption: str) -> str | None:
    lowered = caption.lower()
    for pattern, value in COUNT_PATTERNS:
        if re.search(pattern, lowered):
            return value
    return None


def action_phrase_from_relation(relation: str, entities: Sequence[str]) -> str:
    """Remove leading/trailing entities from a relation surface form.

    This is deliberately a conservative string operation.  The original
    relation remains in every record, so later manually curated action labels
    can replace this derived field without losing provenance.
    """

    phrase = re.sub(r"[^a-z0-9 ]+", " ", relation.lower())
    phrase = re.sub(r"\s+", " ", phrase).strip()
    aliases = sorted(
        {item for entity in entities for item in (entity, *[k for k, v in ENTITY_ALIASES.items() if v == entity])},
        key=len,
        reverse=True,
    )
    for entity in aliases:
        if phrase == entity:
            return ""
        if phrase.startswith(entity + " "):
            phrase = phrase[len(entity) + 1 :]
            break
    for entity in aliases:
        if phrase.endswith(" " + entity):
            phrase = phrase[: -(len(entity) + 1)]
            break
    return phrase.strip()


def coarse_action(action_phrase: str) -> str:
    lowered = action_phrase.lower()
    for label, keywords in COARSE_ACTION_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "other"


@dataclass(frozen=True)
class SemanticRecord:
    schema_version: int
    video_id: str
    category_id: str
    caption: str
    core_entities: tuple[str, ...]
    subjects: tuple[str, ...]
    objects: tuple[str, ...]
    subject_count: str | None
    coarse_actions: tuple[str, ...]
    fine_actions: tuple[str, ...]
    relations: tuple[str, ...]
    source_entities: tuple[str, ...]
    source_manifest: str
    annotation_status: str = "derived_needs_audit"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def semantic_record_from_source(row: dict[str, Any], source_manifest: str) -> SemanticRecord:
    video_id = normalize_video_id(row["video_id"])
    category = category_from_video_id(video_id)
    if category not in CORE_ENTITIES_BY_CATEGORY:
        raise ValueError(f"Unsupported category {category!r} for {video_id}")
    core_entities = CORE_ENTITIES_BY_CATEGORY[category]
    relations = tuple(
        re.sub(r"\s+", " ", str(value).strip().lower())
        for value in row.get("caption_relations", [])
        if str(value).strip()
    )
    fine_actions = tuple(
        dict.fromkeys(
            phrase
            for relation in relations
            if (phrase := action_phrase_from_relation(relation, core_entities))
        )
    )
    actions = tuple(dict.fromkeys(coarse_action(value) for value in fine_actions))
    subjects = tuple(value for value in core_entities if value in ANIMATE_ENTITIES)
    objects = tuple(value for value in core_entities if value not in ANIMATE_ENTITIES)
    return SemanticRecord(
        schema_version=1,
        video_id=video_id,
        category_id=category,
        caption=str(row["caption"]).strip(),
        core_entities=core_entities,
        subjects=subjects,
        objects=objects,
        subject_count=subject_count_from_caption(str(row["caption"])),
        coarse_actions=actions or ("other",),
        fine_actions=fine_actions,
        relations=relations,
        source_entities=tuple(
            dict.fromkeys(normalize_entity(str(value)) for value in row.get("caption_entities", []))
        ),
        source_manifest=source_manifest,
    )


def load_source_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [normalize_video_id(row["video_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate source video IDs: {duplicates[:5]}")
    return rows


def build_semantic_records(path: Path) -> list[SemanticRecord]:
    source = str(path.resolve())
    return [semantic_record_from_source(row, source) for row in load_source_manifest(path)]


def load_semantic_records(path: Path) -> list[SemanticRecord]:
    records: list[SemanticRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in (
            "core_entities", "subjects", "objects", "coarse_actions", "fine_actions",
            "relations", "source_entities",
        ):
            row[field] = tuple(row[field])
        records.append(SemanticRecord(**row))
    ids = [record.video_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Semantic label file contains duplicate video IDs")
    return records


def semantic_audit(records: Iterable[SemanticRecord]) -> dict[str, Any]:
    values = list(records)
    return {
        "schema_version": 1,
        "video_count": len(values),
        "category_counts": dict(sorted(Counter(item.category_id for item in values).items())),
        "subject_counts": dict(sorted(Counter(x for item in values for x in item.subjects).items())),
        "object_counts": dict(sorted(Counter(x for item in values for x in item.objects).items())),
        "coarse_action_counts": dict(
            sorted(Counter(x for item in values for x in item.coarse_actions).items())
        ),
        "fine_action_unique": len({x for item in values for x in item.fine_actions}),
        "relation_unique": len({x for item in values for x in item.relations}),
        "missing_relation_count": sum(not item.relations for item in values),
        "explicit_count_coverage": (
            sum(item.subject_count is not None for item in values) / len(values) if values else 0.0
        ),
        "annotation_status_counts": dict(
            sorted(Counter(item.annotation_status for item in values).items())
        ),
    }
