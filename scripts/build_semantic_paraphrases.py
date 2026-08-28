"""Build deterministic, semantics-preserving caption paraphrases without an online LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.semantic_caption import SlotPrediction, verbalize_relations, verbalize_semantics
from ms_video_eval.semantic_schema import load_semantic_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def variants(record) -> tuple[str, str, str]:  # type: ignore[no-untyped-def]
    slots = {
        "subject": [SlotPrediction(value, 1.0) for value in record.subjects],
        "object": [SlotPrediction(value, 1.0) for value in record.objects],
        "count": [SlotPrediction(record.subject_count, 1.0)] if record.subject_count else [],
        "action": [SlotPrediction(value, 1.0) for value in record.fine_actions[:1]],
    }
    direct = verbalize_semantics(slots, {key: 0.0 for key in slots})
    entities = [*record.subjects, *record.objects]
    entity_sentence = "The video shows " + " and ".join(entities) + "."
    relation = verbalize_relations(
        [SlotPrediction(value, 1.0) for value in record.relations[:1]], threshold=0.0
    )
    if relation == "A video.":
        relation = direct
    return direct, entity_sentence, relation


def main() -> None:
    args = parse_args()
    records = load_semantic_records(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = [variants(record) for record in records]
    for index in range(3):
        path = args.output_dir / f"paraphrase_{index + 1}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        path.write_text(
            "".join(
                json.dumps(
                    {"schema_version": 1, "video_id": record.video_id, "caption": values[index],
                     "paraphrase_index": index + 1, "source": "deterministic_structured_semantics"},
                    ensure_ascii=False,
                ) + "\n"
                for record, values in zip(records, rendered)
            ),
            encoding="utf-8",
        )
    print(f"[semantic-paraphrases] videos={len(records)} variants=3 output={args.output_dir}")


if __name__ == "__main__":
    main()
