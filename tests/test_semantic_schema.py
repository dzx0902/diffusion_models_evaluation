from __future__ import annotations

import unittest

from src.ms_video_eval.semantic_caption import (
    SlotPrediction,
    verbalize_relations,
    verbalize_semantics,
)
from src.ms_video_eval.semantic_schema import (
    action_phrase_from_relation,
    coarse_action,
    normalize_video_id,
    semantic_record_from_source,
)


class SemanticSchemaTest(unittest.TestCase):
    def test_video_id_normalizes_both_separators(self) -> None:
        self.assertEqual(normalize_video_id("01_001"), "01-001")
        self.assertEqual(normalize_video_id("01-001.mp4"), "01-001")

    def test_relation_is_reduced_to_action_phrase(self) -> None:
        self.assertEqual(
            action_phrase_from_relation("dog runs after ball", ("dog", "ball")),
            "runs after",
        )
        self.assertEqual(coarse_action("runs after"), "approach")

    def test_source_record_uses_category_core_entities(self) -> None:
        record = semantic_record_from_source(
            {
                "video_id": "07_001",
                "caption": "A person throws a ball. A dog catches it.",
                "caption_entities": ["person", "dog", "ball"],
                "caption_relations": ["person throws ball", "dog catches ball"],
            },
            "fixture.jsonl",
        )
        self.assertEqual(record.core_entities, ("person", "dog", "ball"))
        self.assertEqual(record.subjects, ("person", "dog"))
        self.assertEqual(record.objects, ("ball",))
        self.assertEqual(record.fine_actions, ("throws", "catches"))

    def test_verbalizer_omits_low_confidence_slots(self) -> None:
        caption = verbalize_semantics(
            {
                "subject": [SlotPrediction("person", 0.9)],
                "object": [SlotPrediction("ball", 0.8)],
                "action": [SlotPrediction("kicks", 0.7)],
                "scene": [SlotPrediction("a park", 0.2)],
            },
            {"subject": 0.5, "object": 0.5, "action": 0.5, "scene": 0.5},
        )
        self.assertEqual(caption, "A person kicks a ball.")

    def test_relation_verbalizer_is_deterministic(self) -> None:
        caption = verbalize_relations(
            [SlotPrediction("person throws ball", 0.8), SlotPrediction("dog catches ball", 0.7)]
        )
        self.assertEqual(caption, "A person throws a ball. A dog catches a ball.")

    def test_multiple_subjects_use_plural_verb(self) -> None:
        caption = verbalize_semantics(
            {
                "subject": [SlotPrediction("person", 0.9), SlotPrediction("dog", 0.8)],
                "object": [SlotPrediction("ball", 0.9)],
                "action": [SlotPrediction("moves toward", 0.8)],
            }
        )
        self.assertEqual(caption, "A person and a dog move toward a ball.")


if __name__ == "__main__":
    unittest.main()
