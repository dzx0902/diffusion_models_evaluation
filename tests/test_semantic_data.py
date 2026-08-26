from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ms_video_eval.semantic_data import SemanticVocabulary, load_video_partitions
from src.ms_video_eval.semantic_metrics import search_slot_thresholds
from src.ms_video_eval.semantic_schema import semantic_record_from_source


class SemanticDataTest(unittest.TestCase):
    def test_threshold_search_uses_validation_f1(self) -> None:
        import torch

        logits = {"subject": torch.tensor([[0.0], [-0.2]])}
        targets = {"subject": torch.tensor([[1.0], [0.0]])}
        masks = {"subject": torch.ones(2)}
        selected = search_slot_thresholds(logits, targets, masks, [0.4, 0.5, 0.6])
        self.assertEqual(selected, {"subject": 0.5})

    def test_split_loader_rejects_video_overlap(self) -> None:
        plan = {
            "experiments": [{
                "name": "bad",
                "train_video_ids": ["01-001"],
                "validation_video_ids": ["01-001"],
                "test_video_ids": ["01-002"],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Video leakage"):
                load_video_partitions(path, "bad")

    def test_vocabulary_maps_rare_values_to_unknown(self) -> None:
        rows = [
            semantic_record_from_source(
                {
                    "video_id": "01_001",
                    "caption": "A person kicks a ball.",
                    "caption_relations": ["person kicks ball"],
                },
                "fixture",
            ),
            semantic_record_from_source(
                {
                    "video_id": "01_002",
                    "caption": "A person kicks a ball.",
                    "caption_relations": ["person kicks ball"],
                },
                "fixture",
            ),
            semantic_record_from_source(
                {
                    "video_id": "01_003",
                    "caption": "A person tosses a ball.",
                    "caption_relations": ["person tosses ball"],
                },
                "fixture",
            ),
        ]
        vocabulary = SemanticVocabulary.fit(rows[:2], ["fine_action"], {"fine_action": 2})
        target, _ = vocabulary.encode(rows[2])
        self.assertEqual(vocabulary.values["fine_action"], ("kicks", "__unknown__"))
        self.assertEqual(target["fine_action"].tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
