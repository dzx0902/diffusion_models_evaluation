from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_eeg_wan_predictions import prompt_retrieval_metrics


class EEGPredictionRankingTest(unittest.TestCase):
    def test_prompt_retrieval_reports_top1_and_positive_margin(self) -> None:
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        metrics = prompt_retrieval_metrics(torch.tensor([0.9, 0.1]), candidates, true_index=0)

        self.assertEqual(metrics["prompt_retrieval_rank"], 1)
        self.assertEqual(metrics["prompt_retrieval_top1"], 1)
        self.assertGreater(metrics["prompt_retrieval_margin"], 0.0)

    def test_prompt_retrieval_reports_wrong_target(self) -> None:
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        metrics = prompt_retrieval_metrics(torch.tensor([0.1, 0.9]), candidates, true_index=0)

        self.assertEqual(metrics["prompt_retrieval_rank"], 2)
        self.assertEqual(metrics["prompt_retrieval_top1"], 0)
        self.assertLess(metrics["prompt_retrieval_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
