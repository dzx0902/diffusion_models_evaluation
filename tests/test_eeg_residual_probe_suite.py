from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_eeg_residual_probe_suite import select_residual_trials


def test_select_residual_trials_uses_best_unique_videos() -> None:
    metrics = [
        {
            "video_id": "01-001",
            "session": "session1",
            "trial_index": "0",
            "residual_prompt_retrieval_top1": "1",
            "residual_prompt_retrieval_margin": "0.4",
            "residual_pooled_cosine": "0.8",
            "residual_mse_fraction": "0.2",
            "residual_nearest_video_id": "01-001",
            "residual_nearest_prompt": "A person kicks a ball.",
        },
        {
            "video_id": "01-001",
            "session": "session2",
            "trial_index": "0",
            "residual_prompt_retrieval_top1": "1",
            "residual_prompt_retrieval_margin": "0.3",
            "residual_pooled_cosine": "0.9",
            "residual_mse_fraction": "0.1",
            "residual_nearest_video_id": "01-001",
            "residual_nearest_prompt": "A person kicks a ball.",
        },
        {
            "video_id": "02-001",
            "session": "session1",
            "trial_index": "1",
            "residual_prompt_retrieval_top1": "0",
            "residual_prompt_retrieval_margin": "-0.1",
            "residual_pooled_cosine": "0.7",
            "residual_mse_fraction": "0.3",
            "residual_nearest_video_id": "02-002",
            "residual_nearest_prompt": "A dog touches a ball.",
        },
    ]
    trials = [
        {
            "video_id": "01-001",
            "session": "session1",
            "trial_index": "0",
            "duration_sec": "4.0",
            "length_samples": "800",
        },
        {
            "video_id": "01-001",
            "session": "session2",
            "trial_index": "0",
            "duration_sec": "4.0",
            "length_samples": "800",
        },
        {
            "video_id": "02-001",
            "session": "session1",
            "trial_index": "1",
            "duration_sec": "4.0",
            "length_samples": "800",
        },
    ]

    selected = select_residual_trials(metrics, trials, top_k=2)

    assert [row["video_id"] for row in selected] == ["01-001", "02-001"]
    assert selected[0]["session"] == "session1"
    assert selected[1]["nearest_exact_video_id"] == "02-002"
