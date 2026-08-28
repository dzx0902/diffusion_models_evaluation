from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_statistics import (
    aggregate_long_metrics,
    paired_comparisons,
    paired_sign_flip_test,
)


def rows() -> list[dict[str, object]]:
    result = []
    for variant, values in (("base", (0.1, 0.2, 0.3)), ("full", (0.3, 0.4, 0.5))):
        for index, value in enumerate(values):
            result.append(
                {"variant": variant, "generator": "tora", "metric": "score",
                 "subject": "s1", "fold": "f1", "seed": 0,
                 "video_id": f"v{index}", "value": value}
            )
    return result


def test_aggregate_and_matched_statistics() -> None:
    summary = aggregate_long_metrics(rows())
    assert len(summary) == 2
    tests = paired_comparisons(rows(), "base")
    assert tests[0]["n"] == 3
    assert abs(float(tests[0]["mean_difference"]) - 0.2) < 1e-9
    assert "p_holm" in tests[0]


def test_sign_flip_is_two_sided_and_bounded() -> None:
    result = paired_sign_flip_test([1.0, 1.0, 1.0])
    assert 0 <= float(result["p_value"]) <= 1
