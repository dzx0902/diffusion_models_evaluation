import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_c2_reconstruction_yolo import score, summarize, ARMS


def test_presence_uses_frames_not_box_count():
    payload = {"detections": [{"detections": [{"class_name": "person"}]*3},
                              {"detections": [{"class_name": "person"}, {"class_name": "ball"}]}]}
    result = score(payload, ("person", "ball"))
    assert result["entity_coverage"] == 1
    assert result["mean_entity_presence"] == .75
    assert result["full_entity_frame_rate"] == .5
    assert result["yolo_entity_score"] == pytest.approx(.825)
    with pytest.raises(ValueError):
        score({"detections": []}, ("person",))


def test_empty_detections_are_zero_not_missing():
    result = score({"detections": [{"detections": []}]}, ("flower",))
    assert result["yolo_entity_score"] == 0
    assert result["sampled_frames"] == 1


def test_paired_summary_and_flower_exclusion():
    rows = []
    for arm in ARMS:
        for category in ("01", "04"):
            rows.append({"variant": arm, "video_id": category+"-001", "generation_seed": 0,
                         "category": category, **score({"detections": [{"detections": []}]}, ("person",))})
    summary, pairs = summarize(rows)
    assert {r["n"] for r in summary if r["group"] == "non_flower_categories"} == {1}
    assert all(r["left_minus_right"] == 0 for r in pairs)
    with pytest.raises(ValueError, match="Unmatched"):
        summarize(rows[:-1])
