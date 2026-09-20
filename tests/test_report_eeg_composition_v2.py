import json

import pytest

from scripts.report_eeg_composition_v2 import development_details, read_report, test_details as print_test_details


def make_report(root, category, variant, value):
    path = root / "development" / f"held_{category}" / "chentianlin/session_average" / variant / "seed42/audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": {"protocol": "session_average", "variant": variant, "seed": 42},
                               "development_only": True, "checkpoint_epoch": 3,
                               "reports": {"4s/session_average": {"video_count": 78,
                                           "topk_metrics": {"exact_set_accuracy": value, "micro_recall": .6}}}}))
    return path


def test_incomplete_development_not_reported_as_complete(tmp_path, capsys):
    make_report(tmp_path, "01", "original", .1)
    make_report(tmp_path, "01", "object_only", .2)
    development_details(tmp_path, "chentianlin", 42, ["session_average"])
    output = capsys.readouterr().out
    assert "reports=2/12" in output
    assert "INCOMPLETE" in output
    assert "six-category macro exact" not in output


def test_complete_development_aggregates_directions(tmp_path, capsys):
    for i in range(1,7):
        make_report(tmp_path, f"{i:02d}", "original", .1)
        make_report(tmp_path, f"{i:02d}", "object_only", .2)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    development_details(tmp_path, "chentianlin", 42, ["session_average"])
    output = capsys.readouterr().out
    assert "reports=12/12" in output
    assert "object_only=0.2000" in output
    assert "wins/ties/losses=6/0/0" in output
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_report_identity_checked(tmp_path):
    path = make_report(tmp_path, "01", "original", .1)
    with pytest.raises(ValueError, match="identity mismatch"):
        read_report(path, "session_average", "original", 43, True)


def test_test_report_includes_person_car_categories_and_null_quantiles(tmp_path, capsys):
    import torch
    from ms_video_eval.eeg_composition import labels_for
    from scripts.run_eeg_composition_v2 import detailed_metrics, shuffle_audit
    y = labels_for(["07-001", "08-001"])
    logits = y * 10 - 5
    result = detailed_metrics(logits, y, .5, 3)
    result["shuffle"] = shuffle_audit(logits, y, .5, 3, repeats=10)
    result["by_category"] = {c: detailed_metrics(logits[i:i+1], y[i:i+1], .5, 3)
                             for i, c in enumerate(("07", "08"))}
    path = tmp_path / "audit-baseline/chentianlin/session_average/original/seed42/audit.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"signature": {"protocol": "session_average", "variant": "original", "seed": 42},
                               "development_only": False, "checkpoint_epoch": 1, "threshold": .5,
                               "reports": {f"{mode}/session_average": result for mode in ("4s", "6s", "6s_windows")}}))
    print_test_details(tmp_path, "chentianlin", 42, ["session_average"])
    output = capsys.readouterr().out
    for phrase in ("person", "car", "class 07", "class 08", "null_q025", "files found: 1/2"):
        assert phrase in output
