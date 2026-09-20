import json

import pytest
import torch

from scripts.evaluate_eeg_triple_top3 import (OBJECTS, labels_for, validate_prediction,
                                             score, sources, evaluate, markdown)


def payload():
    ids = [f"{c:02d}-{v:03d}" for c in (7,8) for v in range(1,79)]
    y = labels_for(ids)
    return {"video_ids":ids,"labels":y,"logits":y*8-4,"object_names":OBJECTS}


def test_fixed_three_includes_person_and_rejects_car():
    ids, logits, labels = validate_prediction(payload())
    r = score(ids,logits,labels,repeats=100)
    assert r["correct"] == 156 and r["subject_recall"] == 1
    assert r["per_object"]["person"]["recall"] == 1
    assert r["per_object"]["car"]["false_positive_rate"] == 0
    assert .4 < r["shuffle"]["mean_exact"] < .6
    assert "threshold" not in json.dumps(r)
    assert r == score(ids,logits,labels,repeats=100)


def test_constant_combination_has_no_shuffle_gain():
    p = payload()
    p["logits"][:] = p["logits"][0].clone()
    r = score(*validate_prediction(p), repeats=30)
    assert r["exact_accuracy"] == .5
    assert r["shuffle"]["matched_minus_shuffled"] == 0


@pytest.mark.parametrize("fault", ["duplicate", "labels", "order", "nan"])
def test_invalid_inputs_rejected(fault):
    p = payload()
    if fault == "duplicate": p["video_ids"][1] = p["video_ids"][0]
    if fault == "labels": p["labels"][0,0] = 0
    if fault == "order": p["object_names"] = tuple(reversed(OBJECTS))
    if fault == "nan": p["logits"][0,0] = float("nan")
    with pytest.raises(ValueError): validate_prediction(p)


def test_full_matrix_is_read_only_and_no_threshold_selection(tmp_path):
    for variant, protocol, mode, audit_path, pred_path in sources(tmp_path,"chentianlin",42):
        pred_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload(),pred_path)
        audit_path.write_text(json.dumps({"development_only":False,
             "checkpoint_epoch":1,"checkpoint_sha256":"checkpoint",
             "signature":{"protocol":protocol,"variant":variant,"seed":42,
                          "prepared_sha256":"same-data",
                          "selection":"01-06 validation allowed-session mean object macro AP"}}))
    before = {str(p):p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = evaluate(tmp_path,"chentianlin",42,10)
    assert len(report["results"]) == 16
    assert report["protocol"]["threshold_used"] is False
    assert "主体召回率" in markdown(report)
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
