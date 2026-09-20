import json

import pytest
import torch

from scripts.build_three_entity_paper_table import paper_metrics, aggregate, latex, collect, section, KEYS
from scripts.evaluate_eeg_triple_top3 import sources, labels_for, OBJECTS


def test_perfect_scores_and_macro_scope():
    y = labels_for(["07-001", "08-001"])
    m = paper_metrics(y*10-5,y)
    assert all(m[k] == 1 for k in KEYS)
    assert m["macro_entities"] == ["dog", "ball", "flower", "bird"]
    assert "person" in m["micro_and_set_entities"]
    assert "car" in m["micro_and_set_entities"]


def test_jaccard_is_mean_of_per_video_ratios():
    y = labels_for(["07-001", "08-001"])
    logits = y*10-5
    # First perfect, second disjoint: mean Jaccard .5, not pooled intersection/union 1/3.
    logits[1] *= -1
    m = paper_metrics(logits,y)
    assert m["jaccard"] == .5
    assert m["top3_f1"] == .5
    assert m["set_accuracy"] == .5


def test_cross_session_means_and_short_table():
    rows = [{"variant":variant,"protocol":protocol,"metrics":{k:value for k in KEYS}}
            for variant in ("original","object_only")
            for protocol,value in (("session_average",.9),("cs_s1",.1),("cs_s2",.2),("cs_s3",.3))]
    summary = aggregate(rows)
    assert len(summary) == 4
    assert summary[-1]["metrics"]["set_accuracy"] == pytest.approx(.2)
    text = latex({"summary":summary})
    assert text.count("Original &") == 2
    assert text.count("Object-only &") == 2
    assert "resizebox" not in text and "shuffle" not in text
    with pytest.raises(ValueError): aggregate(rows[:-1])


def test_collect_does_not_require_windows_or_modify_predictions(tmp_path):
    ids = [f"{c:02d}-{i:03d}" for c in (7,8) for i in range(1,79)]
    y = labels_for(ids)
    for variant,protocol,mode,audit,pred in sources(tmp_path,"chentianlin",42):
        if mode != "6s": continue
        pred.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"video_ids":ids,"labels":y,"logits":y*10-5,"object_names":OBJECTS},pred)
        audit.write_text(json.dumps({"development_only":False,"checkpoint_epoch":1,"checkpoint_sha256":"test",
          "signature":{"protocol":protocol,"variant":variant,"seed":42,"prepared_sha256":"same",
          "selection":"01-06 validation allowed-session mean object macro AP"}}))
    before = {str(p):p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    r = collect(tmp_path,"chentianlin",42)
    assert len(r["rows"]) == 8 and len(r["summary"]) == 4
    assert "\\section{Extension" in section(r)
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
