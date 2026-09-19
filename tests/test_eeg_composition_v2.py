import json

import pytest
import torch

from scripts import run_eeg_composition as base
from scripts import run_eeg_composition_v2 as v2
from ms_video_eval.eeg_composition import labels_for, split_videos


def test_development_excludes_triples_and_preserves_all_objects():
    ids = [f"{c:02d}-{i:03d}" for c in range(1,9) for i in range(1,79)]
    p = {"ids": ids, "labels": labels_for(ids), "splits": split_videos(ids)}
    for category in ("01", "02", "03", "04", "05", "06"):
        q = v2.development_data(p, category)
        assert len(q["splits"]["train"]) == 350
        assert len(q["splits"]["validation"]) == 78
        assert q["splits"]["test"] == []
        assert all(ids[i][:2] != category and int(ids[i][:2]) < 7 for i in q["splits"]["train"])
        assert (p["labels"][q["splits"]["train"]].sum(0) > 0).all()
    assert len(p["splits"]["test"]) == 156


def test_detailed_metrics_and_shuffle_are_video_level():
    labels = labels_for(["07-001"]*20 + ["08-001"]*20)
    logits = (labels*2-1)*5
    result = v2.detailed_metrics(logits, labels, .5, 3)
    assert result["per_object"]["person"]["topk_recall"] == 1
    assert result["per_object"]["car"]["topk_false_positive_rate"] == 0
    assert result["all_positive_baseline"]["micro_f1"] == pytest.approx(2/3)
    shuffled = v2.shuffle_audit(logits, labels, .5, 3)
    assert shuffled == v2.shuffle_audit(logits, labels, .5, 3)
    assert shuffled["matched_exact"] == 1
    assert .4 < shuffled["shuffled_exact_mean"] < .6


def test_windows_use_exact_offsets_and_average_logits(monkeypatch):
    from argparse import Namespace
    calls = []
    def infer(model, eeg, ids, sessions, samples, *args):
        calls.append((eeg.shape[-1], float(eeg[0,0,0,0]), samples))
        return torch.full((1,1,6), float(eeg[0,0,0,0]))
    monkeypatch.setattr(base, "infer", infer)
    p = {"eeg": torch.arange(1200).reshape(1,1,1,1200)}
    result = v2.predict_windows(None, p, [0], [0], None, None,
                               Namespace(device="cpu", batch_size=1), "6s_windows")
    assert calls == [(800,0.,800), (800,200.,800), (800,400.,800)]
    assert (result == 200).all()


def test_development_train_and_audit_with_constant_validation_labels(tmp_path):
    from argparse import Namespace
    torch.set_num_threads(1)
    ids = [f"{c:02d}-{i:03d}" for c in range(1,7) for i in range(1,4)]
    p = {"ids": ids, "labels": labels_for(ids), "eeg": torch.randn(18,3,62,128),
         "splits": {"train": [i for i in range(18) if i%3 != 2], "validation": list(range(2,18,3)), "test": []}}
    q = v2.development_data(p, "01")
    args = Namespace(protocol="cs_s3", variant="object_only", seed=42, epochs=1, batch_size=4,
                     lr=.001, device="cpu", resume=False, selection="top2_exact", shuffle_repeats=10)
    base.train(args, q, "fixture", tmp_path / "train")
    v2.audit(args, q, tmp_path / "train/best.pt", tmp_path / "audit", development=True)
    report = json.loads((tmp_path / "audit/audit.json").read_text())
    assert report["development_only"]
    assert report["validation"]["informative_macro_ap"] is None
    assert list(report["reports"]) == ["4s/session3"]


def test_triple_audit_preserves_checkpoint_and_writes_window_summary(tmp_path):
    from argparse import Namespace
    torch.set_num_threads(1)
    ids = [f"{c:02d}-001" for c in range(1,9)]
    p = {"ids": ids, "labels": labels_for(ids), "eeg": torch.randn(8,3,62,1200),
         "splits": {"train": list(range(6)), "validation": list(range(6)), "test": [6,7]}}
    args = Namespace(protocol="session_average", variant="original", seed=42, epochs=1,
                     batch_size=4, lr=.001, device="cpu", resume=False, shuffle_repeats=5)
    source = tmp_path / "baseline"
    destination = tmp_path / "new" / "run"
    base.train(args, p, "fixture", source)
    hashes = {f.name: base.digest(f) for f in source.iterdir() if f.is_file()}
    v2.audit(args, p, source / "best.pt", destination)
    assert hashes == {f.name: base.digest(f) for f in source.iterdir() if f.is_file()}
    report = json.loads((destination / "audit.json").read_text())
    assert len(report["reports"]) == 12
    assert report["reports"]["6s_windows/session_average"]["video_count"] == 2
    assert report["reports"]["4s/session1"]["by_category"]["07"]["video_count"] == 1
    v2.summarize(tmp_path / "new")
    assert (tmp_path / "new/summary.csv").is_file()
