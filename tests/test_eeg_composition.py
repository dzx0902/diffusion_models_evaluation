from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT / "src"):
    sys.path.insert(0,str(p))
from ms_video_eval.eeg_composition import (labels_for,split_videos,fit_normalization,metrics,
                                          select_threshold,PROTOCOLS)
from EEG2Caption.src.common import CompactEEGClassifier
import scripts.run_eeg_composition as runner


def test_split_is_composition_heldout_and_seed_stable():
    ids=[f"{c:02d}-{i:03d}" for c in range(1,9) for i in range(1,79)]
    splits=split_videos(ids)
    assert list(map(len,splits.values()))==[420,48,156]
    assert splits==split_videos(ids)
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert all(int(ids[i][:2])<7 for i in splits["train"]+splits["validation"])
    assert all(int(ids[i][:2])>=7 for i in splits["test"])
    with pytest.raises(ValueError): split_videos(ids[:-1])
    with pytest.raises(ValueError): split_videos(ids+[ids[0]])


def test_heldout_session_and_videos_do_not_affect_normalization():
    eeg=torch.randn(5,3,62,1200)
    mean,std=fit_normalization(eeg,[0,1],[0,1])
    changed=eeg.clone()
    changed[:,2]=1e8
    changed[2:]=1e8
    changed[:,:,:,800:]=1e8
    mean2,std2=fit_normalization(changed,[0,1],[0,1])
    assert torch.equal(mean,mean2) and torch.equal(std,std2)


def test_triple_metrics_do_not_claim_constant_label_ap():
    labels=labels_for(["07-001","08-001"])
    result=metrics((labels*2-1)*5,labels,.5)
    assert result["topk_metrics"]["exact_set_accuracy"]==1
    assert result["informative_macro_ap"]==1
    assert result["per_object"]["person"]["ap"] is None
    assert result["per_object"]["car"]["ap"] is None
    assert result["per_object"]["car"]["false_positive_rate"]==0
    assert .1 <= select_threshold((labels*2-1)*5,labels) <= .9


def fixture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    ids=[f"{c:02d}-{i:03d}" for c in range(1,7) for i in range(1,4)] + ["07-001","07-002","08-001","08-002"]
    p={"eeg":torch.randn(22,3,62,128),"labels":labels_for(ids),"ids":ids,
       "splits":{"train":[i for i in range(18) if i%3!=2],"validation":list(range(2,18,3)),"test":list(range(18,22))}}
    args=Namespace(protocol="cs_s3",variant="original",seed=42,epochs=2,batch_size=4,
                   lr=.001,device="cpu",resume=False)
    return p,args


@pytest.mark.parametrize("protocol",list(PROTOCOLS))
def test_training_evaluation_protocols(tmp_path,protocol):
    p,args=fixture(tmp_path)
    args.protocol=protocol
    runner.train(args,p,"fixture",tmp_path)
    runner.evaluate(args,p,"fixture",tmp_path)
    report=json.loads((tmp_path / "report.json").read_text())
    assert report["fixed_set_baselines"]["07"]["exact_set_accuracy"]==.5
    assert len(report["reports"])==(8 if protocol=="session_average" else 2)
    for key,r in report["reports"].items():
        assert r["video_count"]==4 and r["topk"]==3
        if protocol!="session_average": assert key.endswith("session"+protocol[-1])
    args.lr=.5
    with pytest.raises(ValueError): runner.train(args,p,"fixture",tmp_path)


def test_resume_matches_uninterrupted(tmp_path,monkeypatch):
    p,args=fixture(tmp_path)
    args.variant="object_only"
    runner.train(args,p,"fixture",tmp_path / "continuous")
    save=runner.atomic_save
    def interrupted(state,path):
        save(state,path)
        if path.name=="last.pt" and state["epoch"]==1: raise RuntimeError("interrupted")
    monkeypatch.setattr(runner,"atomic_save",interrupted)
    with pytest.raises(RuntimeError): runner.train(args,p,"fixture",tmp_path / "resumed")
    monkeypatch.setattr(runner,"atomic_save",save)
    args.resume=True
    runner.train(args,p,"fixture",tmp_path / "resumed")
    a,b=runner.load(tmp_path / "continuous/last.pt"),runner.load(tmp_path / "resumed/last.pt")
    assert a["history"]==b["history"]
    assert all(torch.equal(a["model"][k],b["model"][k]) for k in a["model"])


def test_original_compact_supports_both_durations():
    torch.set_num_threads(1)
    model=CompactEEGClassifier().eval()
    with torch.no_grad():
        for n in (800,1200):
            out=model(torch.randn(2,3,62,n))
            assert out["session_object_logits"].shape==(2,3,6)
            assert torch.allclose(out["fused_object_logits"],out["session_object_logits"].mean(1))


def test_test_data_and_heldout_session_cannot_change_training(tmp_path):
    p,args=fixture(tmp_path)
    args.epochs=1
    runner.train(args,p,"fixture",tmp_path / "a")
    p["eeg"][:,2]=1e7
    p["eeg"][p["splits"]["test"]]=1e7
    p["labels"][p["splits"]["test"]]=0
    runner.train(args,p,"fixture",tmp_path / "b")
    a,b=runner.load(tmp_path / "a/last.pt"),runner.load(tmp_path / "b/last.pt")
    assert a["history"]==b["history"]
    assert all(torch.equal(a["model"][k],b["model"][k]) for k in a["model"])
