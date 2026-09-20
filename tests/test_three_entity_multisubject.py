import json
from argparse import Namespace

import pytest
import torch

from scripts import run_three_entity_multisubject as runner


def rows(value):
    return [{"variant":variant,"protocol":protocol,
             "metrics":{key:value+(0 if protocol=="session_average" else .1) for key in runner.KEYS}}
            for variant in runner.VARIANTS for protocol in runner.base.PROTOCOLS]


def test_subject_sample_sd_not_session_sd():
    per_subject, result = runner.group_statistics({"a":rows(.2),"b":rows(.4)},["a","b"])
    assert len(per_subject) == 2
    assert result[0]["mean"]["set_accuracy"] == pytest.approx(.3)
    assert result[0]["std"]["set_accuracy"] == pytest.approx(.2/(2**.5))
    assert result[2]["mean"]["set_accuracy"] == pytest.approx(.4)
    assert result[2]["std"]["set_accuracy"] == pytest.approx(.2/(2**.5))
    assert all(r["n_subjects"] == 2 for r in result)
    assert "sample standard deviation" in runner.table(result,2)


def test_missing_subject_or_direction_fails():
    with pytest.raises(ValueError): runner.group_statistics({"a":rows(.2)},["a","b"])
    with pytest.raises(ValueError): runner.group_statistics({"a":rows(.2),"b":rows(.4)[:-1]},["a","b"])


def test_fixed_top3_worker_smoke(tmp_path):
    torch.set_num_threads(1)
    ids = [f"{c:02d}-{v:03d}" for c in range(1,9) for v in range(1,79)]
    splits = runner.base.split_videos(ids,42)
    # Small training/validation fixture while preserving full test IDs.
    splits["train"] = [0,78,156,234,312,390]
    splits["validation"] = [1,79,157,235,313,391]
    p = {"ids":ids,"labels":runner.base.labels_for(ids),"eeg":torch.randn(624,3,62,1200),
         "splits":splits,"protocol":{"subject":"fixture","split_seed":42}}
    # Avoid a large on-disk source fixture; production split check is tested separately.
    from unittest.mock import patch
    prepared = tmp_path / "prepared/fixture.pt"
    prepared.parent.mkdir()
    prepared.write_bytes(b"fixture")
    original_load = runner.base.load
    def load(path):
        return p if path == prepared else original_load(path)
    args = Namespace(subjects=["fixture"],output_root=tmp_path,protocol="cs_s3",variant="object_only",
                     seed=42,epochs=1,batch_size=4,lr=.001,device="cpu",resume=False)
    with patch.object(runner.base,"load",load), patch.object(runner.base,"split_videos",return_value=splits):
        runner.worker(args)
    r = json.loads((tmp_path / "runs/fixture/cs_s3/object_only/seed42/test_top3_metrics.json").read_text())
    assert r["n"] == 156 and r["threshold_used"] is False
    assert set(r["metrics"]) >= set(runner.KEYS)
