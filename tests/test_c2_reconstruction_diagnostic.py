from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))
import scripts.run_c2_reconstruction_diagnostic as runner


def package():
    ids = [f"{c:02d}-{i:03d}" for c in range(1, 7) for i in range(4)]
    return {"ids": ids, "captions": [f"caption {v}" for v in ids],
            "splits": {"train": [i for i in range(24) if i % 4 in (1, 2)],
                       "validation": list(range(0, 24, 4)), "test": list(range(3, 24, 4))}}


def test_selection_is_train_only_balanced_and_swaps_same_category():
    p = package()
    indices = runner.select_pairs(p)
    assert len(indices) == 12
    assert set(indices).issubset(p["splits"]["train"])
    for j, i in enumerate(indices):
        donor = indices[j ^ 1]
        assert donor != i
        assert p["ids"][donor][:2] == p["ids"][i][:2]
        assert p["captions"][donor] != p["captions"][i]
    p["captions"][2] = p["captions"][1]
    with pytest.raises(ValueError, match="different train captions"):
        runner.select_pairs(p)


def inputs(tmp_path):
    p = package()
    selection = [{"video_id": p["ids"][i]} for i in runner.select_pairs(p)]
    static = tmp_path / "shared_static.txt"
    static.write_text("128,128\n"*49)
    protocol = {"selection": selection, "assets": {static.name: runner.digest(static)},
                "fold": "video_6fold_1", "training_seed": 42}
    runner.atomic_json(protocol, tmp_path / "protocol.json")
    for arm in runner.ARMS:
        runner.write_jsonl([{"video_id": row["video_id"], "condition_path": str(tmp_path / arm / (row["video_id"]+".pt"))}
                            for row in selection], tmp_path / "inputs" / f"{arm}.jsonl")
    return Namespace(output_root=tmp_path, generator_config=ROOT / "configs/eeg_semantic/generators.server.yaml",
                     models_root=tmp_path / "models", seeds=[0], dry_run=True)


def test_dry_run_has_48_matched_jobs_and_preserves_real_manifest(tmp_path):
    args = inputs(tmp_path)
    existing = tmp_path / "generated/text_full/generation_manifest.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing-real-result")
    runner.generate(args)
    assert existing.read_text() == "existing-real-result"
    assert not (tmp_path / "generation_protocol.json").exists()
    controls = []
    for arm in runner.ARMS:
        rows = [json.loads(line) for line in (tmp_path / f"generated/{arm}/dry_run_manifest.jsonl").read_text().splitlines()]
        assert len(rows) == 12
        for row in rows:
            assert row["diagnostic_only"] and row["partition"] == "train"
            assert row["trajectory_origin"] == "synthetic_shared_static"
            command = row["command"]
            assert command[command.index("--conditioning")+1] == "injected"
            assert command[command.index("--seed")+1] == "42"
            assert command[command.index("--num-frames")+1] == "49"
            assert command[command.index("--fps")+1] == "12"
        controls.append([(row["video_id"], row["trajectory_sha256s"], row["generation_seed"]) for row in rows])
    assert all(c == controls[0] for c in controls)


def test_changed_controls_rejected(tmp_path):
    args = inputs(tmp_path)
    (tmp_path / "shared_static.txt").write_text("0,0\n")
    with pytest.raises(ValueError, match="changed or missing"):
        runner.generate(args)


def test_generation_settings_lock(tmp_path):
    args = inputs(tmp_path)
    runner.atomic_json({"old": "different-settings"}, tmp_path / "generation_protocol.json")
    with pytest.raises(ValueError, match="settings changed"):
        runner.generate(args)


def test_prepare_controls_and_swap_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    torch.set_num_threads(1)
    p = package()
    p["z"] = torch.arange(48).float().reshape(24, 2)
    basis = torch.zeros(2, 226*512)
    basis[:, :2] = torch.eye(2)
    p["pca"] = {"mean": torch.zeros(226*512), "basis": basis,
                "scale": torch.ones(2), "shape": (226, 512)}
    components = torch.zeros(512, 4096)
    components[:2, :2] = torch.eye(2)
    p["token_projector"] = {"mean": torch.zeros(4096), "components": components}
    p["protocol"] = {"fold": "video_6fold_1"}
    args = Namespace(prepared=tmp_path / "prepared.pt", checkpoint=tmp_path / "best.pt",
                     text_index=tmp_path / "index.jsonl", output_root=tmp_path / "diagnostic", device="cpu")
    args.prepared.write_text("prepared-fixture")
    args.checkpoint.write_text("checkpoint-fixture")
    args.text_index.write_text("text-index-fixture")
    signature = {"variant": "long", "prepared_sha256": runner.digest(args.prepared), "seed": 42}
    args.checkpoint.with_name("completed.json").write_text(json.dumps({"signature": signature}))
    checkpoint = {"signature": signature, "model_config": {}, "model": {}, "step": 250}
    monkeypatch.setattr(runner, "load", lambda path: p if path == args.prepared else checkpoint)
    class Model:
        def to(self, device):
            return self
        def load_state_dict(self, state):
            pass
    monkeypatch.setattr(runner, "DeepC2", Model)
    monkeypatch.setattr(runner, "normalize_sessions", lambda value, sessions: value)
    monkeypatch.setattr(runner, "predictions", lambda model, value, ids, device, batch: value["z"][ids]+100)
    monkeypatch.setattr(runner, "read_tora_condition_index", lambda path: {
        v: {"condition_path": v} for v in p["ids"]})
    monkeypatch.setattr(runner, "load_tora_condition", lambda path: SimpleNamespace(
        video_id=str(path), caption=p["captions"][p["ids"].index(str(path))],
        hidden_state=torch.full((226,4096), -1.)))
    real_save = runner.atomic_save
    # Check the full contract, then keep test disk usage small.
    def save_small(payload, path):
        assert payload["hidden_state"].shape == (226,4096)
        real_save({**payload, "hidden_state": payload["hidden_state"][:1,:2].clone()}, path)
    monkeypatch.setattr(runner, "atomic_save", save_small)
    runner.prepare(args)
    protocol = json.loads((args.output_root / "protocol.json").read_text())
    assert len(protocol["selection"]) == 12
    runner.verify_assets(args.output_root, protocol)
    for row in protocol["selection"]:
        video, donor = row["video_id"], row["swapped_source_video_id"]
        matched = torch.load(args.output_root / f"conditions/eeg_matched/{donor}.pt", weights_only=False)
        swapped = torch.load(args.output_root / f"conditions/eeg_swapped/{video}.pt", weights_only=False)
        assert torch.equal(matched["hidden_state"], swapped["hidden_state"])
        assert swapped["source_video_id"] == donor
    runner.prepare(args)  # Same protocol is verified and reused, not regenerated.
