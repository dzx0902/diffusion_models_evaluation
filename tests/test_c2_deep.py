from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from ms_video_eval.c2_deep import (DeepC2, temporal_visual_loss, learning_rate,
                                 validate_splits, validate_donor_ids)
from ms_video_eval.c2_residual import fit_pca, encode, decode
import scripts.run_c2_deep as runner


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    targets = torch.randn(10, 3, 8)
    pca = fit_pca(targets[:6], 3)
    z = encode(targets, pca)
    return {"ids": [f"01-{i:03d}" for i in range(10)], "z": z, "pca": pca,
            "eeg": torch.randn(10, 3, 62, 128), "caption_ids": torch.arange(10),
            "categories": torch.zeros(10, dtype=torch.long),
            "splits": {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]},
            "floor_mse": (targets-decode(z, pca)).square().flatten(1).mean(1),
            "raw_mean_mse": (targets.flatten(1)-pca["mean"]).square().mean(1),
            "shared_mean": torch.zeros(62), "shared_std": torch.ones(62)}


def arguments(path):
    return Namespace(output_root=path, seed=42, updates=4, pretrain_updates=2,
                     warmup=1, lr=.001, batch_size=3, width=16, layers=1, frames=2,
                     dropout=.1, eval_every=2, log_every=1, target_subject="chentianlin",
                     resume=False, device="cpu", partition="test", shuffles=2, export=False)


def test_model_shapes_and_temporal_supervision():
    torch.set_num_threads(1)
    model = DeepC2(dim=3, width=16, layers=1, subjects=2, frames=4)
    result = model(torch.randn(2, 3, 62, 128), torch.tensor([0, 1]))
    assert result["z"].shape == (2, 3)
    assert result["sessions"].shape == (2, 3, 3)
    assert result["visual"].shape == (2, 4, 512)
    target = torch.randn(2, 4, 512)
    assert temporal_visual_loss(target, target) < 1e-6
    assert temporal_visual_loss(target.flip(1), target) > .5
    result["z"].square().mean().backward()
    assert model.subject_scale.weight.grad is not None


def test_shared_init_is_identical_across_subject_counts():
    torch.manual_seed(42)
    a = DeepC2(width=16, layers=1, subjects=1)
    torch.manual_seed(42)
    b = DeepC2(width=16, layers=1, subjects=5)
    for key, value in a.state_dict().items():
        if not key.startswith("subject_"):
            assert torch.equal(value, b.state_dict()[key]), key


def test_train_only_guard_and_update_schedule():
    package = fixture()
    validate_splits(package)
    train_ids = package["ids"][:6]
    validate_donor_ids(train_ids[::-1], package)
    with pytest.raises(ValueError, match="TRAIN"):
        validate_donor_ids(train_ids + [package["ids"][-1]], package)
    with pytest.raises(ValueError, match="TRAIN"):
        validate_donor_ids(train_ids[:-1], package)
    package["splits"]["test"].append(0)
    with pytest.raises(ValueError, match="overlapping"):
        validate_splits(package)
    assert learning_rate(1, 20, 4, 1.) == pytest.approx(.25)
    assert learning_rate(4, 20, 4, 1.) == pytest.approx(1.)
    assert learning_rate(20, 20, 4, 1.) == pytest.approx(.05)


def add_resources(args, package):
    args.output_root.mkdir(parents=True, exist_ok=True)
    runner.atomic_save({"signature": {"prepared_sha256": "fixture"}, "ids": package["ids"],
                        "visual": torch.randn(10, 2, 512), "text": torch.randn(10, 512)}, args.output_root / "visual.pt")
    folder = args.output_root / "donors"
    folder.mkdir()
    path = folder / "donor.pt"
    # Reverse ordering checks explicit ID mapping, not accidental tensor indexing.
    runner.atomic_save({"ids": package["ids"][:6][::-1], "prepared_sha256": "fixture",
                        "eeg": package["eeg"][:6].flip(0)}, path)
    runner.atomic_json({"prepared_sha256": "fixture", "target_subject": "chentianlin",
                        "accepted": [{"file": path.name, "sha256": runner.digest(path)}]}, folder / "audit.json")


@pytest.mark.parametrize("variant", ["long", "joint", "multisubject"])
def test_fixed_budget_and_evaluation(tmp_path, variant):
    package, args = fixture(), arguments(tmp_path)
    if variant != "long":
        add_resources(args, package)
    runner.train(args, package, "fixture", variant)
    directory = tmp_path / variant / "seed42"
    completed = json.loads((directory / "completed.json").read_text())
    assert completed["optimizer_updates"] == 4
    last = runner.load(directory / "last.pt")
    assert last["step"] == 4
    assert max(int(s["step"]) for s in last["optimizer"]["state"].values()) == 4
    assert len(last["history"]) == 2
    if variant == "multisubject":
        assert runner.load(directory / "best.pt")["step"] > args.pretrain_updates
    runner.evaluate(args, package, "fixture", variant)
    report = json.loads((directory / "test/report.json").read_text())
    assert report["video_count"] == 2
    assert report["shuffle"]["count"] == 2
    args.lr = .5
    with pytest.raises(ValueError, match="settings differ"):
        runner.train(args, package, "fixture", variant)


@pytest.mark.parametrize("variant", ["long", "joint", "multisubject"])
def test_exact_resume(tmp_path, monkeypatch, variant):
    package = fixture()
    args = arguments(tmp_path / "continuous")
    if variant != "long":
        add_resources(args, package)
    runner.train(args, package, "fixture", variant)
    save = runner.atomic_save

    def interrupted(state, path):
        save(state, path)
        if path.name == "last.pt" and state["step"] == 2:
            raise RuntimeError("simulated interruption")

    args.output_root = tmp_path / "resumed"
    if variant != "long":
        torch.manual_seed(3)
        # Reuse byte-identical resources so provenance is identical at resume.
        import shutil
        shutil.copytree(tmp_path / "continuous", args.output_root,
                        ignore=shutil.ignore_patterns(variant))
    monkeypatch.setattr(runner, "atomic_save", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.train(args, package, "fixture", variant)
    monkeypatch.setattr(runner, "atomic_save", save)
    args.resume = True
    runner.train(args, package, "fixture", variant)
    a = runner.load(tmp_path / f"continuous/{variant}/seed42/last.pt")
    b = runner.load(tmp_path / f"resumed/{variant}/seed42/last.pt")
    assert a["history"] == b["history"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k


def test_test_targets_never_affect_training(tmp_path):
    package = fixture()
    args = arguments(tmp_path / "a")
    add_resources(args, package)
    runner.train(args, package, "fixture", "joint")
    package["z"][package["splits"]["test"]] = 1e6
    args.output_root = tmp_path / "b"
    args.output_root.mkdir()
    resources = runner.load(tmp_path / "a/visual.pt")
    resources["visual"][8:] = -1e6
    resources["text"][8:] = -1e6
    runner.atomic_save(resources, args.output_root / "visual.pt")
    runner.train(args, package, "fixture", "joint")
    a = runner.load(tmp_path / "a/joint/seed42/last.pt")
    b = runner.load(tmp_path / "b/joint/seed42/last.pt")
    assert a["history"] == b["history"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k])


def test_donor_prepare_excludes_heldout_from_data_and_statistics(tmp_path, monkeypatch):
    package = fixture()
    args = arguments(tmp_path / "output")
    args.eeg_root = tmp_path / "eeg"
    args.subjects = ["donor"]
    channels = [f"C{i}" for i in range(62)]
    signal = np.random.default_rng(42).normal(size=(10, 62, 800)).astype(np.float32)
    signal[6:] = 1e6  # These validation/test videos must not influence normalization.
    for subject in ("chentianlin", "donor"):
        for session in (1, 2, 3):
            folder = args.eeg_root / subject / f"session{session}/EEG"
            folder.mkdir(parents=True)
            np.savez(folder / "eeg_data.npz", channel_names=channels,
                     eeg=signal, sfreq=200.)

    def mapping(folder):
        path = folder / "EEG/eeg_data.npz"
        return folder.name, {video: {"npz_path": str(path), "metadata_path": str(path),
                                    "trial_index": i, "sfreq": 200., "length_samples": 800}
                             for i, video in enumerate(package["ids"])}

    monkeypatch.setattr(runner, "load_session", mapping)
    runner.prepare_donors(args, package, "fixture")
    donor = runner.load(args.output_root / "donors/donor.pt")
    assert donor["ids"] == package["ids"][:6]
    assert donor["eeg"].shape == (6, 3, 62, 800)
    assert donor["mean"].abs().max() < 1e5
    assert donor["eeg"].mean().abs() < 1e-5
    assert donor["eeg"].std() == pytest.approx(1., abs=.001)


def test_export_contract(tmp_path):
    package, args = fixture(), arguments(tmp_path)
    runner.train(args, package, "fixture", "long")
    # Replace only the synthetic PCA decoder to exercise the production 226x4096 contract.
    basis = torch.zeros(3, 226*512)
    basis[:, :3] = torch.eye(3)
    package["pca"] = {"mean": torch.zeros(226*512), "basis": basis,
                      "scale": torch.ones(3), "shape": (226, 512)}
    package["token_projector"] = {"mean": torch.ones(4096)*.3, "components": torch.zeros(512, 4096)}
    args.export = True
    runner.evaluate(args, package, "fixture", "long")
    rows = [json.loads(line) for line in (tmp_path / "long/seed42/test/video_index.jsonl").read_text().splitlines()]
    hidden = runner.load(rows[0]["condition_path"])["hidden_state"]
    assert hidden.shape == (226, 4096)
    assert torch.isfinite(hidden).all()
