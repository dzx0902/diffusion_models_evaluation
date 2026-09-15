from argparse import Namespace
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from ms_video_eval.c2_residual import (
    fit_pca, encode, decode, contrastive_loss, variance_covariance,
    retrieval, within_category_permutation,
)
from scripts.run_c2_residual import train, evaluate
import scripts.run_c2_residual as runner


def test_pca_train_only_roundtrip_and_error_decomposition():
    torch.manual_seed(7)
    train_x = torch.randn(14, 3, 8)
    test_x = torch.randn(4, 3, 8) + 8
    pca = fit_pca(train_x, 5)
    assert torch.allclose(pca["mean"], train_x.flatten(1).mean(0))
    assert torch.allclose(pca["basis"] @ pca["basis"].T, torch.eye(5), atol=1e-5)
    target = encode(test_x, pca)
    pred = torch.randn_like(target)
    floor = (test_x - decode(target, pca)).square().flatten(1).mean(1)
    decomposed = floor + ((pred - target) * pca["scale"]).square().sum(1) / 24
    measured = (test_x - decode(pred, pca)).square().flatten(1).mean(1)
    assert torch.allclose(decomposed, measured, atol=1e-4)
    assert torch.allclose(encode(decode(pred, pca), pca), pred, atol=1e-5)
    with pytest.raises(ValueError):
        fit_pca(torch.ones(10, 3, 8), 3)


def test_tied_constant_predictions_get_chance_not_sort_advantage():
    target = torch.eye(8)
    categories = torch.tensor([0] * 4 + [1] * 4)
    metrics = retrieval(torch.zeros_like(target), target, torch.eye(8).bool(), categories)
    assert metrics["global_r1"] == pytest.approx(1 / 8)
    assert metrics["global_r5"] == pytest.approx(5 / 8)
    assert metrics["within_category_r1"] == pytest.approx(1 / 4)
    assert metrics["within_category_r5"] == pytest.approx(1)
    assert metrics["global_mrr"] == pytest.approx(sum(1 / i for i in range(1, 9)) / 8)


def test_multi_positive_loss_and_metrics():
    target = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    positive = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    pred = target.clone().requires_grad_()
    aligned = contrastive_loss(pred, target, positive)
    wrong = contrastive_loss(pred.flip(1), target, positive)
    assert aligned < wrong
    aligned.backward()
    assert torch.isfinite(pred.grad).all()
    assert retrieval(target, target, positive, torch.zeros(3))["global_r1"] == pytest.approx(1)


def test_variance_penalty_and_category_shuffle():
    constant = torch.zeros(12, 4)
    diverse = torch.randn(12, 4) * 3
    assert variance_covariance(constant)[0] > variance_covariance(diverse)[0]
    categories = torch.tensor([0, 1, 0, 1, 0, 1])
    permutation = within_category_permutation(categories, torch.Generator().manual_seed(5))
    assert torch.equal(categories, categories[permutation])
    assert (permutation != torch.arange(6)).all()
    assert sorted(permutation.tolist()) == list(range(6))


def test_cpu_train_evaluate_and_completion_signature(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    targets = torch.randn(10, 3, 8)
    pca = fit_pca(targets[:6], 3)
    z = encode(targets, pca)
    package = {"z": z, "pca": pca, "eeg": torch.randn(10, 3, 62, 128),
               "splits": {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]},
               "caption_ids": torch.arange(10), "categories": torch.zeros(10, dtype=torch.long),
               "ids": [f"01-{i:03d}" for i in range(10)],
               "floor_mse": (targets - decode(z, pca)).square().flatten(1).mean(1),
               "raw_mean_mse": (targets.flatten(1) - pca["mean"]).square().mean(1)}
    args = Namespace(variant="variance", seed=42, epochs=1, batch_size=3, lr=0.001,
                     weight_decay=0.0001, dropout=0.1, temperature=0.1, patience=2,
                     min_epochs=1, device="cpu", resume=False, partition="test", shuffles=2, export=False)
    train(args, package, "fixture", tmp_path)
    assert (tmp_path / "completed.json").exists()
    evaluate(args, package, "fixture", tmp_path)
    report = json.loads((tmp_path / "test/report.json").read_text())
    assert report["video_count"] == 2
    assert report["checkpoint_epoch"] == 1
    assert len(report["shuffle"]["within_category_mrr"]) == 2
    with pytest.raises(ValueError, match="Prepared data differs"):
        evaluate(args, package, "different", tmp_path)
    args.lr = 0.1
    with pytest.raises(ValueError, match="settings differ"):
        train(args, package, "fixture", tmp_path)


def test_mean_export_keeps_full_tora_shape(tmp_path):
    pca = {"mean": torch.ones(226 * 512) * 0.1,
           "basis": torch.zeros(2, 226 * 512), "scale": torch.ones(2), "shape": (226, 512)}
    pca["basis"][0, 0] = 1
    pca["basis"][1, 1] = 1
    package = {"z": torch.zeros(2, 2), "pca": pca, "splits": {"test": [0, 1]},
               "ids": ["01-001", "01-002"], "caption_ids": torch.arange(2),
               "categories": torch.zeros(2, dtype=torch.long), "floor_mse": torch.zeros(2),
               "raw_mean_mse": torch.zeros(2),
               "token_projector": {"mean": torch.ones(4096) * 0.3,
                                   "components": torch.zeros(512, 4096)}}
    args = Namespace(variant="mean", partition="test", seed=42, shuffles=1, export=True)
    evaluate(args, package, "fixture", tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "test/video_index.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    payload = torch.load(rows[0]["condition_path"], weights_only=False)
    assert payload["hidden_state"].shape == (226, 4096)
    assert torch.allclose(payload["hidden_state"], torch.ones(226, 4096) * 0.3)


def test_interrupted_training_resumes_identically(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    targets = torch.randn(10, 3, 8)
    pca = fit_pca(targets[:6], 3)
    z = encode(targets, pca)
    package = {"z": z, "pca": pca, "eeg": torch.randn(10, 3, 62, 128),
               "splits": {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]},
               "caption_ids": torch.arange(10), "categories": torch.zeros(10, dtype=torch.long),
               "floor_mse": (targets - decode(z, pca)).square().flatten(1).mean(1),
               "raw_mean_mse": (targets.flatten(1) - pca["mean"]).square().mean(1)}
    args = Namespace(variant="variance", seed=42, epochs=2, batch_size=3, lr=0.001,
                     weight_decay=0.0001, dropout=0.1, temperature=0.1, patience=2,
                     min_epochs=1, device="cpu", resume=False)
    torch.manual_seed(42)
    train(args, package, "fixture", tmp_path / "continuous")
    save = runner.atomic_save

    def fail_after_checkpoint(payload, path):
        save(payload, path)
        if path.name == "last.pt" and payload["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "atomic_save", fail_after_checkpoint)
    torch.manual_seed(42)
    with pytest.raises(RuntimeError, match="simulated"):
        train(args, package, "fixture", tmp_path / "resumed")
    monkeypatch.setattr(runner, "atomic_save", save)
    args.resume = True
    train(args, package, "fixture", tmp_path / "resumed")
    a = torch.load(tmp_path / "continuous/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
