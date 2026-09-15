import sys
from pathlib import Path
from argparse import Namespace
import json

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from ms_video_eval.c2_residual import SessionResidualEEG, normalize_sessions, fit_pca, encode, decode
from scripts.run_c2_sessions import ridge_fit, run
from scripts.run_c2_residual import train


def test_holdout_normalization_does_not_fit_session3_or_test():
    package = {"eeg": torch.randn(10, 3, 62, 128), "normalization_mean": torch.zeros(3, 62),
               "normalization_std": torch.ones(3, 62), "splits": {"train": list(range(6))}}
    a = normalize_sessions(package, [0, 1])
    altered = dict(package, eeg=package["eeg"].clone())
    altered["eeg"][:, 2] += 1000
    altered["eeg"][6:] -= 2000
    b = normalize_sessions(altered, [0, 1])
    assert torch.equal(a["shared_mean"], b["shared_mean"])
    assert torch.equal(a["shared_std"], b["shared_std"])
    assert torch.equal(a["eeg"][:6, :2], b["eeg"][:6, :2])


def test_single_session_predictions_are_independent_at_inference():
    torch.set_num_threads(1)
    model = SessionResidualEEG(4).eval()
    eeg = torch.randn(3, 3, 62, 128)
    with torch.no_grad():
        individual = torch.stack([model(eeg[:, i:i+1]) for i in range(3)], 1)
        assert torch.allclose(individual, model.session_predictions(eeg), atol=1e-6)
        assert torch.allclose(model(eeg), individual.mean(1), atol=1e-6)


def test_ridge_dual_equals_primal():
    x, y = torch.randn(8, 12).double(), torch.randn(8, 3).double()
    result = ridge_fit(x, y, 0.1)
    xc, yc = x - x.mean(0), y - y.mean(0)
    expected = torch.linalg.solve(xc.T @ xc + 0.1 * 12 * torch.eye(12), xc.T @ yc)
    assert torch.allclose(result["weights"], expected, atol=1e-8)


def test_session_training_and_ridge_report(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    targets = torch.randn(10, 3, 8)
    pca = fit_pca(targets[:6], 3)
    z = encode(targets, pca)
    package = {"z": z, "pca": pca, "eeg": torch.randn(10, 3, 62, 128),
               "splits": {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]},
               "caption_ids": torch.arange(10), "categories": torch.zeros(10, dtype=torch.long),
               "ids": [str(i) for i in range(10)], "normalization_mean": torch.zeros(3, 62),
               "normalization_std": torch.ones(3, 62),
               "floor_mse": (targets - decode(z, pca)).square().flatten(1).mean(1),
               "raw_mean_mse": (targets.flatten(1) - pca["mean"]).square().mean(1)}
    prepared = tmp_path / "prepared.pt"
    torch.save(package, prepared)
    args = Namespace(prepared=prepared, output_root=tmp_path / "runs", protocol="holdout_s3",
                     variant="consistent", seed=42, epochs=1, batch_size=3, lr=0.001,
                     weight_decay=0.0001, dropout=0.1, temperature=0.1, patience=2,
                     min_epochs=1, device="cpu", resume=False, stage="train", shuffles=1)
    run(args)
    args.stage = "evaluate"
    run(args)
    report = json.loads((args.output_root / "holdout_s3/consistent/seed42/report.json").read_text())
    assert report["reports"]["train/session3"]["heldout_session"]
    assert report["reports"]["train/session3"]["seen_videos"]
    assert not report["reports"]["test/session3"]["seen_videos"]
    args.variant, args.stage = "ridge", "train"
    run(args)
    args.stage = "evaluate"
    run(args)
    report = json.loads((args.output_root / "holdout_s3/ridge/seed42/report.json").read_text())
    assert report["selected_alpha"] in report["protocol"]["ridge_alphas"]
