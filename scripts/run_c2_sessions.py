"""C2 shared single-session prediction and held-out-session diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(value))

from ms_video_eval.c2_residual import SessionResidualEEG, normalize_sessions, within_category_permutation
from scripts.run_c2_residual import train, measure, predict, digest, atomic_json, atomic_save


def features(eeg):
    return F.adaptive_avg_pool1d(eeg.flatten(0, 1), 25).flatten(1).double()


def ridge_fit(x, y, alpha):
    xm, ym = x.mean(0), y.mean(0)
    centered = x - xm
    kernel = centered @ centered.T / x.shape[1]
    dual = torch.linalg.solve(kernel + alpha * torch.eye(len(x), dtype=x.dtype), y - ym)
    return {"xm": xm, "ym": ym, "weights": centered.T @ dual / x.shape[1], "alpha": alpha}


def ridge_predict(fit, eeg):
    values = (features(eeg) - fit["xm"]) @ fit["weights"] + fit["ym"]
    return values.float().reshape(len(eeg), eeg.shape[1], -1).mean(1)


def run(args):
    package = torch.load(args.prepared, map_location="cpu", weights_only=False)
    fingerprint = digest(args.prepared)
    sessions = [0, 1, 2] if args.protocol == "all_sessions" else [0, 1]
    package = normalize_sessions(package, sessions)
    fitting = dict(package, eeg=package["eeg"][:, sessions])
    output = args.output_root / args.protocol / args.variant / f"seed{args.seed}"
    args.session_mode = True
    args.session_protocol = args.protocol
    args.consistency_weight = 0.1 if args.variant == "consistent" else 0.0
    output.mkdir(parents=True, exist_ok=True)
    signature = {"prepared_sha256": fingerprint, "protocol": args.protocol,
                 "variant": args.variant, "seed": args.seed, "normalization_fit_sessions": sessions,
                 "ridge_alphas": [0.001, 0.01, 0.1, 1., 10., 100.], "ridge_time_bins": 25}
    atomic_json(signature, output / "protocol.json")
    if args.stage == "train":
        if args.variant != "ridge":
            train(args, fitting, fingerprint, output)
            return
        if (output / "completed.json").exists():
            saved = torch.load(output / "best.pt", weights_only=False)
            if saved["signature"] != signature:
                raise ValueError("Ridge configuration differs; use a new output root")
            print(f"[c2-sessions] ridge already completed {output}")
            return
        ti, vi = package["splits"]["train"], package["splits"]["validation"]
        x = features(fitting["eeg"][ti])
        y = package["z"][ti, None].expand(-1, len(sessions), -1).flatten(0, 1).double()
        best, rows = -float("inf"), []
        for alpha in signature["ridge_alphas"]:
            fit = ridge_fit(x, y, alpha)
            metrics = measure(ridge_predict(fit, fitting["eeg"][vi]), package, vi)
            rows.append({"alpha": alpha, "validation": metrics})
            if metrics["within_category_mrr"] > best:
                best = metrics["within_category_mrr"]
                atomic_save({"fit": fit, "signature": signature, "validation": metrics}, output / "best.pt")
        atomic_json(rows, output / "validation_grid.json")
        atomic_json({"signature": signature, "status": "complete"}, output / "completed.json")
        return
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["signature"]["prepared_sha256"] != fingerprint:
        raise ValueError("Prepared data changed")
    checkpoint_protocol = checkpoint["signature"].get("session_protocol", checkpoint["signature"].get("protocol"))
    if checkpoint_protocol != args.protocol:
        raise ValueError("Checkpoint session protocol differs")
    if args.variant == "ridge":
        inference = lambda eeg: ridge_predict(checkpoint["fit"], eeg)
    else:
        model = SessionResidualEEG(package["z"].shape[1], checkpoint["signature"]["dropout"]).to(args.device)
        model.load_state_dict(checkpoint["model"])
        inference = lambda eeg: predict(model, eeg, args.device, args.batch_size)
    reports = {}
    for partition in ("train", "validation", "test"):
        indices = package["splits"][partition]
        groups = {f"session{i+1}": [i] for i in range(3)}
        groups["training_sessions_mean"] = sessions
        if args.protocol == "all_sessions":
            groups["all_sessions_mean"] = [0, 1, 2]
        for name, selected in groups.items():
            pred = inference(package["eeg"][indices][:, selected])
            metrics = measure(pred, package, indices)
            generator = torch.Generator().manual_seed(args.seed + 10000)
            shuffled = [measure(pred[within_category_permutation(package["categories"][indices], generator)],
                                package, indices)["within_category_mrr"] for _ in range(args.shuffles)]
            key = f"{partition}/{name}"
            reports[key] = {"metrics": metrics, "mean_baseline": measure(torch.zeros_like(pred), package, indices),
                            "video_count": len(indices), "shuffle_mrr": shuffled,
                            "shuffle_mean_mrr": sum(shuffled) / len(shuffled) if shuffled else None,
                            "seen_videos": partition == "train",
                            "heldout_session": args.protocol == "holdout_s3" and name == "session3"}
            atomic_save({"predicted": pred, "video_ids": [package["ids"][i] for i in indices]},
                        output / f"{partition}_{name}_predictions.pt")
            print(f"[c2-sessions] {args.protocol}/{args.variant} {key} {json.dumps(metrics)}", flush=True)
    atomic_json({"protocol": signature, "checkpoint_epoch": checkpoint.get("epoch"),
                 "selected_alpha": checkpoint.get("fit", {}).get("alpha"), "reports": reports}, output / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate", "wave"), required=True)
    parser.add_argument("--prepared", type=Path, default=Path("outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eeg_semantic/c2_sessions_v3"))
    parser.add_argument("--variant", choices=("ridge", "single_session", "consistent"), default="single_session")
    parser.add_argument("--protocol", choices=("all_sessions", "holdout_s3"), default="all_sessions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "wave":
        for protocol in ("all_sessions", "holdout_s3"):
            for variant in ("ridge", "single_session", "consistent"):
                for stage in ("train", "evaluate"):
                    command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                               "--protocol", protocol, "--variant", variant, "--resume"]
                    for flag in ("prepared", "output_root", "device", "epochs", "min_epochs", "patience", "batch_size",
                                 "lr", "weight_decay", "dropout", "temperature", "seed", "threads", "shuffles"):
                        command.extend(["--" + flag.replace("_", "-"), str(getattr(args, flag))])
                    print("[c2-sessions] " + " ".join(command), flush=True)
                    if not args.dry_run:
                        subprocess.run(command, check=True)
        print("[c2-sessions] " + ("dry-run COMPLETE" if args.dry_run else "wave COMPLETE"))
        return
    if args.dry_run:
        parser.error("--dry-run requires --stage wave")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
