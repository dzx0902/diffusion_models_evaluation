"""Six-class control for within-session and held-out-session generalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))
from EEG2Caption.src.common import CompactEEGClassifier
from ms_video_eval.c2_residual import normalize_sessions
from scripts.run_c2_sessions import features, ridge_fit, ridge_predict
from scripts.run_c2_residual import digest, atomic_json, atomic_save


def metrics(scores, labels):
    predicted = scores.argmax(-1)
    confusion = torch.bincount(labels * 6 + predicted, minlength=36).reshape(6, 6)
    recall = confusion.diag() / confusion.sum(1).clamp_min(1)
    return {"accuracy": float((predicted == labels).float().mean()), "balanced_accuracy": float(recall.mean()),
            "per_class_recall": recall.tolist(), "confusion": confusion.tolist(), "chance": 1/6}


@torch.no_grad()
def infer(model, eeg, device, batch):
    model.eval()
    return torch.cat([model(x.to(device))["session_pair_logits"].mean(1).cpu() for x in eeg.split(batch)])


def run(args):
    fingerprint = digest(args.prepared)
    package = torch.load(args.prepared, map_location="cpu", weights_only=False)
    sessions = [0, 1] if args.protocol == "holdout_s3" else [0, 1, 2]
    package = normalize_sessions(package, sessions)
    eeg, labels = package["eeg"], package["categories"]
    if set(labels.tolist()) != set(range(6)):
        raise ValueError("Require the first six categories")
    ti, vi = package["splits"]["train"], package["splits"]["validation"]
    output = args.output_root / args.protocol / args.model / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    signature = {"prepared_sha256": fingerprint, "protocol": args.protocol, "model": args.model,
                 "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                 "normalization_fit_sessions": sessions, "selection": "validation allowed-session mean accuracy",
                 "learning_rate": 0.0003, "dropout": 0.25, "patience": 20, "min_epochs": 20,
                 "ridge_alphas": [0.001, 0.01, 0.1, 1., 10., 100.]}
    if (output / "completed.json").exists():
        checkpoint = torch.load(output / "best.pt", weights_only=False, map_location="cpu")
        if checkpoint["signature"] != signature:
            raise ValueError("Completed configuration differs; use a new output root")
    elif args.model == "ridge":
        x = features(eeg[ti][:, sessions])
        y = F.one_hot(labels[ti], 6)[:, None].expand(-1, len(sessions), -1).flatten(0, 1).double()
        best, grid = -1., []
        for alpha in signature["ridge_alphas"]:
            fit = ridge_fit(x, y, alpha)
            result = metrics(ridge_predict(fit, eeg[vi][:, sessions]), labels[vi])
            grid.append({"alpha": alpha, "validation": result})
            if result["accuracy"] > best:
                best = result["accuracy"]
                atomic_save({"signature": signature, "fit": fit, "validation": result}, output / "best.pt")
        atomic_json(grid, output / "validation_grid.json")
        atomic_json({"signature": signature}, output / "completed.json")
    else:
        model = CompactEEGClassifier(num_pairs=6, dropout=0.25).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        generator = torch.Generator().manual_seed(args.seed)
        best, stale, start = -1., 0, 1
        last_path = output / "last.pt"
        if last_path.exists():
            last = torch.load(last_path, weights_only=False, map_location="cpu")
            if last["signature"] != signature:
                raise ValueError("Resume configuration differs")
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            generator.set_state(last["loader_rng"])
            torch.set_rng_state(last["rng"])
            if torch.cuda.is_available() and last["cuda_rng"]:
                torch.cuda.set_rng_state_all(last["cuda_rng"])
            best, stale, start = last["best"], last["stale"], last["epoch"] + 1
        loader = DataLoader(TensorDataset(eeg[ti][:, sessions], labels[ti]), batch_size=args.batch_size,
                            shuffle=True, generator=generator)
        history = output / "history.jsonl"
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            history.write_text("".join(json.dumps(r) + "\n" for r in rows if r["epoch"] < start))
        for epoch in range(start, args.epochs + 1):
            if epoch > 20 and stale >= 20:
                break
            model.train()
            total = 0.
            for x, y in loader:
                scores = model(x.to(args.device))["session_pair_logits"]
                truth = y.to(args.device)[:, None].expand(-1, len(sessions)).flatten()
                loss = F.cross_entropy(scores.flatten(0, 1), truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite classification loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach()) * len(x)
            scheduler.step()
            result = metrics(infer(model, eeg[vi][:, sessions], args.device, args.batch_size), labels[vi])
            improved = result["accuracy"] > best + 1e-6
            best, stale = (result["accuracy"], 0) if improved else (best, stale + 1)
            checkpoint = {"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                          "epoch": epoch, "best": best, "stale": stale, "validation": result}
            if improved:
                atomic_save(checkpoint, output / "best.pt")
            atomic_save(checkpoint, last_path)
            with history.open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "train_loss": total/len(ti), "validation": result}) + "\n")
            print(f"[category] {args.protocol} epoch={epoch} val_acc={result['accuracy']:.4f}", flush=True)
        atomic_json({"signature": signature}, output / "completed.json")
    checkpoint = torch.load(output / "best.pt", weights_only=False, map_location="cpu")
    if args.model == "compact":
        model = CompactEEGClassifier(num_pairs=6, dropout=0.25).to(args.device)
        model.load_state_dict(checkpoint["model"])
        predict = lambda x: infer(model, x, args.device, args.batch_size)
    else:
        predict = lambda x: ridge_predict(checkpoint["fit"], x)
    reports = {}
    for partition, indices in package["splits"].items():
        for name, subset in {"session1": [0], "session2": [1], "session3": [2], "training_sessions_mean": sessions}.items():
            scores = predict(eeg[indices][:, subset])
            result = metrics(scores, labels[indices])
            result.update(video_count=len(indices), seen_video=partition == "train",
                          heldout_session=args.protocol == "holdout_s3" and name == "session3")
            reports[f"{partition}/{name}"] = result
            atomic_save({"scores": scores, "labels": labels[indices], "video_ids": [package["ids"][i] for i in indices]},
                        output / f"{partition}_{name}_predictions.pt")
    atomic_json({"protocol": signature, "epoch": checkpoint.get("epoch"),
                 "alpha": checkpoint.get("fit", {}).get("alpha"), "reports": reports}, output / "report.json")
    print(f"[category] report={output / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eeg_semantic/session_category_control"))
    parser.add_argument("--protocol", choices=("all_sessions", "holdout_s3"), default="holdout_s3")
    parser.add_argument("--model", choices=("ridge", "compact"), default="compact")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wave", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.wave:
        for protocol in ("all_sessions", "holdout_s3"):
            for model in ("ridge", "compact"):
                command = [sys.executable, str(Path(__file__).resolve()), "--protocol", protocol, "--model", model]
                for flag in ("prepared", "output_root", "epochs", "batch_size", "seed", "threads", "device"):
                    command.extend(["--" + flag.replace("_", "-"), str(getattr(args, flag))])
                print("[category] " + " ".join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)
        print("[category] " + ("dry-run COMPLETE" if args.dry_run else "wave COMPLETE"))
        return
    if args.dry_run:
        parser.error("--dry-run requires --wave")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
