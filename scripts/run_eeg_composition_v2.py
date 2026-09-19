"""Non-destructive composition diagnostics and pair-held-out development runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import run_eeg_composition as base
from ms_video_eval.eeg_composition import OBJECTS, PROTOCOLS, metrics, set_metrics, select_threshold
from EEG2Caption.src.common import CompactEEGClassifier


def development_data(p, category):
    """Held combination is used only for development; never load triple EEG here."""
    pool = p["splits"]["train"] + p["splits"]["validation"]
    train = [i for i in p["splits"]["train"] if p["ids"][i][:2] != category]
    valid = [i for i in pool if p["ids"][i][:2] == category]
    if not train or not valid or not (p["labels"][train].sum(0) > 0).all():
        raise ValueError("Development split must retain every object in training")
    q = dict(p)
    q["splits"] = {"train": train, "validation": valid, "test": []}
    return q


def detailed_metrics(logits, labels, threshold, cardinality):
    result = metrics(logits, labels, threshold, cardinality)
    top = torch.zeros_like(labels, dtype=torch.bool)
    top.scatter_(1, logits.topk(cardinality, dim=1).indices, True)
    for j, name in enumerate(OBJECTS):
        pos = labels[:, j].bool()
        item = result["per_object"][name]
        item["topk_recall"] = float(top[pos, j].float().mean()) if pos.any() else None
        item["topk_false_positive_rate"] = float(top[~pos, j].float().mean()) if (~pos).any() else None
        probs = logits[:, j].sigmoid()
        item["positive_mean_score"] = float(probs[pos].mean()) if pos.any() else None
        item["negative_mean_score"] = float(probs[~pos].mean()) if (~pos).any() else None
    result["all_positive_baseline"] = set_metrics(torch.ones_like(labels), labels)
    return result


def shuffle_audit(logits, labels, threshold, cardinality, repeats=200, seed=42):
    # Shuffle entire video predictions, not individual objects or individual sessions.
    rng = torch.Generator().manual_seed(seed)
    values = []
    truth = labels.bool()
    for _ in range(repeats):
        scores = logits[torch.randperm(len(logits), generator=rng)]
        pred = torch.zeros_like(truth)
        pred.scatter_(1, scores.topk(cardinality, dim=1).indices, True)
        values.append(float((pred == truth).all(1).float().mean()))
    matched = metrics(logits, labels, threshold, cardinality)["topk_metrics"]["exact_set_accuracy"]
    values = torch.tensor(values)
    return {"scope": "global video permutation; all sessions/windows move together",
            "repeats": repeats, "seed": seed, "matched_exact": matched,
            "shuffled_exact_mean": float(values.mean()),
            "shuffled_exact_q025": float(values.quantile(.025)),
            "shuffled_exact_q975": float(values.quantile(.975)),
            "matched_minus_shuffled": matched - float(values.mean()),
            "note": "Exploratory null reference, not an independent-sample confidence interval"}


@torch.no_grad()
def predict_windows(model, p, indices, sessions, mean, std, args, mode):
    if mode == "4s":
        return base.infer(model, p["eeg"], indices, sessions, 800, mean, std, args.device, args.batch_size)
    if p["eeg"].shape[-1] < 1200:
        raise ValueError("Six-second evaluation requires 1200 real samples")
    if mode == "6s":
        return base.infer(model, p["eeg"], indices, sessions, 1200, mean, std, args.device, args.batch_size)
    if mode != "6s_windows":
        raise ValueError(mode)
    # Keep the input window length at the trained 800 samples. Pool logits, not EEG.
    return torch.stack([
        base.infer(model, p["eeg"][..., offset:offset+800], indices, sessions,
                   800, mean, std, args.device, args.batch_size)
        for offset in (0, 200, 400)
    ]).mean(0)


def audit(args, p, checkpoint_path, output, development=False):
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    ckpt = base.load(checkpoint_path)
    if ckpt["signature"]["protocol"] != args.protocol or ckpt["signature"]["variant"] != args.variant:
        raise ValueError("Checkpoint protocol/variant mismatch")
    if ckpt["signature"]["seed"] != args.seed:
        raise ValueError("Checkpoint seed mismatch")
    model = CompactEEGClassifier(num_pairs=6, dropout=.35).to(args.device)
    model.load_state_dict(ckpt["model"])
    mean, std = ckpt["mean"], ckpt["std"]
    allowed = PROTOCOLS[args.protocol]
    vi = p["splits"]["validation"]
    val = base.infer(model, p["eeg"], vi, allowed, 800, mean, std, args.device, args.batch_size).mean(1)
    threshold = select_threshold(val, p["labels"][vi])
    report = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": base.digest(checkpoint_path),
              "checkpoint_epoch": ckpt["epoch"], "signature": ckpt["signature"],
              "threshold": threshold, "object_names": OBJECTS,
              "exploratory": True, "independent_stimulus_alignment": "NOT_VERIFIED",
              "development_only": development,
              "development_split": ({k: [p["ids"][i] for i in v] for k, v in p["splits"].items()}
                                    if development else None),
              "validation": detailed_metrics(val, p["labels"][vi], threshold, 2),
              "threshold_curve": [dict(threshold=t, **set_metrics(val.sigmoid() >= t, p["labels"][vi]))
                                  for t in (.05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9)],
              "threshold_curve_note": "Diagnostic only; original .10:.05:.90 selection unchanged",
              "reports": {}}
    sessions = [0, 1, 2] if args.protocol == "session_average" else [s for s in range(3) if s not in allowed]
    indices = vi if development else p["splits"]["test"]
    labels = p["labels"][indices]
    for mode in (["4s"] if development else ["4s", "6s", "6s_windows"]):
        logits = predict_windows(model, p, indices, sessions, mean, std, args, mode)
        groups = {f"session{s+1}": logits[:, j] for j, s in enumerate(sessions)}
        if len(sessions) == 3:
            groups["session_average"] = logits.mean(1)
        for name, scores in groups.items():
            result = detailed_metrics(scores, labels, threshold, 2 if development else 3)
            if not development:
                result["shuffle"] = shuffle_audit(scores, labels, threshold, 3, args.shuffle_repeats, args.seed)
                result["by_category"] = {}
                for category in ("07", "08"):
                    ix = [j for j, i in enumerate(indices) if p["ids"][i][:2] == category]
                    result["by_category"][category] = detailed_metrics(scores[ix], labels[ix], threshold, 3)
            key = f"{mode}/{name}"
            report["reports"][key] = result
            base.atomic_save({"video_ids": [p["ids"][i] for i in indices], "logits": scores,
                              "labels": labels, "object_names": OBJECTS}, output / "predictions" / f"{mode}_{name}.pt")
            print(f"[composition-v2] {key} exact={result['topk_metrics']['exact_set_accuracy']:.4f}", flush=True)
    if development:
        report["development_warning"] = "Single held combination has constant labels: AP is undefined. Use exact/recall; do not claim within-combination discrimination. No 07/08 evaluation."
    base.atomic_json(report, output / "audit.json")


def summarize(root):
    rows = []
    for path in sorted(root.glob("**/audit.json")):
        report = json.loads(path.read_text())
        for group, result in report["reports"].items():
            rows.append({"run": str(path.parent.relative_to(root)), "group": group,
                         "development_only": report["development_only"],
                         "n": result["video_count"], "epoch": report["checkpoint_epoch"],
                         "topk_exact": result["topk_metrics"]["exact_set_accuracy"],
                         "topk_recall": result["topk_metrics"]["micro_recall"],
                         "threshold_exact": result["threshold_metrics"]["exact_set_accuracy"],
                         "threshold_f1": result["threshold_metrics"]["micro_f1"],
                         "informative_macro_ap": result["informative_macro_ap"],
                         "matched_minus_shuffled": result.get("shuffle", {}).get("matched_minus_shuffled")})
    if rows:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("audit-baseline", "object-only", "development"))
    parser.add_argument("--prepared", type=Path, default=Path("outputs/eeg_composition/chentianlin/prepared.pt"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/eeg_composition"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eeg_composition_v2"))
    parser.add_argument("--subject", default="chentianlin")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--variants", nargs="+", choices=("original", "object_only"), default=["original", "object_only"])
    parser.add_argument("--held-categories", nargs="+", choices=[f"{i:02d}" for i in range(1,7)], default=[f"{i:02d}" for i in range(1,7)])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.shuffle_repeats) < 1 or args.lr <= 0:
        parser.error("Budgets and learning rate must be positive")
    if args.output_root.resolve() == args.baseline_root.resolve():
        parser.error("Output root must differ from baseline root")
    tasks = []
    for protocol in args.protocols:
        variants = ["original"] if args.stage == "audit-baseline" else (["object_only"] if args.stage == "object-only" else args.variants)
        for variant in variants:
            for category in (args.held_categories if args.stage == "development" else [None]):
                tasks.append((protocol, variant, category))
    for protocol, variant, category in tasks:
        print(f"[composition-v2] stage={args.stage} protocol={protocol} variant={variant} held={category} seed={args.seed}", flush=True)
    if args.dry_run:
        print(f"[composition-v2] dry-run jobs={len(tasks)}; no files changed")
        return
    torch.set_num_threads(args.threads)
    p = base.load(args.prepared)
    if p["protocol"]["subject"] != args.subject:
        raise ValueError("Prepared subject mismatch")
    if p["splits"] != base.split_videos(p["ids"], p["protocol"]["split_seed"]) or not torch.equal(p["labels"], base.labels_for(p["ids"])):
        raise ValueError("Prepared split or labels mismatch")
    fingerprint = base.digest(args.prepared)
    for protocol, variant, category in tasks:
        args.protocol, args.variant = protocol, variant
        args.selection = "top2_exact" if category else "macro_ap"
        relative = Path(args.subject) / protocol / variant / f"seed{args.seed}"
        q, run_hash = p, fingerprint
        if category:
            q = development_data(p, category)
            run_hash = hashlib.sha256((fingerprint + json.dumps(q["splits"], sort_keys=True)).encode()).hexdigest()
            output = args.output_root / "development" / f"held_{category}" / relative
        else:
            output = args.output_root / args.stage / relative
        if args.stage == "audit-baseline":
            checkpoint = args.baseline_root / relative / "best.pt"
        else:
            base.train(args, q, run_hash, output)
            checkpoint = output / "best.pt"
        ckpt = base.load(checkpoint)
        if ckpt["signature"]["prepared_sha256"] != run_hash:
            raise ValueError("Checkpoint prepared-data/split fingerprint mismatch")
        completed = json.loads((checkpoint.parent / "completed.json").read_text())
        if completed["signature"] != ckpt["signature"]:
            raise ValueError("Checkpoint completion signature mismatch")
        audit(args, q, checkpoint, output, development=category is not None)
    summarize(args.output_root)
    print(f"[composition-v2] COMPLETE jobs={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
