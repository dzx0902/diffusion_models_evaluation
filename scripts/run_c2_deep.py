"""C2-v5: fixed optimizer budgets, temporal visual supervision and donor pretraining."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from ms_video_eval.c2_deep import (DeepC2, temporal_visual_loss, learning_rate,
                                 validate_splits, validate_donor_ids)
from ms_video_eval.c2_residual import (normalize_sessions, contrastive_loss,
                                     variance_covariance, within_category_permutation, decode)
from ms_video_eval.tora_conditioning import ToraPCAProjector
from scripts.run_c2_residual import digest, measure, resolve
from scripts.train_compact_tora_alignment import atomic_save, atomic_json
from scripts.build_eeg_video_manifest import load_session


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare_visual(args, package, fingerprint):
    import cv2
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    destination = args.output_root / "visual.pt"
    signature = {"prepared_sha256": fingerprint, "manifest_sha256": digest(args.video_manifest),
                 "clip_root": str(args.clip_root.resolve()), "frames": args.frames,
                 "duration_sec": 4, "sampling": "ordered_bin_centers_first_4s"}
    if destination.exists():
        if load(destination)["signature"] != signature:
            raise ValueError("Visual cache signature differs; use another output root")
        print("[c2-deep] visual cache already complete", flush=True)
        return
    processor = CLIPProcessor.from_pretrained(args.clip_root, local_files_only=True)
    model = CLIPModel.from_pretrained(args.clip_root, local_files_only=True).eval().to(args.device)
    rows = [json.loads(line) for line in args.video_manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    mapping = {}
    for row in rows:
        if row["video_id"] in mapping:
            raise ValueError(f"Duplicate video manifest ID: {row['video_id']}")
        mapping[row["video_id"]] = row
    images, texts = [], []
    for i, video in enumerate(package["ids"]):
        cap = cv2.VideoCapture(str(resolve(mapping[video]["video_path"])))
        frames = []
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or fps <= 0 or count/fps < 3.9:
                raise ValueError(f"Unreadable/short 4s video: {video}")
            for t in (np.arange(args.frames) + 0.5) * 4 / args.frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(round(t*fps)), int(count)-1))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Frame decoding failed: {video}, t={t}")
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
        with torch.inference_mode():
            visual = model.get_image_features(**processor(images=frames, return_tensors="pt").to(args.device))
            text = model.get_text_features(**processor(text=[package["captions"][i]], padding=True,
                                                      truncation=True, return_tensors="pt").to(args.device))
        images.append(F.normalize(visual.float(), dim=-1).cpu())
        texts.append(F.normalize(text.float(), dim=-1).cpu()[0])
        if (i+1) % 20 == 0 or i+1 == len(package["ids"]):
            print(f"[c2-deep] visual cache {i+1}/{len(package['ids'])}", flush=True)
    atomic_save({"signature": signature, "ids": package["ids"],
                 "visual": torch.stack(images), "text": torch.stack(texts)}, destination)


def prepare_donors(args, package, fingerprint):
    """Read audited NPZ/metadata joins; materialize only global TRAIN videos."""
    destination = args.output_root / "donors"
    destination.mkdir(parents=True, exist_ok=True)
    names = args.subjects or sorted(p.name for p in args.eeg_root.iterdir()
                                   if p.is_dir() and p.name != args.target_subject)
    if len(names) != len(set(names)) or args.target_subject in names:
        raise ValueError("Donors must be unique and exclude the target subject")
    ids = [package["ids"][i] for i in package["splits"]["train"]]
    reference = args.eeg_root / args.target_subject / "session1/EEG/eeg_data.npz"
    with np.load(reference, allow_pickle=False) as data:
        channels = data["channel_names"].tolist()
    if len(channels) != 62 or len(set(channels)) != 62:
        raise ValueError("Expected 62 distinct target channels")
    for session in (2, 3):
        with np.load(args.eeg_root / args.target_subject / f"session{session}/EEG/eeg_data.npz", allow_pickle=False) as data:
            if data["channel_names"].tolist() != channels:
                raise ValueError("Target session channel orders differ")
    accepted, excluded = [], {}
    for subject in names:
        # A missing session is explicit in the audit. Malformed present data fails closed.
        folders = [args.eeg_root / subject / f"session{s}" for s in (1, 2, 3)]
        if any(not (f / "EEG/eeg_data.npz").is_file() for f in folders):
            excluded[subject] = "missing one or more of session1/2/3 EEG/eeg_data.npz"
            if args.subjects:
                raise ValueError(f"Requested donor {subject}: {excluded[subject]}")
            continue
        source_hashes = {}
        sessions = []
        for folder in folders:
            _, trials = load_session(folder)
            missing = set(ids) - set(trials)
            if missing:
                raise ValueError(f"{subject}/{folder.name}: missing global train videos: {sorted(missing)[:5]}")
            path = resolve(trials[ids[0]]["npz_path"])
            source_hashes[folder.name] = digest(path)
            source_hashes[folder.name + "_metadata"] = digest(resolve(trials[ids[0]]["metadata_path"]))
            with np.load(path, allow_pickle=False) as data:
                names_here = data["channel_names"].tolist()
                if len(names_here) != 62 or set(names_here) != set(channels):
                    raise ValueError(f"{subject}: channel set mismatch")
                order = [names_here.index(c) for c in channels]
                eeg = data["eeg"]
                values = []
                for video in ids:
                    row = trials[video]
                    if abs(row["sfreq"]-200) > 1e-6 or row["length_samples"] != 800:
                        raise ValueError(f"{subject}/{video}: require 200Hz and exactly 4s")
                    values.append(torch.from_numpy(np.asarray(eeg[row["trial_index"], order, :800], dtype=np.float32).copy()))
                sessions.append(torch.stack(values))
        eeg = torch.stack(sessions, dim=1) * 1e6
        if not torch.isfinite(eeg).all():
            raise ValueError(f"Nonfinite EEG: {subject}")
        mean = eeg.mean((0, 1, 3))
        std = eeg.std((0, 1, 3)).clamp_min(1e-6)
        cache = {"subject": subject, "ids": ids, "prepared_sha256": fingerprint,
                 "eeg": (eeg-mean[None,None,:,None])/std[None,None,:,None],
                 "mean": mean, "std": std, "channel_names": channels, "source_hashes": source_hashes}
        path = destination / f"{subject}.pt"
        if path.exists():
            old = load(path)
            if any(old[k] != cache[k] for k in ("source_hashes", "prepared_sha256", "ids")):
                raise ValueError(f"Existing donor changed: {subject}; use a new output root")
        else:
            atomic_save(cache, path)
        accepted.append({"subject": subject, "file": path.name, "sha256": digest(path)})
        print(f"[c2-deep] donor {subject}: {len(ids)} TRAIN videos x 3 sessions", flush=True)
    if not accepted:
        raise ValueError("No valid donors found")
    atomic_json({"prepared_sha256": fingerprint, "target_subject": args.target_subject,
                 "train_ids": ids, "accepted": accepted, "excluded": excluded,
                 "heldout_video_leakage": False, "normalization_fit": "train_only",
                 "independent_stimulus_alignment": "NOT_VERIFIED"}, destination / "audit.json")


def resources(args, package, fingerprint, variant):
    visual, donors = None, []
    signatures = {}
    if variant != "long":
        path = args.output_root / "visual.pt"
        visual = load(path)
        if visual["ids"] != package["ids"] or visual["signature"]["prepared_sha256"] != fingerprint:
            raise ValueError("Visual cache / split mismatch")
        if visual["visual"].shape != (len(package["ids"]), args.frames, 512):
            raise ValueError("Require configured ordered frames and CLIP projection dim 512")
        if not all(torch.isfinite(visual[k]).all() for k in ("visual", "text")):
            raise ValueError("Nonfinite visual targets")
        signatures["visual_sha256"] = digest(path)
    if variant == "multisubject":
        path = args.output_root / "donors/audit.json"
        audit = json.loads(path.read_text())
        if audit["prepared_sha256"] != fingerprint or audit["target_subject"] != args.target_subject:
            raise ValueError("Donor audit / target split mismatch")
        signatures["donor_audit_sha256"] = digest(path)
        for entry in audit["accepted"]:
            file = path.parent / entry["file"]
            if digest(file) != entry["sha256"]:
                raise ValueError(f"Donor cache modified: {file}")
            item = torch.load(file, map_location="cpu", weights_only=False, mmap=True)
            validate_donor_ids(item["ids"], package)
            if item["prepared_sha256"] != fingerprint:
                raise ValueError("Donor source signature mismatch")
            donors.append(item)
        if not donors:
            raise ValueError("Multi-subject training needs donors")
    return visual, donors, signatures


@torch.no_grad()
def predictions(model, package, indices, device, batch):
    model.eval()
    output = []
    for start in range(0, len(indices), batch):
        eeg = package["eeg"][indices[start:start+batch]].to(device)
        output.append(model(eeg, torch.zeros(len(eeg), dtype=torch.long, device=device))["z"].cpu())
    return torch.cat(output)


def train(args, package, fingerprint, variant):
    visual, donors, extra = resources(args, package, fingerprint, variant)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_config = dict(dim=package["z"].shape[1], subjects=1+len(donors), width=args.width,
                        layers=args.layers, frames=args.frames, dropout=args.dropout)
    signature = {"schema_version": 1, "prepared_sha256": fingerprint, "variant": variant,
                 "updates": args.updates, "pretrain_updates": args.pretrain_updates if donors else 0,
                 "warmup": args.warmup, "lr": args.lr, "batch_size": args.batch_size,
                 "seed": args.seed, "model": model_config, "eval_every": args.eval_every,
                 "selection_metric": "validation/within_category_mrr", "early_stopping": False,
                 "target_subject": args.target_subject, "normalization": "train_shared_sessions",
                 "loss_weights": {"mse": .05, "cosine": 1., "contrastive": 2., "variance": .1,
                                  "covariance": .001, "session": .1, "visual": 1., "text": .5},
                 "weight_decay": .01, "contrastive_temperature": .1, **extra}
    output = args.output_root / variant / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        if json.loads((output / "completed.json").read_text())["signature"] != signature:
            raise ValueError("Completed run settings differ; use a new output root")
        print(f"[c2-deep] skip completed {variant}", flush=True)
        return
    model = DeepC2(**model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    generator = torch.Generator().manual_seed(args.seed)
    start, best, history = 0, -float("inf"), []
    subject_updates = [0] * (len(donors)+1)
    if (output / "last.pt").exists():
        if not args.resume:
            raise FileExistsError("Run exists: pass --resume")
        ckpt = load(output / "last.pt")
        if ckpt["signature"] != signature:
            raise ValueError("Resume signature differs")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, best, history = ckpt["step"], ckpt["best"], ckpt["history"]
        subject_updates = ckpt["subject_updates"]
        generator.set_state(ckpt["sampler_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    elif (output / "run_config.json").exists() and not args.resume:
        raise FileExistsError("Incomplete run exists: inspect then use --resume")
    atomic_json(signature, output / "run_config.json")
    # Donor rows are indexed through video IDs, never assumed to match tensor row order.
    lookup = {v: i for i, v in enumerate(package["ids"])}
    donor_indices = [torch.tensor([lookup[v] for v in d["ids"]]) for d in donors]
    train_ids = torch.tensor(package["splits"]["train"])
    train_eeg = package["eeg"][train_ids]
    if len(train_ids) < args.batch_size:
        raise ValueError("batch-size exceeds training video count; budget would not match")
    unique = {}
    for i in train_ids.tolist():
        unique.setdefault(int(package["caption_ids"][i]), i)
    bank_idx = list(unique.values())
    bank = package["z"][bank_idx].to(args.device)
    bank_labels = package["caption_ids"][bank_idx].to(args.device)
    print(f"[c2-deep] {variant} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"start={start} budget={args.updates} early_stop=OFF donors={len(donors)}", flush=True)
    tick = time.monotonic()
    running, elapsed_steps = 0., 0
    for step in range(start+1, args.updates+1):
        model.train()
        subject = int(torch.randint(len(donors)+1, (), generator=generator)) if donors and step <= args.pretrain_updates else 0
        source = donors[subject-1]["eeg"] if subject else train_eeg
        rows = torch.randperm(len(source), generator=generator)[:args.batch_size]
        indices = donor_indices[subject-1][rows] if subject else train_ids[rows]
        eeg = source[rows].to(args.device)
        target = package["z"][indices].to(args.device)
        subject_ids = torch.full((len(rows),), subject, device=args.device, dtype=torch.long)
        lr = learning_rate(step, args.updates, args.warmup, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        result = model(eeg, subject_ids)
        pred = result["z"]
        positive = package["caption_ids"][indices].to(args.device)[:,None] == bank_labels[None,:]
        variance, covariance = variance_covariance(pred)
        loss = (.05 * F.mse_loss(pred, target) + (1-F.cosine_similarity(pred, target)).mean()
                + 2 * contrastive_loss(pred, bank, positive) + .1*variance + .001*covariance
                + .1 * (result["sessions"]-pred[:,None,:]).square().mean())
        if visual is not None:
            loss = loss + temporal_visual_loss(result["visual"], visual["visual"][indices].to(args.device))
            loss = loss + .5 * (1-F.cosine_similarity(result["text"], visual["text"][indices].to(args.device))).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        subject_updates[subject] += 1
        running += float(loss.detach())
        elapsed_steps += 1
        if step % args.log_every == 0:
            print(f"[c2-deep] {variant} update={step}/{args.updates} loss={running/elapsed_steps:.5f} "
                  f"lr={lr:.3g} grad_norm={float(norm):.3f} "
                  f"updates_per_sec={elapsed_steps/max(time.monotonic()-tick,1e-6):.2f}", flush=True)
            running, elapsed_steps, tick = 0., 0, time.monotonic()
        if step % args.eval_every == 0 or step == args.updates:
            valid_ids = package["splits"]["validation"]
            metrics = measure(predictions(model, package, valid_ids, args.device, args.batch_size), package, valid_ids)
            score = metrics["within_category_mrr"]
            # Multi-subject final model is selected only after target fine-tuning starts.
            eligible = not donors or step > args.pretrain_updates
            improved = eligible and score > best
            if improved:
                best = score
            history.append({"step": step, "stage": "pretrain" if donors and not eligible else "target",
                            "validation": metrics})
            state = {"signature": signature, "model_config": model_config, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step, "best": best, "history": history,
                     "subject_updates": subject_updates,
                     "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                     "normalization_mean": package["shared_mean"], "normalization_std": package["shared_std"]}
            if improved:
                atomic_save(state, output / "best.pt")
            atomic_save(state, output / "last.pt")
            atomic_json(history, output / "history.json")
            print(f"[c2-deep] validation update={step} within_MRR={score:.5f} best={best:.5f}", flush=True)
    atomic_json({"signature": signature, "optimizer_updates": args.updates,
                 "video_examples_seen": args.updates * args.batch_size,
                 "session_examples_seen": args.updates * args.batch_size * 3,
                 "subject_updates": subject_updates,
                 "best_validation_mrr": best, "status": "COMPLETE"}, output / "completed.json")


def evaluate(args, package, fingerprint, variant):
    output = args.output_root / variant / f"seed{args.seed}"
    ckpt = load(output / "best.pt")
    if ckpt["signature"]["prepared_sha256"] != fingerprint:
        raise ValueError("Checkpoint / prepared data mismatch")
    if not (output / "completed.json").exists():
        raise ValueError("Fixed-update training incomplete; evaluation requires completed.json")
    model = DeepC2(**ckpt["model_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    indices = package["splits"][args.partition]
    pred = predictions(model, package, indices, args.device, args.batch_size)
    metrics = measure(pred, package, indices)
    shuffled = []
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.shuffles):
        perm = within_category_permutation(package["categories"][indices], generator)
        shuffled.append(measure(pred[perm], package, indices)["within_category_mrr"])
    report = {"variant": variant, "partition": args.partition, "video_count": len(indices),
              "checkpoint_step": ckpt["step"], "optimizer_updates": ckpt["signature"]["updates"],
              "metrics": metrics, "mean_baseline": measure(torch.zeros_like(pred), package, indices),
              "shuffle": {"count": len(shuffled), "mean_mrr": float(np.mean(shuffled)) if shuffled else None},
              "protocol": ckpt["signature"], "independent_stimulus_alignment": "NOT_VERIFIED",
              "within_category_note": "Oracle-category diagnostic, not end-to-end category prediction"}
    destination = output / args.partition
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(report, destination / "report.json")
    atomic_save({"ids": [package["ids"][i] for i in indices], "predictions": pred}, destination / "residual_predictions.pt")
    if args.export:
        projector = ToraPCAProjector(**package["token_projector"])
        states = decode(pred, package["pca"])
        directory = destination / "video_aggregated"
        directory.mkdir(exist_ok=True)
        rows = []
        for j, i in enumerate(indices):
            hidden = projector.decode(states[j])
            if hidden.shape != (226, 4096) or not torch.isfinite(hidden).all():
                raise ValueError("Invalid exported Tora condition")
            path = directory / f"{package['ids'][i]}.pt"
            atomic_save({"video_id": package["ids"][i], "hidden_state": hidden,
                         "caption": "EEG-only C2-v5 continuous condition", "source_method": f"c2_deep_{variant}",
                         "source_checkpoint": str((output/'best.pt').resolve())}, path)
            rows.append({"video_id": package["ids"][i], "condition_path": str(path.resolve()), "trial_count": 3})
        temporary = destination / "video_index.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        temporary.replace(destination / "video_index.jsonl")
    print(json.dumps(report, indent=2), flush=True)


def summarize(args):
    rows = []
    for variant in ("long", "joint", "multisubject"):
        path = args.output_root / variant / f"seed{args.seed}" / args.partition / "report.json"
        report = json.loads(path.read_text())
        m = report["metrics"]
        baseline = report["mean_baseline"]
        shuffled = report["shuffle"]["mean_mrr"]
        rows.append({"variant": variant, "partition": report["partition"], "videos": report["video_count"],
                     "optimizer_updates": report["optimizer_updates"], "selected_step": report["checkpoint_step"],
                     **m, "mean_baseline_within_mrr": baseline["within_category_mrr"],
                     "shuffled_within_mrr": shuffled,
                     "matched_minus_shuffled": m["within_category_mrr"]-shuffled if shuffled is not None else None})
    path = args.output_root / f"summary_{args.partition}_seed{args.seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"[c2-deep] summary={path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare-visual", "prepare-donors", "train", "evaluate", "wave", "summarize"))
    parser.add_argument("--prepared", type=Path, default=Path("outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eeg_semantic/c2_deep_v5"))
    parser.add_argument("--variant", choices=("long", "joint", "multisubject"), default="long")
    parser.add_argument("--video-manifest", type=Path, default=Path("data/manifests/video_manifest.jsonl"))
    parser.add_argument("--clip-root", type=Path, default=Path(".ms_video_models/CLIP/clip-vit-base-patch32"))
    parser.add_argument("--eeg-root", type=Path, default=Path("data/EEG_and_EYE"))
    parser.add_argument("--target-subject", choices=("chentianlin",), default="chentianlin")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--pretrain-updates", type=int, default=15000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.layers, args.eval_every, args.log_every, args.threads) < 1:
        parser.error("Budgets, model depth and intervals must be positive")
    if args.width % 8 or args.width < 8 or not 2 <= args.frames <= 50 or not 0 <= args.dropout < 1 or args.lr <= 0:
        parser.error("Invalid model configuration")
    if not 0 <= args.warmup < args.updates or args.shuffles < 0:
        parser.error("Require 0 <= warmup < updates and shuffles >= 0")
    if (args.stage == "wave" or args.variant == "multisubject") and not 0 < args.pretrain_updates < args.updates:
        parser.error("Multi-subject budget requires 0 < pretrain-updates < updates")
    variants = ("long", "joint", "multisubject") if args.stage == "wave" else (args.variant,)
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "variants": variants, "updates_each": args.updates,
                          "pretrain_updates": args.pretrain_updates, "early_stopping": False,
                          "note": "Plan only; does not validate caches, CUDA or model weights",
                          "prepared": str(args.prepared), "output_root": str(args.output_root)}, indent=2))
        return
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.stage == "summarize":
        summarize(args)
        return
    package = load(args.prepared)
    validate_splits(package)
    fingerprint = digest(args.prepared)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare-visual":
        prepare_visual(args, package, fingerprint)
    elif args.stage == "prepare-donors":
        prepare_donors(args, package, fingerprint)
    else:
        package = normalize_sessions(package, [0, 1, 2])
        if not torch.isfinite(package["eeg"]).all():
            raise ValueError("Nonfinite target EEG")
        if args.stage == "wave":
            # Fail before the first expensive run if later-stage resources are absent.
            resources(args, package, fingerprint, "multisubject")
        for variant in variants:
            if args.stage in ("train", "wave"):
                train(args, package, fingerprint, variant)
            if args.stage in ("evaluate", "wave"):
                evaluate(args, package, fingerprint, variant)
        if args.stage == "wave":
            summarize(args)


if __name__ == "__main__":
    main()
