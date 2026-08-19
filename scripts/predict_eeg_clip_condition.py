"""Predict a fixed CLIP video condition from one EEG trial without loading diffusion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditioner, EEGConditionerConfig, add_condition_offset
from ms_video_eval.eeg_protocol import trial_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export target, EEG, shuffled-EEG, or zero CLIP condition.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--video-id", required=True, help="Semantic target video.")
    parser.add_argument("--session", default="session3")
    parser.add_argument("--eeg-video-id", default=None)
    parser.add_argument("--eeg-session", default=None)
    parser.add_argument("--condition-source", choices=["eeg", "target", "zero"], default="eeg")
    parser.add_argument("--expected-duration-sec", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_targets(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["video_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def select_trial(args: argparse.Namespace) -> dict[str, str]:
    eeg_video_id = args.eeg_video_id or args.video_id
    eeg_session = args.eeg_session or args.session
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row["video_id"] == eeg_video_id and row["session"] == eeg_session
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one EEG trial for {eeg_video_id}/{eeg_session}; found {len(matches)}")
    duration = trial_duration(matches[0])
    if abs(duration - args.expected_duration_sec) > 1e-6:
        raise ValueError(f"EEG duration={duration}; expected {args.expected_duration_sec}")
    return matches[0]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = EEGConditionerConfig(**checkpoint["config"])
    model = EEGConditioner(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    targets = read_targets(args.targets)
    if args.video_id not in targets:
        raise KeyError(f"Missing condition target for {args.video_id}")
    target_payload = torch.load(
        targets[args.video_id]["latent_path"],
        map_location="cpu",
        weights_only=True,
    )
    target = target_payload["latent"].float()
    if target.shape != (config.slots, config.latent_dim):
        raise ValueError(
            f"Target shape={tuple(target.shape)}, checkpoint expects={(config.slots, config.latent_dim)}"
        )

    trial = select_trial(args)
    npz = np.load(trial["npz_path"], allow_pickle=False)
    length = int(trial["length_samples"])
    signal = np.asarray(npz["eeg"][int(trial["trial_index"]), :, :length], dtype=np.float32)
    signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-6)
    with torch.inference_mode():
        predicted, _ = model(torch.from_numpy(signal).unsqueeze(0).to(device))
        predicted = add_condition_offset(predicted, checkpoint.get("target_mean"))
    predicted = predicted.squeeze(0).float().cpu()

    if args.condition_source == "target":
        condition = target
    elif args.condition_source == "zero":
        condition = torch.zeros_like(target)
    else:
        condition = predicted

    mse = float(F.mse_loss(predicted, target).item())
    token_cosine = float(F.cosine_similarity(predicted, target, dim=-1).mean().item())
    pooled_cosine = float(
        F.cosine_similarity(predicted.mean(dim=0), target.mean(dim=0), dim=0).item()
    )
    summary = {
        "video_id": args.video_id,
        "session": args.session,
        "eeg_video_id": trial["video_id"],
        "eeg_session": trial["session"],
        "trial_index": int(trial["trial_index"]),
        "condition_source": args.condition_source,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "condition_shape": list(condition.shape),
        "mse": mse,
        "mean_token_cosine": token_cosine,
        "pooled_cosine": pooled_cosine,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": condition, "tokens": config.slots, "summary": summary}, args.output)
    print("[eeg-clip] " + json.dumps(summary), flush=True)
    print(f"[eeg-clip] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
