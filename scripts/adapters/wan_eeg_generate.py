"""Generate a Wan video from one EEG trial through a trained EEG conditioner.

The adapter predicts the fixed PCA target ``[128, 512]``, applies the fold's
PCA inverse transform, patches Wan's T5 call at runtime, and delegates video
generation to Wan's original ``generate.py``.  ``--length-source target`` is
an oracle-length diagnostic; ``predicted`` is the end-to-end EEG setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditioner, EEGConditionerConfig
from ms_video_eval.wan_condition_autoencoder import WanConditionAutoencoder, WanConditionAutoencoderConfig


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Generate Wan output from one EEG trial.")
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="EEGConditioner checkpoint.")
    parser.add_argument("--trials", type=Path, required=True, help="Subject eeg_trials.csv.")
    parser.add_argument("--targets", type=Path, required=True, help="Fold-specific wan_targets.jsonl.")
    parser.add_argument("--projector", type=Path, default=None, help="Fold-specific token_pca_projector.npz.")
    parser.add_argument("--autoencoder-checkpoint", type=Path, default=None, help="Frozen Wan condition autoencoder checkpoint.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--session", default="session3")
    parser.add_argument("--trial-index", type=int, default=None, help="Disambiguate an EEG trial if required.")
    parser.add_argument(
        "--condition-source",
        choices=["eeg", "target"],
        default="eeg",
        help="Use EEG prediction or the exact stored PCA target as the Wan condition.",
    )
    parser.add_argument("--length-source", choices=["predicted", "target", "fixed"], default="predicted")
    parser.add_argument("--fixed-tokens", type=int, default=0, help="Required when --length-source fixed.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition-output", type=Path, default=None, help="Optional .pt artifact for diagnostics.")
    parser.add_argument("--enable-tf32", action="store_true")
    args, wan_args = parser.parse_known_args()
    if wan_args and wan_args[0] == "--":
        wan_args = wan_args[1:]
    return args, wan_args


def read_targets(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["video_id"]): row for row in rows}


def find_trial(args: argparse.Namespace) -> dict[str, str]:
    with args.trials.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row["video_id"] == args.video_id and row["session"] == args.session
        and (args.trial_index is None or int(row["trial_index"]) == args.trial_index)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one trial for video_id={args.video_id!r}, session={args.session!r}; "
            f"found {len(matches)}. Pass --trial-index when needed."
        )
    return matches[0]


def load_eeg(row: dict[str, str]) -> torch.Tensor:
    with np.load(row["npz_path"], allow_pickle=False) as payload:
        length = int(row["length_samples"])
        signal = np.asarray(payload["eeg"][int(row["trial_index"]), :, :length], dtype=np.float32)
    signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-6)
    return torch.from_numpy(signal).unsqueeze(0)


def patch_wan_t5(context: torch.Tensor) -> None:
    """Replace Wan's text encoder call while keeping its original pipeline intact."""
    from wan.modules.t5 import T5EncoderModel

    def patched_call(self, texts, device):  # type: ignore[no-untyped-def]
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Wan T5 expects a non-empty batch of prompts.")
        condition = context.to(device=torch.device(device), dtype=torch.bfloat16)
        return [condition.clone() for _ in texts]

    T5EncoderModel.__call__ = patched_call


def main() -> None:
    args, wan_args = parse_args()
    if (args.projector is None) == (args.autoencoder_checkpoint is None):
        raise ValueError("Pass exactly one of --projector or --autoencoder-checkpoint.")
    if args.length_source == "fixed" and not 1 <= args.fixed_tokens <= 128:
        raise ValueError("--fixed-tokens must be within 1..128 when --length-source fixed.")
    if not wan_args:
        raise ValueError("Pass Wan generate.py arguments after '--'.")
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = EEGConditionerConfig(**checkpoint["config"])
    model = EEGConditioner(config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])

    trial = find_trial(args)
    targets = read_targets(args.targets)
    if args.video_id not in targets:
        raise KeyError(f"No target for video_id={args.video_id!r}")
    target_payload = torch.load(targets[args.video_id]["latent_path"], map_location="cpu", weights_only=True)
    target_latent = target_payload["latent"].float()
    target_tokens = int(target_payload["tokens"])

    with torch.inference_mode():
        predicted, logits = model(load_eeg(trial).to(device))
    predicted = predicted.squeeze(0).float()
    predicted_tokens = int(logits.argmax(dim=-1).item() + config.min_tokens)
    confidence = float(logits.softmax(dim=-1).max().item())
    if args.length_source == "target":
        token_count = target_tokens
    elif args.length_source == "fixed":
        token_count = args.fixed_tokens
    else:
        token_count = predicted_tokens

    condition_latent = target_latent.to(device) if args.condition_source == "target" else predicted
    if args.projector is not None:
        pca = np.load(args.projector)
        components = torch.from_numpy(pca["components"][: config.latent_dim].astype(np.float32)).to(device)
        mean = torch.from_numpy(pca["mean"].astype(np.float32)).to(device)
        if components.shape != (config.latent_dim, 4096):
            raise ValueError(f"Unexpected PCA components shape: {tuple(components.shape)}")
        context = condition_latent[:token_count].to(device) @ components + mean
        decoder_type = "pca"
    else:
        decoder_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device, weights_only=False)
        decoder_config = WanConditionAutoencoderConfig(**decoder_checkpoint["config"])
        if decoder_config.latent_dim != config.latent_dim or decoder_config.slots != predicted.shape[0]:
            raise ValueError("EEG target shape and autoencoder decoder configuration differ.")
        decoder = WanConditionAutoencoder(decoder_config).to(device).eval()
        decoder.load_state_dict(decoder_checkpoint["state_dict"])
        with torch.inference_mode():
            context = decoder.decode(condition_latent.unsqueeze(0), torch.tensor([token_count], device=device))[0, :token_count]
        decoder_type = "autoencoder"

    valid = min(target_tokens, predicted.shape[0])
    latent_mse = float((predicted[:valid].cpu() - target_latent[:valid]).square().mean().item())
    pooled_cosine = float(
        F.cosine_similarity(
            predicted[:valid].mean(dim=0), target_latent[:valid].to(device).mean(dim=0), dim=0
        ).cpu().item()
    )
    summary = {
        "video_id": args.video_id,
        "session": args.session,
        "trial_index": int(trial["trial_index"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "condition_source": args.condition_source,
        "decoder_type": decoder_type,
        "length_source": args.length_source,
        "target_tokens": target_tokens,
        "predicted_tokens": predicted_tokens,
        "used_tokens": token_count,
        "length_confidence": confidence,
        "valid_latent_mse": latent_mse,
        "pooled_cosine": pooled_cosine,
    }
    print("[wan-eeg] " + json.dumps(summary, ensure_ascii=True), flush=True)
    if args.condition_output is not None:
        args.condition_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "eeg_pca_latent": predicted.cpu(),
                "condition_pca_latent": condition_latent.cpu(),
                "wan_context": context.cpu(),
                **summary,
            },
            args.condition_output,
        )

    repo = args.wan_repo.resolve()
    generate_py = repo / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(generate_py)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    patch_wan_t5(context)
    old_argv = sys.argv
    try:
        sys.argv = [str(generate_py), *wan_args]
        runpy.run_path(str(generate_py), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
