"""Export fused three-session Compact C1/C2/C3 conditions for Tora injection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from ms_video_eval.eeg2caption_adapter import (
    CompactToraAlignmentModel,
    load_eeg2caption_fold,
)
from ms_video_eval.tora_conditioning import ToraPCAProjector, read_tora_condition_index
from ms_video_eval.tora_text_autoencoder import ToraTextAutoencoder, ToraTextAutoencoderConfig
from scripts.train_compact_tora_alignment import CompactAlignmentDataset, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data = config["data"]
    model_values = config["model"]
    method = checkpoint["method"]
    slots = int(model_values["condition_slots"])
    dimension = int(model_values["condition_dim"])
    fold = load_eeg2caption_fold(
        ROOT, resolve(data["trials"]), resolve(data["semantic_labels"]),
        resolve(data["split_plan"]), data["fold"],
        tuple(data.get("sessions", ("session1", "session2", "session3"))),
        int(model_values.get("sample_points", 800)),
    )
    target_index = read_tora_condition_index(resolve(data["tora_target_index"]))
    dataset = CompactAlignmentDataset(
        eeg=fold.eeg, labels=fold.object_labels, pair_indices=fold.category_targets,
        cardinalities=fold.cardinalities, ids=fold.video_ids,
        indices=fold.split_indices[args.partition],
        mean=checkpoint["normalization_mean"], std=checkpoint["normalization_std"],
        eeg_scale=float(checkpoint["eeg_scale"]), target_index=target_index,
        target_kind=method, expected_shape=(slots, dimension),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=False, num_workers=int(config["training"].get("workers", 0)),
    )
    model = CompactToraAlignmentModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_mean = checkpoint.get("target_mean")
    target_mean_device = None if target_mean is None else target_mean.to(device)
    metrics, raw = evaluate(model, loader, device, target_mean_device)
    projector = None
    autoencoder = None
    if method == "tora_pca":
        projector = ToraPCAProjector.load(resolve(data["pca_projector"]), dim=dimension)
    elif method == "tora_autoencoder":
        package = torch.load(
            resolve(data["autoencoder_checkpoint"]), map_location=device, weights_only=False
        )
        autoencoder = ToraTextAutoencoder(ToraTextAutoencoderConfig(**package["config"])).to(device)
        autoencoder.load_state_dict(package["state_dict"])
        autoencoder.eval()
    video_dir = args.output_dir / "video_aggregated"
    video_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predicted = raw["predicted"]
    target = raw["target"]
    for index, video_id in enumerate(raw["video_ids"]):
        latent = predicted[index]
        if projector is not None:
            hidden_state = projector.decode(latent.unsqueeze(0))[0].float()
        elif autoencoder is not None:
            with torch.inference_mode():
                hidden_state = autoencoder.decode(latent.unsqueeze(0).to(device))[0].float().cpu()
        else:
            hidden_state = latent.float()
        path = video_dir / f"{video_id}.pt"
        torch.save({
            "schema_version": 2, "video_id": video_id,
            "caption": f"Fused three-session EEG semantic condition for {video_id}",
            "hidden_state": hidden_state, "source_checkpoint": str(args.checkpoint.resolve()),
            "source_sessions": list(data.get("sessions", ("session1", "session2", "session3"))),
            "source_method": method, "fusion": "Compact features + query cross-attention",
        }, path)
        rows.append({
            "video_id": video_id, "condition_path": str(path.resolve()), "trial_count": 3,
            "target_space_mse": float(torch.nn.functional.mse_loss(latent, target[index])),
            "target_space_token_cosine": float(
                torch.nn.functional.cosine_similarity(latent, target[index], dim=-1).mean()
            ),
        })
        if args.smoke:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "video_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "schema_version": 2, "implementation": "EEG2Caption Compact shared",
        "checkpoint": str(args.checkpoint.resolve()), "method": method,
        "partition": args.partition, "trial_count": len(raw["video_ids"]) * 3,
        "video_count": len(raw["video_ids"]), "single_trial_primary": False,
        "session_fusion_primary": True,
        "target_space_mse": metrics["mse"],
        "target_space_token_cosine": metrics["token_cosine"],
        "video_retrieval_top1": metrics["retrieval_top1"],
        "video_retrieval_top5": metrics["retrieval_top5"],
        "object_macro_ap": metrics["object_macro_ap"],
        "category_accuracy": metrics["category_accuracy"],
        "chance_top1": 1.0 / max(1, len(raw["video_ids"])),
        "chance_top5": min(5, len(raw["video_ids"])) / max(1, len(raw["video_ids"])),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
