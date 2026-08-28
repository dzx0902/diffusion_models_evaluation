"""Export trial-level and session-aggregated EEG predictions for Tora injection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.eeg_conditioner import EEGConditionerConfig
from ms_video_eval.eeg_semantic import DirectSemanticAlignmentModel
from ms_video_eval.semantic_data import (
    EEGSemanticDataset,
    SemanticVocabulary,
    load_semantic_record_map,
    load_trial_rows,
    load_video_partitions,
    select_partition_trials,
    semantic_collate,
)
from ms_video_eval.tora_conditioning import (
    ToraPCAProjector,
    load_tora_condition,
    read_tora_condition_index,
)
from ms_video_eval.tora_text_autoencoder import ToraTextAutoencoder, ToraTextAutoencoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def restore_vocabulary(payload: dict[str, Any]) -> SemanticVocabulary:
    return SemanticVocabulary(
        values={key: tuple(value) for key, value in payload["values"].items()},
        min_frequency={key: int(value) for key, value in payload["min_frequency"].items()},
        unknown_token=payload.get("unknown_token", "__unknown__"),
    )


def load_target(row: dict[str, Any], method: str) -> torch.Tensor:
    if method == "direct_tora_text":
        return load_tora_condition(Path(row["condition_path"])).hidden_state
    payload = torch.load(row["latent_path"], map_location="cpu", weights_only=False)
    return payload["latent"].float()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    method = checkpoint["method"]
    if method not in {"direct_tora_text", "tora_pca", "tora_autoencoder"}:
        raise ValueError(f"Unsupported checkpoint method: {method}")
    config = checkpoint["config"]
    data = config["data"]
    partitions = load_video_partitions(resolve_path(data["split_plan"]), data["fold"])
    rows = select_partition_trials(
        load_trial_rows(resolve_path(data["trials"])),
        partitions[args.partition],
        tuple(data.get("sessions", ["session1", "session2", "session3"])),
    )
    vocabulary = restore_vocabulary(checkpoint["vocabulary"])
    dataset = EEGSemanticDataset(
        rows,
        load_semantic_record_map(resolve_path(data["semantic_labels"])),
        vocabulary,
        ROOT,
        int(config["model"]["sample_points"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("workers", 0)),
        collate_fn=semantic_collate,
    )
    encoder_config = EEGConditionerConfig(**checkpoint["encoder_config"])
    model = DirectSemanticAlignmentModel(
        encoder_config,
        {slot: len(values) for slot, values in vocabulary.values.items()},
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    target_mean = checkpoint.get("target_mean")
    if target_mean is not None:
        target_mean = target_mean.to(device)
    target_index = read_tora_condition_index(resolve_path(data["tora_target_index"]))
    projector = None
    autoencoder = None
    if method == "tora_pca":
        projector = ToraPCAProjector.load(
            resolve_path(data["pca_projector"]), dim=encoder_config.latent_dim
        )
    elif method == "tora_autoencoder":
        autoencoder_checkpoint = torch.load(
            resolve_path(data["autoencoder_checkpoint"]), map_location=device, weights_only=False
        )
        autoencoder = ToraTextAutoencoder(
            ToraTextAutoencoderConfig(**autoencoder_checkpoint["config"])
        ).to(device)
        autoencoder.load_state_dict(autoencoder_checkpoint["state_dict"])
        autoencoder.eval()
    trial_dir = args.output_dir / "trials"
    video_dir = args.output_dir / "video_aggregated"
    trial_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = []
    groups: dict[str, list[tuple[Path, torch.Tensor]]] = {}
    mse_sum = cosine_sum = 0.0
    count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            output = model(batch["eeg"].to(device))
            latent = output["latent"]
            if target_mean is not None:
                latent = latent + target_mean
            targets = torch.stack(
                [load_target(target_index[video_id], method) for video_id in batch["video_id"]]
            ).to(device)
            mse_sum += float(F.mse_loss(latent, targets, reduction="sum"))
            cosine_sum += float(F.cosine_similarity(latent, targets, dim=-1).sum())
            count += latent.shape[0] * latent.shape[1]
            if projector:
                full_state = projector.decode(latent.float().cpu())
            elif autoencoder:
                full_state = autoencoder.decode(latent).float().cpu()
            else:
                full_state = latent.float().cpu()
            for index, (video_id, session) in enumerate(zip(batch["video_id"], batch["session"])):
                path = trial_dir / f"{video_id}_{session}.pt"
                torch.save(
                    {
                        "schema_version": 1,
                        "video_id": video_id,
                        "caption": f"EEG-predicted semantic condition for {video_id}",
                        "hidden_state": full_state[index],
                        "source_checkpoint": str(args.checkpoint.resolve()),
                        "source_session": session,
                        "source_method": method,
                    },
                    path,
                )
                groups.setdefault(video_id, []).append((path, latent[index].float().cpu()))
                trial_rows.append(
                    {
                        "video_id": video_id,
                        "session": session,
                        "condition_path": str(path.resolve()),
                        "target_space_mse": float(F.mse_loss(latent[index], targets[index])),
                        "target_space_token_cosine": float(
                            F.cosine_similarity(latent[index], targets[index], dim=-1).mean()
                        ),
                    }
                )
            if args.smoke:
                break
    video_rows = []
    retrieval_predictions = []
    retrieval_targets = []
    for video_id, items in groups.items():
        paths = [item[0] for item in items]
        states = [load_tora_condition(path).hidden_state for path in paths]
        mean_latent = torch.stack([item[1] for item in items]).mean(dim=0)
        target = load_target(target_index[video_id], method)
        retrieval_predictions.append(mean_latent.mean(dim=0))
        retrieval_targets.append(target.mean(dim=0))
        path = video_dir / f"{video_id}.pt"
        torch.save(
            {
                "schema_version": 1,
                "video_id": video_id,
                "caption": f"Session-aggregated EEG semantic condition for {video_id}",
                "hidden_state": torch.stack(states).mean(dim=0),
                "source_checkpoint": str(args.checkpoint.resolve()),
                "source_sessions": [item.stem.rsplit("_", 1)[-1] for item in paths],
                "source_method": method,
            },
            path,
        )
        video_rows.append(
            {
                "video_id": video_id,
                "condition_path": str(path.resolve()),
                "trial_count": len(paths),
                "target_space_mse": float(F.mse_loss(mean_latent, target)),
                "target_space_token_cosine": float(
                    F.cosine_similarity(mean_latent, target, dim=-1).mean()
                ),
            }
        )
    predicted_matrix = F.normalize(torch.stack(retrieval_predictions), dim=-1)
    target_matrix = F.normalize(torch.stack(retrieval_targets), dim=-1)
    ranking = (predicted_matrix @ target_matrix.t()).argsort(dim=1, descending=True)
    truth = torch.arange(len(video_rows)).unsqueeze(1)
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "method": method,
        "partition": args.partition,
        "target_space_mse": mse_sum / max(1, count * encoder_config.latent_dim),
        "target_space_token_cosine": cosine_sum / max(1, count),
        "trial_count": len(trial_rows),
        "video_count": len(video_rows),
        "single_trial_primary": True,
        "session_aggregation_is_secondary": True,
        "video_retrieval_top1": float((ranking[:, :1] == truth).any(dim=1).float().mean()),
        "video_retrieval_top5": float(
            (ranking[:, : min(5, ranking.shape[1])] == truth).any(dim=1).float().mean()
        ),
    }
    (args.output_dir / "trial_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trial_rows), encoding="utf-8"
    )
    (args.output_dir / "video_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in video_rows), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
