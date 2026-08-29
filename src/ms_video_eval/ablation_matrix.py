"""Materialize and validate config-driven EEG semantic ablation jobs."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def fold_index(fold: str) -> int:
    try:
        return int(fold.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Fold must end in a numeric index: {fold}") from exc


def interpolate(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(variables)
    if isinstance(value, list):
        return [interpolate(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: interpolate(item, variables) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class AblationJob:
    variant: str
    method: str
    subject: str
    fold: str
    seed: int
    config_path: Path
    output_dir: Path


def materialize_jobs(matrix_path: Path, root: Path, output_root: Path) -> tuple[dict[str, Any], list[AblationJob]]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    protocol = matrix["protocol"]
    variants = matrix.get("variants", [])
    if not variants or len({item["id"] for item in variants}) != len(variants):
        raise ValueError("Ablation matrix requires unique variants")
    jobs: list[AblationJob] = []
    for variant in variants:
        if not bool(variant.get("enabled", True)):
            continue
        base_path = root / str(variant["base_config"])
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        for subject in protocol["subjects"]:
            for fold in protocol["folds"]:
                for seed in protocol["seeds"]:
                    variables = {
                        "variant": variant["id"],
                        "subject": subject,
                        "fold": fold,
                        "fold_index": fold_index(fold),
                        "seed": int(seed),
                    }
                    config = deep_merge(base, interpolate(variant.get("overrides", {}), variables))
                    run_template = variant.get("run_dir_template", protocol["run_dir_template"])
                    run_dir = Path(str(run_template).format_map(variables))
                    config["experiment"].update(
                        name=f"{variant['id']}_{subject}_{fold}_seed{seed}",
                        seed=int(seed),
                        output_dir=str(run_dir).replace("\\", "/"),
                        ablation_variant=str(variant["id"]),
                    )
                    config["data"].update(
                        trials=f"data/manifests/{subject}/eeg_trials.csv",
                        split_plan=f"outputs/eeg_wan/splits/{subject}_video_6fold_plan.json",
                        fold=fold,
                    )
                    if config["experiment"]["method"] in {"direct_tora_text", "tora_pca", "tora_autoencoder"}:
                        if config["experiment"]["method"] == "direct_tora_text":
                            config["data"]["tora_target_index"] = config["data"].pop(
                                "tora_target_index_template", "outputs/tora/text_cache/index.jsonl"
                            )
                        elif config["experiment"]["method"] == "tora_pca":
                            config["data"]["tora_target_index"] = config["data"].pop(
                                "tora_target_index_template",
                                f"outputs/tora/pca/fold{variables['fold_index']}/dim"
                                f"{config['model']['condition_dim']}/index.jsonl",
                            )
                            config["data"]["pca_projector"] = config["data"].pop(
                                "pca_projector_template",
                                f"outputs/tora/pca/fold{variables['fold_index']}/tora_text_pca_"
                                f"{max(1024, int(config['model']['condition_dim']))}.npz",
                            )
                        else:
                            config["data"]["tora_target_index"] = config["data"].pop(
                                "tora_target_index_template",
                                f"outputs/tora/autoencoder/fold{variables['fold_index']}/dim"
                                f"{config['model']['condition_dim']}/index.jsonl",
                            )
                            config["data"]["autoencoder_checkpoint"] = config["data"].pop(
                                "autoencoder_checkpoint_template",
                                f"outputs/tora/autoencoder/fold{variables['fold_index']}/dim"
                                f"{config['model']['condition_dim']}/best.pt",
                            )
                    config_path = output_root / variant["id"] / subject / fold / f"seed{seed}.yaml"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = config_path.with_suffix(".yaml.tmp")
                    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                    temporary.replace(config_path)
                    jobs.append(
                        AblationJob(
                            variant=str(variant["id"]),
                            method=str(config["experiment"]["method"]),
                            subject=str(subject),
                            fold=str(fold),
                            seed=int(seed),
                            config_path=config_path,
                            output_dir=root / run_dir,
                        )
                    )
    manifest = output_root / "jobs.jsonl"
    manifest.write_text(
        "".join(json.dumps({**job.__dict__, "config_path": str(job.config_path), "output_dir": str(job.output_dir)}) + "\n" for job in jobs),
        encoding="utf-8",
    )
    return matrix, jobs


def assert_matched_protocol(jobs: list[AblationJob]) -> None:
    grouped: dict[str, set[tuple[str, str, int]]] = {}
    for job in jobs:
        grouped.setdefault(job.variant, set()).add((job.subject, job.fold, job.seed))
    expected = next(iter(grouped.values()), set())
    mismatched = {key: value for key, value in grouped.items() if value != expected}
    if mismatched:
        raise ValueError(f"Ablation variants do not share a matched protocol: {sorted(mismatched)}")
