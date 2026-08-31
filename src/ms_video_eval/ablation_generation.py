"""Unified caption/latent generation matrix for EEG semantic ablations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class AblationGenerator:
    id: str
    condition_kinds: tuple[str, ...]
    command: tuple[str, ...]
    required_paths: tuple[str, ...] = ()
    requires_trajectory: bool = False
    enabled: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AblationGenerator":
        return cls(
            id=str(values["id"]),
            condition_kinds=tuple(map(str, values.get("condition_kinds", ("caption",)))),
            command=tuple(map(str, values["command"])),
            required_paths=tuple(map(str, values.get("required_paths", ()))),
            requires_trajectory=bool(values.get("requires_trajectory", False)),
            enabled=bool(values.get("enabled", True)),
        )


def load_generators(path: Path) -> dict[str, AblationGenerator]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    generators = [AblationGenerator.from_mapping(item) for item in payload.get("generators", [])]
    if not generators or len({item.id for item in generators}) != len(generators):
        raise ValueError("Generator config must contain unique generator IDs")
    return {item.id: item for item in generators}


def read_conditions(path: Path, kind: str) -> list[dict[str, str]]:
    if kind == "caption":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("video_aggregation_records", payload if isinstance(payload, list) else None)
        if not isinstance(rows, list):
            raise ValueError("Caption input must contain video_aggregation_records")
        result = [{"video_id": str(row["video_id"]), "prompt": str(row["caption"])} for row in rows]
    elif kind == "tora_state":
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                result.append({"video_id": str(row["video_id"]), "condition": str(row["condition_path"])})
    else:
        raise ValueError(f"Unknown condition kind: {kind}")
    ids = [row["video_id"] for row in result]
    if not result or len(ids) != len(set(ids)):
        raise ValueError("Condition input must contain one unique row per video")
    return result


def read_trajectories(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id in rows:
                raise ValueError(f"Duplicate trajectory for {video_id}")
            values = row.get("trajectory_paths")
            if values is None:
                values = [row["trajectory_path"]]
            rows[video_id] = [str(value) for value in values]
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_tokens(tokens: Iterable[str], variables: Mapping[str, Any]) -> list[str]:
    result = []
    for token in tokens:
        if token.startswith("{") and token.endswith("}"):
            key = token[1:-1]
            if isinstance(variables.get(key), (list, tuple)):
                result.extend(map(str, variables[key]))
                continue
        result.append(token.format_map(variables))
    return result


def run_generation_matrix(
    rows: list[dict[str, str]],
    kind: str,
    generators: Mapping[str, AblationGenerator],
    selected: Iterable[str],
    seeds: Iterable[int],
    output_root: Path,
    variables: Mapping[str, Any],
    trajectories: Mapping[str, list[str]] | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    """Run a matched generator matrix and record every command/result."""

    trajectories = trajectories or {}
    selected = list(selected)
    seeds = list(seeds)
    unknown = set(selected) - set(generators)
    if unknown:
        raise KeyError(f"Unknown generators: {sorted(unknown)}")

    # Validate every pending job before launching the first expensive model.
    # This prevents a late generator with an incomplete checkpoint tree from
    # failing only after earlier generators have already consumed GPU hours.
    if not dry_run:
        missing_by_generator: dict[str, set[Path]] = {}
        for generator_id in selected:
            spec = generators[generator_id]
            if not spec.enabled:
                raise ValueError(f"Generator {generator_id} is disabled")
            if kind not in spec.condition_kinds:
                raise ValueError(f"Generator {generator_id} does not support {kind}")
            for row in rows:
                video_id = row["video_id"]
                if spec.requires_trajectory and video_id not in trajectories:
                    raise KeyError(f"Missing fixed trajectory for {video_id}")
                raw_trajectories = trajectories.get(video_id, [])
                trajectory_paths = (
                    [raw_trajectories]
                    if isinstance(raw_trajectories, str)
                    else list(raw_trajectories)
                )
                for seed in seeds:
                    output = output_root / generator_id / f"{video_id}_seed{seed}.mp4"
                    if skip_existing and output.is_file() and output.stat().st_size:
                        continue
                    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
                    values = {
                        **variables,
                        **row,
                        "video_id": video_id,
                        # Keep paired methods on the same deterministic seed
                        # while giving generation repeats distinct RNG states.
                        "seed": int(variables.get("training_seed", 0)) + int(seed),
                        "generation_seed": seed,
                        "trajectory": (trajectory_paths or [""])[0],
                        "trajectory_paths": trajectory_paths,
                        "output": str(temporary),
                    }
                    required = [
                        Path(value) for value in format_tokens(spec.required_paths, values)
                    ]
                    missing = {path for path in required if not path.exists()}
                    if missing:
                        missing_by_generator.setdefault(generator_id, set()).update(missing)
        if missing_by_generator:
            details = "; ".join(
                f"{generator_id}: {', '.join(map(str, sorted(paths)))}"
                for generator_id, paths in missing_by_generator.items()
            )
            raise FileNotFoundError(f"Generation preflight found missing inputs: {details}")

    manifest: list[dict[str, Any]] = []
    for generator_id in selected:
        spec = generators[generator_id]
        if not spec.enabled:
            raise ValueError(f"Generator {generator_id} is disabled")
        if kind not in spec.condition_kinds:
            raise ValueError(f"Generator {generator_id} does not support {kind}")
        for row in rows:
            video_id = row["video_id"]
            if spec.requires_trajectory and video_id not in trajectories:
                raise KeyError(f"Missing fixed trajectory for {video_id}")
            raw_trajectories = trajectories.get(video_id, [])
            trajectory_paths = (
                [raw_trajectories] if isinstance(raw_trajectories, str) else list(raw_trajectories)
            )
            for seed in seeds:
                output = output_root / generator_id / f"{video_id}_seed{seed}.mp4"
                temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
                values = {
                    **variables,
                    **row,
                    "video_id": video_id,
                    "seed": int(variables.get("training_seed", 0)) + int(seed),
                    "generation_seed": seed,
                    "trajectory": (trajectory_paths or [""])[0],
                    "trajectory_paths": trajectory_paths,
                    "output": str(temporary),
                }
                command = format_tokens(spec.command, values)
                record: dict[str, Any] = {
                    "schema_version": 1,
                    "generator": generator_id,
                    "condition_kind": kind,
                    "video_id": video_id,
                    # Keep training and stochastic generation seeds distinct in
                    # every downstream paired-analysis record.
                    "seed": int(variables.get("training_seed", seed)),
                    "generation_seed": int(seed),
                    "output": str(output),
                    "command": command,
                    "trajectory": (trajectory_paths or [None])[0],
                    "trajectory_paths": trajectory_paths,
                    "trajectory_sha256": None,
                    "trajectory_sha256s": [],
                    "status": "planned",
                    **{
                        key: variables[key]
                        for key in ("variant", "subject", "fold")
                        if key in variables
                    },
                }
                if trajectory_paths and all(Path(path).is_file() for path in trajectory_paths):
                    hashes = [file_sha256(Path(path)) for path in trajectory_paths]
                    record["trajectory_sha256"] = hashes[0]
                    record["trajectory_sha256s"] = hashes
                if skip_existing and output.is_file() and output.stat().st_size:
                    record["status"] = "skipped_existing"
                elif dry_run:
                    record["status"] = "dry_run"
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if temporary.exists():
                        temporary.unlink()
                    environment = dict(os.environ)
                    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
                    completed = subprocess.run(command, check=False, env=environment)
                    if completed.returncode or not temporary.is_file() or not temporary.stat().st_size:
                        record["status"] = "failed"
                        record["returncode"] = completed.returncode
                        manifest.append(record)
                        raise RuntimeError(f"Generation failed for {generator_id}/{video_id}/seed{seed}")
                    os.replace(temporary, output)
                    record["status"] = "success"
                manifest.append(record)
    return manifest
