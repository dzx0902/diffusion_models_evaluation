"""Run official Alibaba Tora with one cached/predicted cross-attention state.

Invoke this adapter with ``torchrun`` inside the Tora environment.  Arguments
after ``--`` are forwarded unchanged to official ``sat/arguments.py``.  The
adapter wraps model construction and replaces only conditional
``c['crossattn']``; unconditional text conditioning and trajectory processing
remain on the official sampling path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.tora_conditioning import inject_tora_crossattn, load_tora_condition


def parse_adapter_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tora-repo", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("Pass official Tora arguments after --")
    return args, forwarded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    adapter_args, forwarded = parse_adapter_args()
    tora_repo = adapter_args.tora_repo.resolve()
    sat_root = tora_repo / "sat"
    if not (sat_root / "sample_video.py").is_file():
        raise FileNotFoundError(f"Official Tora sat/sample_video.py not found under {tora_repo}")
    condition_path = adapter_args.condition.resolve()
    condition = load_tora_condition(condition_path)
    os.chdir(sat_root)
    sys.path.insert(0, str(sat_root))

    import sample_video  # type: ignore[import-not-found]

    original_get_model = sample_video.get_model
    injection_count = 0

    def wrapped_get_model(args: Any, model_cls: Any):  # type: ignore[no-untyped-def]
        nonlocal injection_count
        model = original_get_model(args, model_cls)
        conditioner = model.conditioner
        original_get_conditioning = conditioner.get_unconditional_conditioning

        def wrapped_get_conditioning(
            self: Any,
            batch_c: Any,
            batch_uc: Any = None,
            force_uc_zero_embeddings: Any = None,
        ):  # type: ignore[no-untyped-def]
            nonlocal injection_count
            c, uc = original_get_conditioning(
                batch_c,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            c = inject_tora_crossattn(c, condition.hidden_state)
            injection_count += 1
            return c, uc

        conditioner.get_unconditional_conditioning = types.MethodType(
            wrapped_get_conditioning, conditioner
        )
        return model

    sample_video.get_model = wrapped_get_model
    tora_args = sample_video.get_args(forwarded)
    if hasattr(tora_args, "deepspeed_config"):
        del tora_args.deepspeed_config
    tora_args.model_config.first_stage_config.params.cp_size = 1
    tora_args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    tora_args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    tora_args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    tora_args.model_config.en_and_decode_n_samples_a_time = 1
    sample_video.sampling_main(tora_args, model_cls=sample_video.SATVideoDiffusionEngine)
    if injection_count < 1:
        raise RuntimeError("Tora sampling finished without injecting a semantic condition")
    if int(os.environ.get("RANK", "0")) == 0:
        adapter_args.adapter_report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "adapter": "official_tora_crossattn_override",
            "tora_repo": str(tora_repo),
            "condition_path": str(condition_path),
            "condition_sha256": sha256(condition_path),
            "video_id": condition.video_id,
            "caption": condition.caption,
            "condition_shape": list(condition.hidden_state.shape),
            "injection_count": injection_count,
            "forwarded_args": forwarded,
            "trajectory_path_is_official_tora_argument": True,
            "unconditional_condition_was_not_replaced": True,
        }
        adapter_args.adapter_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[tora-adapter] report: {adapter_args.adapter_report}")


if __name__ == "__main__":
    main()
