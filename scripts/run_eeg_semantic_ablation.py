"""Materialize and execute the EEG semantic ablation training/prediction plan."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_matrix import assert_matched_protocol, materialize_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/eeg_semantic/ablation_matrix.yaml")
    parser.add_argument("--stage", choices=("materialize", "train", "predict", "generate", "all"), default="materialize")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models-root", type=Path, default=ROOT / ".ms_video_models")
    parser.add_argument("--trajectory-manifest", type=Path, default=None)
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("[eeg-ablation] " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    generated = ROOT / "outputs/eeg_semantic/experiment_configs"
    matrix, jobs = materialize_jobs(args.matrix, ROOT, generated)
    assert_matched_protocol(jobs)
    if args.variants:
        unknown = set(args.variants) - {job.variant for job in jobs}
        if unknown:
            raise KeyError(f"Unknown or disabled variants: {sorted(unknown)}")
        jobs = [job for job in jobs if job.variant in set(args.variants)]
    if args.stage == "materialize":
        print(f"[eeg-ablation] materialized {len(jobs)} jobs under {generated}")
        return
    for job in jobs:
        trainer = (
            ROOT / "scripts/train_eeg_semantic.py"
            if job.method in {"coarse_template", "structured_semantic"}
            else ROOT / "scripts/train_eeg_tora_alignment.py"
        )
        checkpoint = job.output_dir / "best.pt"
        if args.stage in {"train", "all"}:
            if args.skip_existing and checkpoint.is_file():
                print(f"[eeg-ablation] skip trained {job.variant}/{job.subject}/{job.fold}/seed{job.seed}")
            else:
                command = [sys.executable, str(trainer), "--config", str(job.config_path), "--device", args.device]
                last = job.output_dir / "last.pt"
                if last.is_file():
                    command.append("--resume")
                run(command, args.dry_run)
        if args.stage in {"predict", "all"}:
            if not args.dry_run and not checkpoint.is_file():
                raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
            if job.method in {"coarse_template", "structured_semantic"}:
                script = ROOT / "scripts/evaluate_eeg_semantic.py"
                output = job.output_dir / "test_semantic"
            else:
                script = ROOT / "scripts/predict_eeg_tora_conditions.py"
                output = job.output_dir / "test_conditions"
            if args.skip_existing and (output / ("predictions.json" if "semantic" in output.name else "report.json")).is_file():
                print(f"[eeg-ablation] skip prediction {job.variant}/{job.subject}/{job.fold}/seed{job.seed}")
                continue
            run(
                [sys.executable, str(script), "--checkpoint", str(checkpoint), "--partition", "test",
                 "--output-dir", str(output), "--device", args.device],
                args.dry_run,
            )
        if args.stage in {"generate", "all"}:
            protocol = matrix["protocol"]
            if job.method in {"coarse_template", "structured_semantic"}:
                condition_input = job.output_dir / "test_semantic/predictions.json"
                condition_kind = "caption"
                generators = list(protocol["caption_generators"])
            else:
                condition_input = job.output_dir / "test_conditions/video_index.jsonl"
                condition_kind = "tora_state"
                generators = list(protocol["latent_generators"])
            trajectory_manifest = args.trajectory_manifest or ROOT / protocol["trajectory_manifest"]
            output = ROOT / "outputs/eeg_semantic/generated" / job.variant / job.subject / job.fold / f"seed{job.seed}"
            command = [
                sys.executable, str(ROOT / "scripts/run_eeg_semantic_generation.py"),
                "--input", str(condition_input), "--condition-kind", condition_kind,
                "--generator-config", str(ROOT / protocol["generator_config"]),
                "--generators", *generators,
                "--trajectory-manifest", str(trajectory_manifest),
                "--models-root", str(args.models_root), "--output-dir", str(output),
                "--variant", job.variant, "--subject", job.subject, "--fold", job.fold,
                "--training-seed", str(job.seed),
                "--seeds", *[str(value) for value in protocol["generation_seeds"]],
            ]
            if args.skip_existing:
                command.append("--skip-existing")
            if args.dry_run:
                command.append("--dry-run")
            run(command, args.dry_run)
    print(f"[eeg-ablation] completed stage={args.stage} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
