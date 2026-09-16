"""Four paired train-memory controls; NOT an unseen-video reconstruction benchmark."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts.run_c2_deep import load, predictions
from scripts.run_c2_residual import digest
from scripts.train_compact_tora_alignment import atomic_json, atomic_save
from ms_video_eval.c2_deep import DeepC2, validate_splits
from ms_video_eval.c2_residual import normalize_sessions, decode
from ms_video_eval.tora_conditioning import ToraPCAProjector, read_tora_condition_index, load_tora_condition
from ms_video_eval.ablation_generation import load_generators, read_conditions, run_generation_matrix

ARMS = ("text_full", "text_pca", "eeg_matched", "eeg_swapped")


def select_pairs(package):
    """Lexicographic train IDs; require different captions to make swaps meaningful."""
    validate_splits(package)
    result = []
    for category in ("01", "02", "03", "04", "05", "06"):
        candidates = sorted((i for i in package["splits"]["train"]
                             if package["ids"][i].startswith(category + "-")),
                            key=lambda i: package["ids"][i])
        if not candidates:
            raise ValueError(f"No train videos for {category}")
        first = candidates[0]
        second = next((i for i in candidates[1:] if package["captions"][i] != package["captions"][first]), None)
        if second is None:
            raise ValueError(f"Require two different train captions in {category}")
        result.extend([first, second])
    return result


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def prepare(args):
    package = load(args.prepared)
    selected = select_pairs(package)
    ckpt = load(args.checkpoint)
    completed = json.loads(args.checkpoint.with_name("completed.json").read_text())
    if ckpt["signature"]["variant"] != "long" or ckpt["signature"]["prepared_sha256"] != digest(args.prepared):
        raise ValueError("Require matching C2-v5 long checkpoint and prepared cache")
    if completed["signature"] != ckpt["signature"]:
        raise ValueError("Training completion/checkpoint mismatch")
    signature = {"prepared_sha256": digest(args.prepared), "checkpoint_sha256": digest(args.checkpoint),
                 "text_index_sha256": digest(args.text_index), "selection": "first_two_distinct_captions_per_category",
                 "selected_ids": [package["ids"][i] for i in selected], "schema_version": 1}
    marker = args.output_root / "protocol.json"
    if marker.exists():
        protocol = json.loads(marker.read_text())
        if protocol["signature"] != signature:
            raise ValueError("Diagnostic already prepared with different inputs; use another --output-root")
        verify_assets(args.output_root, protocol)
        print("[c2-diagnostic] prepared inputs verified", flush=True)
        return
    if (args.output_root / "generated").exists():
        raise ValueError("Generated files exist without preparation protocol; use another output root")
    index = read_tora_condition_index(args.text_index)
    full_states = []
    for i in selected:
        condition = load_tora_condition(Path(index[package["ids"][i]]["condition_path"]))
        if condition.video_id != package["ids"][i] or condition.caption.strip() != package["captions"][i].strip():
            raise ValueError("Full text cache ID/caption does not match prepared targets")
        full_states.append(condition.hidden_state.float())
    model = DeepC2(**ckpt["model_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    package = normalize_sessions(package, [0, 1, 2])
    pred = predictions(model, package, selected, args.device, 12)
    projector = ToraPCAProjector(**package["token_projector"])
    pca_states = decode(package["z"][selected], package["pca"])
    eeg_states = decode(pred, package["pca"])
    assets, selection = {}, []
    for position, i in enumerate(selected):
        swapped = position ^ 1
        video = package["ids"][i]
        donor = package["ids"][selected[swapped]]
        selection.append({"video_id": video, "caption": package["captions"][i],
                          "swapped_source_video_id": donor,
                          "swapped_caption": package["captions"][selected[swapped]], "partition": "train"})
        conditions = {"text_full": full_states[position], "text_pca": projector.decode(pca_states[position]),
                      "eeg_matched": projector.decode(eeg_states[position]),
                      "eeg_swapped": projector.decode(eeg_states[swapped])}
        for arm, hidden in conditions.items():
            if hidden.shape != (226, 4096) or not torch.isfinite(hidden).all():
                raise ValueError("Nonfinite/invalid Tora condition")
            path = args.output_root / "conditions" / arm / f"{video}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_save({"video_id": video, "hidden_state": hidden,
                         "caption": package["captions"][i], "control": arm,
                         "source_video_id": donor if arm == "eeg_swapped" else video,
                         "diagnostic_only": True}, path)
            assets[str(path.relative_to(args.output_root))] = digest(path)
    for arm in ARMS:
        path = args.output_root / "inputs" / f"{arm}.jsonl"
        write_jsonl([{"video_id": package["ids"][i], "condition_path": str(
            (args.output_root / "conditions" / arm / f"{package['ids'][i]}.pt").resolve())} for i in selected], path)
        assets[str(path.relative_to(args.output_root))] = digest(path)
    # Adapter expects a trajectory. This synthetic input is shared across all 48 cells;
    # it is NOT extracted from the stimulus and is NOT EEG-predicted motion.
    trajectory = args.output_root / "shared_static.txt"
    trajectory.write_text("128,128\n" * 49, encoding="utf-8")
    assets[trajectory.name] = digest(trajectory)
    atomic_json({"signature": signature, "selection": selection, "assets": assets,
                 "diagnostic_only": True, "partition": "train", "checkpoint_step": ckpt["step"],
                 "training_seed": ckpt["signature"]["seed"], "fold": package["protocol"]["fold"],
                 "trajectory_origin": "synthetic_shared_static", "independent_stimulus_alignment": "NOT_VERIFIED",
                 "note": "All arms use injected states. Static control may constrain motion; no generalization claim."}, marker)
    print(f"[c2-diagnostic] prepared 12 videos x 4 controls: {marker}", flush=True)


def verify_assets(root, protocol):
    for relative, expected in protocol["assets"].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Prepared condition/control changed or missing: {path}")


def generate(args):
    protocol = json.loads((args.output_root / "protocol.json").read_text())
    verify_assets(args.output_root, protocol)
    spec = load_generators(args.generator_config)["tora_injected"]
    command = list(spec.command)
    if command[:2] == ["conda", "run"] and "--no-capture-output" not in command:
        command.insert(2, "--no-capture-output")
    # Retain tested 49-frame Tora sampling; 48 intervals / 12fps = 4s.
    command[command.index("--fps") + 1] = "12"
    command[command.index("--num-frames") + 1] = "49"
    spec = replace(spec, command=tuple(command))
    signature = {"protocol_sha256": digest(args.output_root / "protocol.json"),
                 "generator_config_sha256": digest(args.generator_config), "command": command,
                 "models_root": str(args.models_root.resolve()), "generation_seeds": args.seeds}
    lock = args.output_root / "generation_protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != signature:
        raise ValueError("Generation settings changed; use another output root")
    # A dry-run never overwrites an actual generation manifest or locks the run.
    if not args.dry_run:
        atomic_json(signature, lock)
    trajectories = {row["video_id"]: [str((args.output_root / "shared_static.txt").resolve())]
                    for row in protocol["selection"]}
    for arm in ARMS:
        rows = read_conditions(args.output_root / "inputs" / f"{arm}.jsonl", "tora_state")
        records = run_generation_matrix(rows, "tora_state", {spec.id: spec}, [spec.id], args.seeds,
                    args.output_root / "generated" / arm,
                    {"repo_root": str(ROOT), "models_root": str(args.models_root.resolve()), "home": str(Path.home()),
                     "variant": arm, "subject": "chentianlin", "fold": protocol["fold"],
                     "training_seed": protocol["training_seed"]}, trajectories,
                    dry_run=args.dry_run, skip_existing=True)
        for row in records:
            row.update(partition="train", diagnostic_only=True, trajectory_origin="synthetic_shared_static")
        name = "dry_run_manifest.jsonl" if args.dry_run else "generation_manifest.jsonl"
        write_jsonl(records, args.output_root / "generated" / arm / name)
        print(f"[c2-diagnostic] {arm}: {len(records)} jobs; dry_run={args.dry_run}", flush=True)


def evaluate(args):
    manifests = [args.output_root / "generated" / arm / "generation_manifest.jsonl" for arm in ARMS]
    protocol = json.loads((args.output_root / "protocol.json").read_text())
    verify_assets(args.output_root, protocol)
    expected_ids = {r["video_id"] for r in protocol["selection"]}
    generation = json.loads((args.output_root / "generation_protocol.json").read_text())
    if generation["protocol_sha256"] != digest(args.output_root / "protocol.json"):
        raise ValueError("Generation/preparation protocol mismatch")
    expected = {(v, s) for v in expected_ids for s in generation["generation_seeds"]}
    for path in manifests:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != len(expected) or {(r["video_id"], r["generation_seed"]) for r in rows} != expected:
            raise ValueError(f"Incomplete/duplicate generation cells: {path}")
        if any(r["status"] not in ("success", "skipped_existing") or not Path(r["output"]).is_file() for r in rows):
            raise ValueError(f"Unfinished generation: {path}")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_eeg_ablation_videos.py"),
                    "--generation-manifests", *map(str, manifests), "--semantic-labels", str(args.semantic_labels),
                    "--output", str(args.output_root / "evaluation/video_metrics.csv"),
                    "--clip-model", str(args.clip_root.resolve()), "--device", args.device], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "generate", "evaluate"))
    parser.add_argument("--prepared", type=Path, default=Path("outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/eeg_semantic/c2_deep_v5/long/seed42/best.pt"))
    parser.add_argument("--text-index", type=Path, default=Path("outputs/tora/text_cache/index.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eeg_semantic/c2_train_reconstruction"))
    parser.add_argument("--generator-config", type=Path, default=Path("configs/eeg_semantic/generators.server.yaml"))
    parser.add_argument("--models-root", type=Path, default=Path(".ms_video_models"))
    parser.add_argument("--semantic-labels", type=Path, default=Path("outputs/semantic_labels/eeg_semantic_labels_v1.jsonl"))
    parser.add_argument("--clip-root", type=Path, default=Path(".ms_video_models/CLIP/clip-vit-base-patch32"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Generation seeds must be unique")
    if args.dry_run and args.stage != "generate":
        parser.error("--dry-run is available for generate only")
    torch.set_num_threads(4)
    args.output_root = args.output_root.resolve()
    {"prepare": prepare, "generate": generate, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
