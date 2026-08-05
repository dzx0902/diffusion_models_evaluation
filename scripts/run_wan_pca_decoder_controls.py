"""Generate native-text and exact PCA round-trip controls for selected videos.

No EEG model is loaded.  Each PCA condition encodes the manifest caption with
Wan's native T5, applies a train-fold projector round trip, then uses that
reconstructed context for Wan generation.  This isolates decoder-space loss.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native and PCA-decoder Wan controls for selected manifest videos.")
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 768, 1024, 1536])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--task", default="ti2v-5B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-model", choices=["True", "False"], default="True")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--enable-tf32", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["video_id"]): row for row in rows}


def run(command: list[str]) -> None:
    print("[wan-pca-control] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not (args.wan_repo / "generate.py").exists():
        raise FileNotFoundError(args.wan_repo / "generate.py")
    if not args.projector.exists():
        raise FileNotFoundError(args.projector)
    records = read_manifest(args.manifest)
    missing = [video_id for video_id in args.video_ids if video_id not in records]
    if missing:
        raise KeyError(f"video_id values absent from manifest: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ROOT / "scripts" / "adapters" / "wan_projected_generate.py"
    ckpt_dir = args.wan_repo / "Wan2.2-TI2V-5B"

    variants: list[tuple[str, int]] = [("native_text", 0)] + [(f"pca_{dim}", dim) for dim in args.dims]
    for video_id in args.video_ids:
        prompt = str(records[video_id]["caption"])
        for label, dim in variants:
            output = args.output_dir / f"{video_id}_{label}_seed{args.seed}.mp4"
            if args.skip_existing and output.exists() and output.stat().st_size > 0:
                print(f"[wan-pca-control] skip existing {output}", flush=True)
                continue
            command = [sys.executable, str(wrapper), "--wan-repo", str(args.wan_repo)]
            if dim:
                command.extend(["--projector", str(args.projector), "--project-dim", str(dim), "--report-error"])
            else:
                command.append("--disable-projection")
            if args.enable_tf32:
                command.append("--enable-tf32")
            command.extend(
                [
                    "--",
                    "--task", args.task,
                    "--size", args.size,
                    "--ckpt_dir", str(ckpt_dir),
                    "--offload_model", args.offload_model,
                    "--convert_model_dtype",
                    "--t5_cpu",
                    "--base_seed", str(args.seed),
                    "--prompt", prompt,
                    "--save_file", str(output),
                ]
            )
            run(command)
    print(f"[wan-pca-control] outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
