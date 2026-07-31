"""Bind explicit video manifests to cached fixed [128, 512] Wan PCA targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a video_id -> Wan PCA target manifest.")
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--latent-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eeg_wan" / "wan_targets.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    latents: dict[str, dict] = {}
    for row in read_jsonl(args.latent_index):
        prompt = str(row.get("prompt", ""))
        if not prompt or prompt in latents:
            raise ValueError(f"Missing or duplicate prompt in latent index: {prompt[:100]!r}")
        path = Path(row["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        latents[prompt] = row
    records = read_jsonl(args.video_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = str(record["caption"])
            latent = latents.get(prompt)
            if latent is None:
                raise KeyError(f"No latent target for {record['video_id']}: {prompt[:100]!r}")
            handle.write(json.dumps({
                "video_id": record["video_id"],
                "prompt": prompt,
                "latent_path": str(Path(latent["path"]).resolve()),
                "tokens": int(latent["tokens"]),
            }, ensure_ascii=False) + "\n")
    print(f"[eeg-targets] wrote {args.output} ({len(records)} video targets)")


if __name__ == "__main__":
    main()
