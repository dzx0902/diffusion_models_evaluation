"""Evaluate generated videos into the ablation long-form metric schema."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.ablation_video_metrics import (
    CLIPFrameScorer, frame_diagnostics, parse_trajectory_points, sample_video,
    trajectory_direction_score,
)
from ms_video_eval.semantic_schema import load_semantic_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--clip-model", default=None, help="Local path or HF ID; omit to skip CLIP metrics.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    labels = {record.video_id: record for record in load_semantic_records(args.semantic_labels)}
    scorer = CLIPFrameScorer(args.clip_model, args.device) if args.clip_model else None
    rows = []
    for manifest in args.generation_manifests:
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for item in records:
            if item["status"] not in {"success", "skipped_existing"}:
                continue
            video_id = str(item["video_id"]); record = labels[video_id]
            frames = sample_video(Path(item["output"]), args.sample_every)
            metrics = frame_diagnostics(frames)
            trajectories = item.get("trajectory_paths") or ([item["trajectory"]] if item.get("trajectory") else [])
            if trajectories:
                scores = [
                    trajectory_direction_score(frames, parse_trajectory_points(Path(path)))
                    for path in trajectories
                ]
                metrics["trajectory_direction_score"] = sum(scores) / len(scores)
            if scorer is not None:
                metrics["semantic_clip_score"] = scorer.score(frames, record.caption)
                if record.subjects:
                    metrics["subject_clip_score"] = scorer.score(frames, " and ".join(record.subjects))
                if record.objects:
                    metrics["object_clip_score"] = scorer.score(frames, " and ".join(record.objects))
                if record.fine_actions:
                    metrics["action_clip_score"] = scorer.score(frames, " and ".join(record.fine_actions))
            generator_route = str(item["generator"])
            # Native-caption Tora and injected-condition Tora use the same
            # video backbone and fixed trajectory. Normalize only the
            # comparison family while preserving the original route.
            comparison_generator = "tora" if generator_route == "tora_injected" else generator_route
            common = {"variant": item["variant"], "generator": comparison_generator,
                      "generator_route": generator_route,
                      "subject": item["subject"], "fold": item["fold"], "seed": item["seed"],
                      "generation_seed": item["generation_seed"],
                      "video_id": video_id}
            rows.extend({**common, "metric": key, "value": value} for key, value in metrics.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "generator", "generator_route", "subject", "fold", "seed", "generation_seed", "video_id", "metric", "value"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"[eeg-ablation-video-eval] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
