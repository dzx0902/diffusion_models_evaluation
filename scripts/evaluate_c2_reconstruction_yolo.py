"""Reuse benchmark YOLO for four-arm C2 train reconstruction object diagnostics."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ms_video_eval.detector import FrameDetector, load_detector_settings
from ms_video_eval.semantic_schema import CORE_ENTITIES_BY_CATEGORY
from ms_video_eval.video_io import extract_frames

ARMS = ("text_full", "text_pca", "eeg_matched", "eeg_swapped")
METRICS = ("entity_coverage", "mean_entity_presence", "full_entity_frame_rate", "yolo_entity_score")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_write(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def csv_write(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(payload, expected):
    frames = payload["detections"]
    if not frames or not expected:
        raise ValueError("Require nonempty frames and expected entities")
    found = [{d["class_name"] for d in frame["detections"]} for frame in frames]
    rates = {e: sum(e in f for f in found)/len(frames) for e in expected}
    coverage = sum(v > 0 for v in rates.values())/len(expected)
    presence = statistics.mean(rates.values())
    full = sum(set(expected).issubset(f) for f in found)/len(frames)
    return {"sampled_frames": len(frames), "entity_coverage": coverage,
            "mean_entity_presence": presence, "full_entity_frame_rate": full,
            "yolo_entity_score": .5*coverage + .3*presence + .2*full,
            "per_entity_frame_rate": json.dumps(rates, sort_keys=True),
            "detected_entities": ";".join(sorted(set.union(*found)))}


def annotate(payload, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for frame in payload["detections"]:
        image = cv2.imread(frame["frame_path"])
        if image is None:
            raise ValueError(f"Unreadable extracted frame: {frame['frame_path']}")
        for d in frame["detections"]:
            x1, y1, x2, y2 = map(round, d["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"{d['class_name']} [{d['raw_class_name']}] {d['confidence']:.2f}"
            cv2.putText(image, label, (max(0,x1), max(18,y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0,220,0), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(destination / Path(frame["frame_path"]).name), image):
            raise IOError("Could not save annotated frame")


def summarize(rows):
    summaries, paired = [], []
    for group in ("all", "non_flower_categories"):
        selected = [r for r in rows if group == "all" or r["category"] not in ("04", "05")]
        for arm in ARMS:
            items = [r for r in selected if r["variant"] == arm]
            for metric in METRICS:
                values = [r[metric] for r in items]
                summaries.append({"group": group, "variant": arm, "metric": metric,
                                  "n": len(values), "mean": statistics.mean(values),
                                  "std": statistics.stdev(values) if len(values)>1 else 0.})
        for left, right in (("text_pca", "text_full"), ("eeg_matched", "text_pca"), ("eeg_matched", "eeg_swapped")):
            a = {(r["video_id"], r["generation_seed"]): r for r in selected if r["variant"] == left}
            b = {(r["video_id"], r["generation_seed"]): r for r in selected if r["variant"] == right}
            if a.keys() != b.keys():
                raise ValueError("Unmatched four-arm comparison")
            for metric in METRICS:
                differences = [a[k][metric]-b[k][metric] for k in sorted(a)]
                paired.append({"group": group, "left": left, "right": right, "metric": metric,
                               "n": len(differences), "left_minus_right": statistics.mean(differences),
                               "left_higher": sum(d > 1e-8 for d in differences)})
    return summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eeg_semantic/c2_train_reconstruction"))
    parser.add_argument("--settings", type=Path, default=Path("configs/ms_eval_settings.yaml"))
    parser.add_argument("--model", type=Path, help="Override weights with an existing local YOLO checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-every", type=int, default=4)
    args = parser.parse_args()
    if args.sample_every < 1:
        parser.error("sample-every must be positive")
    output = args.output_dir or args.root / "yolo"
    protocol = json.loads((args.root / "protocol.json").read_text())
    generation = json.loads((args.root / "generation_protocol.json").read_text())
    if generation["protocol_sha256"] != sha(args.root / "protocol.json"):
        raise ValueError("Generation and selection protocol do not match")
    selections = {r["video_id"]: r for r in protocol["selection"]}
    expected_cells = {(v, s) for v in selections for s in generation["generation_seeds"]}
    settings = load_detector_settings(args.settings)
    model = args.model or Path(settings.model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Local YOLO weights missing: {model}. Use --model PATH; no automatic download.")
    settings.model_path = str(model.resolve())
    if not set(e for v in selections for e in CORE_ENTITIES_BY_CATEGORY[v[:2]]).issubset(settings.class_aliases or {}):
        raise ValueError("Detector aliases omit expected core entities")
    jobs = []
    for arm in ARMS:
        path = args.root / "generated" / arm / "generation_manifest.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) != len(expected_cells) or {(r["video_id"], r["generation_seed"]) for r in records} != expected_cells:
            raise ValueError(f"Missing/duplicate generation cells: {arm}")
        for row in records:
            if row["variant"] != arm:
                raise ValueError(f"Wrong arm in generation manifest: {arm}")
            if row["status"] not in ("success", "skipped_existing") or not Path(row["output"]).is_file():
                raise ValueError(f"Unfinished generation: {row['output']}")
            jobs.append((arm, row, sha(Path(row["output"]))))
    signature = {"settings": asdict(settings), "weights_sha256": sha(model), "sample_every": args.sample_every,
                 "protocol_sha256": sha(args.root / "protocol.json"),
                 "videos": [{"arm": arm, "id": r["video_id"], "seed": r["generation_seed"], "sha256": h} for arm,r,h in jobs]}
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "settings.json"
    if lock.exists() and json.loads(lock.read_text()) != signature:
        raise ValueError("Detection settings/videos changed; use a different --output-dir")
    json_write(signature, lock)
    detector = FrameDetector(settings)
    rows, frame_rows = [], []
    for number, (arm, row, _) in enumerate(jobs, 1):
        video = row["video_id"]
        label = f"{video}_seed{row['generation_seed']}"
        folder = output / arm / label
        detection_path = folder / "detections.json"
        if detection_path.exists():
            payload = json.loads(detection_path.read_text())
        else:
            files = extract_frames(Path(row["output"]), folder / "frames", args.sample_every)
            if len(files) != len(range(0, 49, args.sample_every)):
                raise ValueError(f"Unexpected frame count for 49-frame control: {row['output']}")
            payload = detector.detect_frames(folder / "frames", detection_path)
        if len(payload["detections"]) != len(range(0, 49, args.sample_every)):
            raise ValueError(f"Missing decoded/detected frames: {label}")
        annotate(payload, folder / "annotated")
        expected = CORE_ENTITIES_BY_CATEGORY[video[:2]]
        common = {"variant": arm, "video_id": video, "generation_seed": row["generation_seed"]}
        result = {**common, "category": video[:2], "expected_entities": ";".join(expected),
                  "flower_proxy": "flower" in expected, **score(payload, expected)}
        rows.append(result)
        for frame in payload["detections"]:
            found = {d["class_name"] for d in frame["detections"]}
            frame_rows.append({**common, "source_frame_index": frame["frame_idx"]*args.sample_every,
                               "detected_entities": ";".join(sorted(found)), "all_expected_present": set(expected).issubset(found),
                               "detections": json.dumps(frame["detections"]), "frame_path": frame["frame_path"]})
        print(f"[c2-yolo] {number}/{len(jobs)} {arm}/{label} score={result['yolo_entity_score']:.4f}", flush=True)
    summary, paired = summarize(rows)
    csv_write(rows, output / "video_scores.csv")
    csv_write(frame_rows, output / "frame_scores.csv")
    csv_write(summary, output / "summary.csv")
    csv_write(paired, output / "paired_deltas.csv")
    report = ["# C2 training reconstruction: YOLO object-presence diagnostic", "",
              "Not a held-out semantic reconstruction result. Same-category swaps share the same core entities.",
              "flower is a potted-plant/vase proxy; inspect non_flower_categories separately.",
              "Score = 0.5 coverage + 0.3 mean presence + 0.2 full-frame rate; heuristic, not accuracy.",
              "No action/relation/identity or fine-semantic claims; no frame-independent significance tests.", "",
              "| group | arm | n | mean entity score |", "|---|---|---:|---:|"]
    for item in summary:
        if item["metric"] == "yolo_entity_score":
            report.append(f"| {item['group']} | {item['variant']} | {item['n']} | {item['mean']:.4f} |")
    (output / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    json_write({"status": "COMPLETE", "videos": len(rows), "sampled_frames": len(frame_rows)}, output / "completed.json")
    print("\n".join(report), flush=True)
    print(f"[c2-yolo] results: {output}", flush=True)


if __name__ == "__main__":
    main()
