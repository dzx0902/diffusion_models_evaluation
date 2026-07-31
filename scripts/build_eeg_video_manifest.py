"""Validate and export explicit EEG, caption, and video mappings by video_id.

The source data contains the same videos in three sessions with different playback
orders. This script never relies on a directory listing or row position to join
modalities: all joins use the normalized ``NN-NNN`` video_id.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID_RE = re.compile(r"^(?P<category>\d{2})[-_](?P<index>\d{3})(?:\.mp4)?$")
CAPTION_RE = re.compile(r"^(?P<index>\d+)\.\s+(?P<caption>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and export video_id-based EEG/caption/video manifests."
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--subject", default="chentianlin")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "manifests",
        help="Directory for video_manifest.jsonl and eeg_trials.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(f"[manifest] {message}")


def normalized_video_id(value: str) -> str:
    match = VIDEO_ID_RE.fullmatch(value.strip())
    if not match:
        fail(f"invalid video identifier or filename: {value!r}")
    return f"{match.group('category')}-{match.group('index')}"


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def require_single(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        fail(f"expected exactly one {description}, found {len(paths)}: {paths}")
    return paths[0]


def load_captions(captions_dir: Path) -> dict[str, dict[str, str]]:
    if not captions_dir.is_dir():
        fail(f"caption directory does not exist: {captions_dir}")
    captions: dict[str, dict[str, str]] = {}
    for path in sorted(captions_dir.glob("*.txt")):
        prefix_match = re.match(r"^(\d{2})_", path.name)
        if not prefix_match:
            fail(f"caption filename must start with NN_: {path}")
        category = prefix_match.group(1)
        if category in captions:
            fail(f"duplicate caption file for category {category}: {path}")
        entries: dict[str, str] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            match = CAPTION_RE.fullmatch(line.strip())
            if not match:
                fail(f"invalid caption line {path}:{line_number}: {line!r}")
            video_id = f"{category}-{int(match.group('index')):03d}"
            if video_id in entries:
                fail(f"duplicate caption for {video_id}: {path}")
            entries[video_id] = match.group("caption").strip()
        if not entries:
            fail(f"caption file is empty: {path}")
        captions[category] = entries
    if not captions:
        fail(f"no caption files found in {captions_dir}")
    return captions


def load_videos(videos_dir: Path) -> dict[str, Path]:
    if not videos_dir.is_dir():
        fail(f"video directory does not exist: {videos_dir}")
    videos: dict[str, Path] = {}
    for path in sorted(videos_dir.rglob("*.mp4")):
        video_id = normalized_video_id(path.name)
        if not path.parent.name.startswith(video_id[:2]):
            fail(f"video category directory conflicts with filename id {video_id}: {path}")
        if video_id in videos:
            fail(f"duplicate video_id {video_id}: {videos[video_id]} and {path}")
        videos[video_id] = path
    if not videos:
        fail(f"no .mp4 files found in {videos_dir}")
    return videos


def scalar_float(value: Any) -> float:
    return float(np.asarray(value).item())


def load_session(session_dir: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    eeg_dir = session_dir / "EEG"
    npz_path = require_single(list(eeg_dir.glob("eeg_data.npz")), f"eeg_data.npz in {eeg_dir}")
    metadata_path = require_single(list(eeg_dir.glob("*_trial_metadata.csv")), f"trial metadata in {eeg_dir}")
    session_name = session_dir.name

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    metadata: dict[str, tuple[int, dict[str, str]]] = {}
    for row_index, row in enumerate(metadata_rows):
        video_id = normalized_video_id(row.get("video_id", ""))
        if video_id in metadata:
            fail(f"{session_name}: duplicate metadata video_id {video_id}")
        metadata[video_id] = (row_index, row)

    with np.load(npz_path, allow_pickle=False) as payload:
        required = {"eeg", "mask", "length", "filename", "sfreq"}
        missing = required - set(payload.files)
        if missing:
            fail(f"{session_name}: NPZ missing keys {sorted(missing)}")
        eeg_shape = payload["eeg"].shape
        mask = payload["mask"]
        lengths = payload["length"]
        filenames = payload["filename"]
        sfreq = scalar_float(payload["sfreq"])

        if len(eeg_shape) != 3 or eeg_shape[0] != len(lengths):
            fail(f"{session_name}: invalid eeg shape {eeg_shape} for {len(lengths)} lengths")
        if mask.shape != (eeg_shape[0], eeg_shape[2]):
            fail(f"{session_name}: mask shape {mask.shape} != {(eeg_shape[0], eeg_shape[2])}")
        if len(filenames) != eeg_shape[0] or len(metadata) != eeg_shape[0]:
            fail(
                f"{session_name}: NPZ trials={eeg_shape[0]}, filenames={len(filenames)}, "
                f"metadata={len(metadata)}"
            )
        if np.any(lengths <= 0) or np.any(lengths > eeg_shape[2]):
            fail(f"{session_name}: invalid EEG lengths")
        if not np.array_equal(mask.sum(axis=1), lengths):
            fail(f"{session_name}: mask valid counts do not equal length")

        trials: dict[str, dict[str, Any]] = {}
        for trial_index, filename in enumerate(filenames.tolist()):
            video_id = normalized_video_id(str(filename))
            if video_id in trials:
                fail(f"{session_name}: duplicate NPZ video_id {video_id}")
            if video_id not in metadata:
                fail(f"{session_name}: NPZ video_id missing metadata: {video_id}")
            metadata_row_index, row = metadata[video_id]
            length = int(lengths[trial_index])
            if int(float(row["n_times"])) != length:
                fail(f"{session_name}/{video_id}: metadata n_times != NPZ length")
            if abs(float(row["sfreq"]) - sfreq) > 1e-6:
                fail(f"{session_name}/{video_id}: metadata sfreq != NPZ sfreq")
            if abs(float(row["duration_sec"]) - length / sfreq) > 1e-6:
                fail(f"{session_name}/{video_id}: metadata duration_sec != length/sfreq")
            trials[video_id] = {
                "session": session_name,
                "npz_path": relative(npz_path),
                "trial_index": trial_index,
                "metadata_path": relative(metadata_path),
                "metadata_row": metadata_row_index,
                "length_samples": length,
                "duration_sec": length / sfreq,
                "sfreq": sfreq,
                "order_index": int(row["order_index"]),
                "playback_order_index": int(row["playback_order_index"]),
                "sorted_index": int(row["sorted_index"]),
            }
    if set(trials) != set(metadata):
        fail(f"{session_name}: metadata and NPZ video_id sets differ")
    return session_name, trials


def write_outputs(
    output_dir: Path,
    video_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "video_manifest.jsonl"
    trial_path = output_dir / "eeg_trials.csv"
    existing = [path for path in (video_path, trial_path) if path.exists()]
    if existing and not overwrite:
        fail(f"output exists; pass --overwrite: {existing}")
    with video_path.open("w", encoding="utf-8") as handle:
        for row in video_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fieldnames = [
        "subject", "video_id", "category_id", "caption", "caption_path", "video_path",
        "session", "npz_path", "trial_index", "metadata_path", "metadata_row",
        "length_samples", "duration_sec", "sfreq", "order_index", "playback_order_index", "sorted_index",
    ]
    with trial_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)
    print(f"[manifest] wrote {video_path} ({len(video_rows)} videos)")
    print(f"[manifest] wrote {trial_path} ({len(trial_rows)} EEG trials)")


def main() -> None:
    args = parse_args()
    captions_dir = args.data_root / "captions"
    videos_dir = args.data_root / "videos"
    subject_dir = args.data_root / "EEG_and_EYE" / args.subject
    captions = load_captions(captions_dir)
    videos = load_videos(videos_dir)
    caption_rows = {video_id: (category, caption) for category, rows in captions.items() for video_id, caption in rows.items()}
    if set(caption_rows) != set(videos):
        fail(
            f"caption/video id sets differ: missing captions={sorted(set(videos) - set(caption_rows))[:10]}, "
            f"missing videos={sorted(set(caption_rows) - set(videos))[:10]}"
        )
    # ``session_average`` is intentionally excluded. It has no trial metadata and
    # contains information from all three sessions, so it would leak into a
    # session-held-out experiment.
    sessions = [
        path
        for path in sorted(subject_dir.glob("session*"))
        if path.is_dir() and re.fullmatch(r"session\d+", path.name)
    ]
    if not sessions:
        fail(f"no session directories found in {subject_dir}")
    session_trials = dict(load_session(session) for session in sessions)
    for session, trials in session_trials.items():
        if set(trials) != set(videos):
            fail(f"{session}: EEG/video id sets differ")

    video_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for video_id in sorted(videos):
        category_id, caption = caption_rows[video_id]
        session_entries = [session_trials[name][video_id] for name in sorted(session_trials)]
        video_rows.append(
            {
                "subject": args.subject,
                "video_id": video_id,
                "category_id": category_id,
                "caption": caption,
                "caption_path": relative(next(path for path in captions_dir.glob(f"{category_id}_*.txt"))),
                "video_path": relative(videos[video_id]),
                "sessions": session_entries,
            }
        )
        for session_entry in session_entries:
            trial_rows.append(
                {
                    "subject": args.subject,
                    "video_id": video_id,
                    "category_id": category_id,
                    "caption": caption,
                    "caption_path": relative(next(path for path in captions_dir.glob(f"{category_id}_*.txt"))),
                    "video_path": relative(videos[video_id]),
                    **session_entry,
                }
            )
    counts = Counter(row["category_id"] for row in video_rows)
    print(f"[manifest] validated {len(video_rows)} videos across {len(session_trials)} sessions: {dict(sorted(counts.items()))}")
    write_outputs(args.output_dir, video_rows, trial_rows, args.overwrite)


if __name__ == "__main__":
    main()
