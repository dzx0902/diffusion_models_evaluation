"""Read-only cross-session manifest/NPZ/metadata and signal audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ms_video_eval.semantic_data import load_trial_rows
from ms_video_eval.semantic_schema import normalize_video_id


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def signal_statistics(signal, sfreq):
    valid = np.isfinite(signal)
    if not valid.all():
        return {"nonfinite_samples": int((~valid).sum())}
    freq, psd = welch(signal, fs=sfreq, nperseg=min(400, signal.shape[-1]), axis=-1)
    band = {}
    total = psd[:, (freq >= 1) & (freq < 45)].sum(1).clip(min=1e-30)
    for name, low, high in (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma_low", 30, 45)):
        band[name] = float((psd[:, (freq >= low) & (freq < high)].sum(1) / total).mean())
    std = signal.std(-1)
    centered = signal - signal.mean(-1, keepdims=True)
    return {"nonfinite_samples": 0, "rms_native_units": float(np.sqrt(np.mean(signal.astype(np.float64)**2))),
            "flat_channel_count": int((std == 0).sum()), "channel_mean": signal.mean(-1).tolist(),
            "channel_std": std.tolist(), "max_abs_native_units": float(np.abs(signal).max()),
            "samples_beyond_8_within_trial_std_fraction": float((np.abs(centered) > 8 * np.maximum(std[:, None], 1e-30)).mean()),
            "band_relative_power": band}


def audit(rows, events=None):
    errors, warnings, records, sessions = [], [], [], {}
    if not rows:
        errors.append("No selected first-six trials")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["session"], str(resolve(row["npz_path"])))].append(row)
    keys = Counter((row["video_id"], row["session"]) for row in rows)
    errors.extend(f"duplicate manifest key {key}" for key, count in keys.items() if count != 1)
    ids = sorted({row["video_id"] for row in rows})
    for video in ids:
        for session in ("session1", "session2", "session3"):
            if (video, session) not in keys:
                errors.append(f"missing trial {video}/{session}")
    event_map = {}
    if events is not None:
        for event in events:
            key = (event["session"], int(event["onset_sample"]))
            if key in event_map:
                errors.append(f"duplicate independent event {key}")
            event_map[key] = normalize_video_id(event["video_id"])
    metadata_cache, fingerprints = {}, defaultdict(list)
    event_compared = 0
    for (session, path), group in sorted(groups.items()):
        print(f"[session-audit] reading {session}: {path}", flush=True)
        try:
            with np.load(path, allow_pickle=False) as source:
                required = {"eeg", "filename", "length", "mask", "sfreq"}
                if not required <= set(source.files):
                    errors.append(f"{session}: missing NPZ keys {sorted(required-set(source.files))}")
                    continue
                eeg = source["eeg"]
                names, lengths, mask = source["filename"], source["length"], source["mask"]
                sfreq = float(source["sfreq"])
                channels = source["channel_names"].tolist() if "channel_names" in source.files else None
                if channels is None:
                    warnings.append(f"{session}: channel names unavailable")
                if sfreq != 200 or eeg.ndim != 3 or eeg.shape[1] != 62:
                    errors.append(f"{session}: expected 200 Hz and [trial,62,time], got {sfreq}, {eeg.shape}")
                    continue
                session_info = {"npz_path": path, "npz_keys": source.files, "sfreq": sfreq,
                                "shape": list(eeg.shape), "channel_names": channels,
                                "units": "unknown: native array units", "reference": "unknown", "filtering": "unknown"}
                if session in sessions:
                    errors.append(f"{session}: multiple NPZ files; inspect session grouping")
                sessions[session] = session_info
                npz_orders = {key: source[key] for key in ("order_index", "playback_order_index") if key in source.files}
                sorted_order = ("order_index" in npz_orders and
                                np.array_equal(npz_orders["order_index"], np.arange(1, len(eeg)+1)))
                session_info["npz_order_index_interpretation"] = "matches_one_based_array_order" if sorted_order else "not_array_order"
                used = set()
                for row in group:
                    video, index = row["video_id"], int(row["trial_index"])
                    prefix = f"{session}/{video}"
                    if index in used:
                        errors.append(f"{prefix}: reused trial_index {index}")
                    used.add(index)
                    if not 0 <= index < len(eeg):
                        errors.append(f"{prefix}: out-of-range trial index")
                        continue
                    if normalize_video_id(str(names[index])) != video:
                        errors.append(f"{prefix}: NPZ filename disagrees with manifest")
                    if int(lengths[index]) != int(row["length_samples"]) or int(lengths[index]) != 800:
                        errors.append(f"{prefix}: first-six length must be 800 and match manifest")
                    if not np.all(mask[index, :800]) or np.any(mask[index, 800:]):
                        errors.append(f"{prefix}: invalid or noncontiguous 4-second mask")
                    if row.get("sfreq") and float(row["sfreq"]) != sfreq:
                        errors.append(f"{prefix}: manifest sampling rate mismatch")
                    if row.get("duration_sec") and abs(float(row["duration_sec"]) - 4) > 1e-6:
                        errors.append(f"{prefix}: manifest duration mismatch")
                    record = {"session": session, "video_id": video, "trial_index": index}
                    metadata_path = row.get("metadata_path")
                    if metadata_path:
                        metadata_path = resolve(metadata_path)
                        if metadata_path not in metadata_cache:
                            metadata_cache[metadata_path] = read_csv(metadata_path)
                        metadata = metadata_cache[metadata_path]
                        position = int(row["metadata_row"])
                        if not 0 <= position < len(metadata):
                            errors.append(f"{prefix}: metadata row out of range")
                            continue
                        meta = metadata[position]
                        if normalize_video_id(meta["video_id"]) != video:
                            errors.append(f"{prefix}: metadata video_id mismatch")
                        for field, expected in (("sfreq", sfreq), ("n_times", 800), ("duration_sec", 4), ("n_eeg_channels", 62)):
                            if meta.get(field) and abs(float(meta[field]) - expected) > 1e-6:
                                errors.append(f"{prefix}: metadata {field} mismatch")
                        for field in ("order_index", "playback_order_index", "sorted_index"):
                            if row.get(field) and meta.get(field) and int(row[field]) != int(meta[field]):
                                errors.append(f"{prefix}: manifest/metadata {field} mismatch")
                            npz_compare_field = "sorted_index" if field == "order_index" and sorted_order else field
                            if field in npz_orders and meta.get(npz_compare_field) and int(npz_orders[field][index]) != int(meta[npz_compare_field]):
                                errors.append(f"{prefix}: NPZ/metadata {field} mismatch")
                        onset = int(meta["onset_sample"]) if meta.get("onset_sample") else None
                        record["onset_sample"] = onset
                        record["playback_order_index"] = meta.get("playback_order_index")
                        if onset is not None and meta.get("annotation_event_sample"):
                            delta = onset - int(meta["annotation_event_sample"])
                            record["onset_minus_annotation_samples"] = delta
                        if events is not None:
                            if onset is None or (session, onset) not in event_map:
                                errors.append(f"{prefix}: no independent event at metadata onset")
                            elif event_map[(session, onset)] != video:
                                errors.append(f"{prefix}: independent event video differs")
                            else:
                                event_compared += 1
                    else:
                        warnings.append(f"{prefix}: metadata path unavailable")
                        if events is not None:
                            errors.append(f"{prefix}: cannot compare independent events without onset metadata")
                    signal = eeg[index, :, :800]
                    record.update(signal_statistics(signal, sfreq))
                    if record["nonfinite_samples"]:
                        errors.append(f"{prefix}: nonfinite EEG")
                    fingerprints[hashlib.sha256(signal.tobytes()).hexdigest()].append(prefix)
                    records.append(record)
        except (OSError, ValueError, KeyError, IndexError) as error:
            errors.append(f"{session}: {type(error).__name__}: {error}")
    available_channels = [item["channel_names"] for item in sessions.values() if item["channel_names"] is not None]
    if available_channels and any(names != available_channels[0] for names in available_channels[1:]):
        errors.append("Channel names/order differ across sessions")
    duplicates = [group for group in fingerprints.values() if len(group) > 1]
    if duplicates:
        warnings.append(f"{len(duplicates)} groups of byte-identical EEG trials")
    summaries = {}
    for session in sessions:
        selected = [row for row in records if row["session"] == session and row.get("nonfinite_samples") == 0]
        if not selected:
            continue
        summary = {"trials": len(selected), "median_trial_rms_native_units": float(np.median([r["rms_native_units"] for r in selected])),
                   "mean_channel_std": np.mean([r["channel_std"] for r in selected], axis=0).tolist(),
                   "mean_channel_mean": np.mean([r["channel_mean"] for r in selected], axis=0).tolist(),
                   "flat_trial_count": sum(r["flat_channel_count"] > 0 for r in selected),
                   "band_relative_power": {band: float(np.mean([r["band_relative_power"][band] for r in selected]))
                                           for band in selected[0]["band_relative_power"]}}
        summaries[session] = summary
        offsets = Counter(r["onset_minus_annotation_samples"] for r in selected if "onset_minus_annotation_samples" in r)
        summary["onset_minus_annotation_sample_counts"] = dict(offsets)
        if any(offset != 0 for offset in offsets):
            warnings.append(f"{session}: onset/annotation coordinate offset {dict(offsets)}; verify first_samp/crop origin before interpreting as timing error")
        ordered = sorted([r for r in selected if r.get("playback_order_index") and r.get("onset_sample") is not None],
                         key=lambda r: int(r["playback_order_index"]))
        if any(b["onset_sample"] <= a["onset_sample"] for a, b in zip(ordered, ordered[1:])):
            warnings.append(f"{session}: onset is not strictly increasing with playback order")
    return {"schema_version": 1, "manifest_trial_count": len(rows), "audited_trials": len(records),
            "internal_mapping_status": "FAIL" if errors else "PASS",
            "independent_stimulus_alignment": ("NOT_VERIFIED" if events is None else
                                               "MATCHED_PROVIDED_EVENTS" if event_compared == len(rows) and not errors else "INCOMPLETE_OR_MISMATCH"),
            "independent_events_compared": event_compared, "errors": errors, "warnings": warnings,
            "limitations": ["NPZ and metadata may share the same upstream labeling error",
                            "Event matching does not establish monitor onset latency or preprocessing correctness",
                            "Units/reference/filtering remain unknown unless independently documented"],
            "sessions": sessions, "signal_summary": summaries, "identical_signal_groups": duplicates,
            "trial_statistics": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=ROOT / "data/manifests/chentianlin/eeg_trials.csv")
    parser.add_argument("--events", type=Path, help="Independent stimulus log CSV: session,onset_sample,video_id; onset in 200-Hz EEG coordinates")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eeg_semantic/session_audit/audit.json")
    args = parser.parse_args()
    rows = [row for row in load_trial_rows(args.trials) if row["video_id"].split("-")[0] in {"01", "02", "03", "04", "05", "06"}]
    result = audit(rows, read_csv(args.events) if args.events else None)
    result["manifest"] = str(args.trials.resolve())
    result["events_source"] = str(args.events.resolve()) if args.events else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    console = {k: result[k] for k in ("manifest_trial_count", "audited_trials", "internal_mapping_status", "independent_stimulus_alignment", "warnings")}
    console["error_count"] = len(result["errors"])
    console["errors_first20"] = result["errors"][:20]
    console["signal_summary"] = {s: {k: v for k, v in values.items() if not k.startswith("mean_channel")}
                                 for s, values in result["signal_summary"].items()}
    print(json.dumps(console, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
