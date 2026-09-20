"""Re-score stored six-second EEG predictions with fixed cardinality three only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ms_video_eval.eeg_composition import OBJECTS, PROTOCOLS, labels_for


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_prediction(data):
    expected = [f"{c:02d}-{v:03d}" for c in (7, 8) for v in range(1,79)]
    ids = data["video_ids"]
    if len(ids) != 156 or len(set(ids)) != 156 or set(ids) != set(expected):
        raise ValueError("Expected all 156 unique 07/08 video IDs")
    if tuple(data["object_names"]) != tuple(OBJECTS):
        raise ValueError("Object order mismatch")
    logits, labels = data["logits"].float(), data["labels"].float()
    if logits.shape != (156,6) or labels.shape != (156,6) or not torch.isfinite(logits).all():
        raise ValueError("Invalid logits/labels shape or nonfinite logits")
    if not torch.equal(labels, labels_for(ids)):
        raise ValueError("Labels disagree with video category")
    order = [ids.index(video) for video in expected]
    return expected, logits[order], labels[order]


def score(ids, logits, labels, repeats=5000, seed=42):
    if repeats < 1:
        raise ValueError("Positive permutation budget required")
    # Match the existing top-k implementation, including its treatment of ties.
    top = logits.sigmoid().topk(3, dim=1).indices
    pred = torch.zeros_like(labels, dtype=torch.bool)
    pred.scatter_(1, top, True)
    y = labels.bool()
    exact = (pred == y).all(1)
    tp = (pred & y).sum(1)
    per_object = {}
    for j, obj in enumerate(OBJECTS):
        pos = y[:,j]
        per_object[obj] = {"positive_count": int(pos.sum()), "negative_count": int((~pos).sum()),
                           "recall": float(pred[pos,j].float().mean()) if pos.any() else None,
                           "false_positive_rate": float(pred[~pos,j].float().mean()) if (~pos).any() else None}
    encoded_pred = (pred.long() * (2**torch.arange(6))).sum(1).numpy()
    encoded_y = (y.long() * (2**torch.arange(6))).sum(1).numpy()
    rng = np.random.default_rng(seed)
    null = np.array([(encoded_pred[rng.permutation(len(ids))] == encoded_y).mean() for _ in range(repeats)])
    accuracy = float(exact.double().mean())
    rows = [{"video_id": video, "truth": [OBJECTS[k] for k in range(6) if y[i,k]],
             "predicted": [OBJECTS[k] for k in range(6) if pred[i,k]],
             "correct": bool(exact[i]), "correct_subject_count": int(tp[i])} for i, video in enumerate(ids)]
    by_category = {}
    for category in ("07", "08"):
        ix = [i for i,v in enumerate(ids) if v.startswith(category+"-")]
        combinations = Counter("+".join(rows[i]["predicted"]) for i in ix)
        by_category[category] = {"n": len(ix), "correct": int(exact[ix].sum()),
                                 "exact_accuracy": float(exact[ix].double().mean()),
                                 "subject_recall": float(tp[ix].double().mean()/3),
                                 "predicted_combinations": dict(combinations)}
    return {"n": len(ids), "correct": int(exact.sum()), "exact_accuracy": accuracy,
            "subject_recall": float(tp.double().mean()/3),
            "mean_correct_subjects": float(tp.double().mean()), "per_object": per_object,
            "by_category": by_category, "video_predictions": rows,
            "shuffle": {"repeats": repeats, "seed": seed, "mean_exact": float(null.mean()),
                        "q025": float(np.quantile(null,.025)), "q975": float(np.quantile(null,.975)),
                        "matched_minus_shuffled": accuracy-float(null.mean()),
                        "note": "Global video permutation of complete predictions. Null quantiles are not accuracy confidence intervals; no significance claim."}}


def sources(root, subject, seed):
    for stage, variant in (("audit-baseline", "original"), ("object-only", "object_only")):
        for protocol in PROTOCOLS:
            run = root / stage / subject / protocol / variant / f"seed{seed}"
            group = "session_average" if protocol == "session_average" else "session"+protocol[-1]
            for mode in ("6s", "6s_windows"):
                yield variant, protocol, mode, run / "audit.json", run / "predictions" / f"{mode}_{group}.pt"


def evaluate(root, subject, seed, repeats):
    results, fingerprints = [], set()
    for variant, protocol, mode, audit_path, pred_path in sources(root, subject, seed):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        sig = audit["signature"]
        if audit["development_only"] or (sig["protocol"], sig["variant"], sig["seed"]) != (protocol,variant,seed):
            raise ValueError(f"Protocol signature mismatch: {audit_path}")
        if sig["selection"] != "01-06 validation allowed-session mean object macro AP":
            raise ValueError("Unexpected checkpoint selection protocol")
        fingerprints.add(sig["prepared_sha256"])
        ids, logits, labels = validate_prediction(torch.load(pred_path, map_location="cpu", weights_only=True))
        metrics = score(ids, logits, labels, repeats, seed)
        results.append({"variant":variant,"protocol":protocol,"input":mode,
                        "checkpoint_epoch":audit["checkpoint_epoch"], "checkpoint_sha256":audit["checkpoint_sha256"],
                        "prediction_file":str(pred_path), "prediction_sha256":sha256(pred_path),
                        "audit_sha256":sha256(audit_path), "metrics":metrics})
        print(f"[triple-top3] {variant}/{protocol}/{mode} {metrics['correct']}/156 exact={metrics['exact_accuracy']:.4f}", flush=True)
    if len(fingerprints) != 1:
        raise ValueError("Input runs do not share the same prepared data")
    return {"schema_version":1,"subject":subject,"seed":seed,"prepared_sha256":fingerprints.pop(),
            "protocol":{"decoder":"fixed top3 of six objects, unconstrained combinations",
                        "primary_input":"6s", "secondary_input":"6s_windows: 0-4,1-5,2-6 seconds; mean logits",
                        "training":"01-06 pairs only; 420 train/48 validation; 4s EEG",
                        "test":"07/08 unseen triples; 156 videos; all six labels including person",
                        "known_cardinality":3,"threshold_used":False,"retraining":False,
                        "reused_predictions":True,"exploratory":True,"independent_stimulus_alignment":"NOT_VERIFIED",
                        "random_uniform_exact":.05,"fixed_07_or_08_exact":.5,
                        "limitations":"Previously inspected test set, one subject/seed. Sessions, windows and protocols are not independent samples. Fixed-combination baseline uses test-distribution prior."},
            "results":results}


def markdown(report):
    lines = ["# 三主体6秒EEG组合泛化：固定Top-3评测", "",
             "已知主体数为3；六选三，不限制只能输出07/08。无阈值检测。",
             "复用已保存预测重新计分，不是重新训练或新增独立测试。完整6秒为主表，滑窗为补充。",
             "单被试、单seed，测试集已查看；结果为探索性。不同协议不可当成独立重复。", "",
             "| 方法 | 协议 | 输入 | 正确/156 | 集合正确率 | 主体召回率 | 07正确率 | 08正确率 | 高于打乱均值 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        m = row["metrics"]
        lines.append(f"| {row['variant']} | {row['protocol']} | {row['input']} | {m['correct']} | "
                     f"{m['exact_accuracy']:.2%} | {m['subject_recall']:.2%} | "
                     f"{m['by_category']['07']['exact_accuracy']:.2%} | {m['by_category']['08']['exact_accuracy']:.2%} | "
                     f"{m['shuffle']['matched_minus_shuffled']*100:+.2f}个百分点 |")
    lines += ["", "均匀随机六选三：5%；固定输出07或08：50%（利用测试组合分布的参考基线，并非同等开放组合任务）。",
              "打乱差值衡量正确EEG配对相对保留预测偏好的打乱对照的增益，不是自动显著性结论。",
              "详细JSON保留六主体召回/误报、所有预测组合、逐视频结果和打乱零假设区间。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eeg_composition_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eeg_composition_top3_6s/seed42"))
    parser.add_argument("--subject", default="chentianlin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-repeats", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shuffle_repeats < 1:
        parser.error("Positive shuffle budget required")
    missing = []
    for variant, protocol, mode, audit_path, pred_path in sources(args.root,args.subject,args.seed):
        print(f"{variant}/{protocol}/{mode}: {pred_path}")
        missing.extend(str(p) for p in (audit_path,pred_path) if not p.is_file())
    if missing:
        raise FileNotFoundError("Missing inputs:\n"+"\n".join(sorted(set(missing))))
    if args.output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new --output-dir to preserve reports")
    if args.dry_run:
        print("Preflight PASS: 16 prediction files; no files changed")
        return
    torch.set_num_threads(1)
    report = evaluate(args.root,args.subject,args.seed,args.shuffle_repeats)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output_dir / "report.zh-CN.md").write_text(markdown(report),encoding="utf-8")
    print(f"[triple-top3] COMPLETE: {args.output_dir}")


if __name__ == "__main__":
    main()
