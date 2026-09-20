"""Compute paper-aligned metrics from saved predictions; no training or thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_eeg_triple_top3 import sources, validate_prediction, sha256, OBJECTS

KEYS = ("macro_ap", "micro_ap", "macro_auc", "micro_auc", "top3_f1", "set_accuracy", "jaccard")
LABELS = ("Macro AP", "Micro AP", "Macro AUC", "Micro AUC", "Top-3 F1", "Set Acc.", "Jaccard")


def paper_metrics(logits, labels):
    probs = logits.sigmoid()
    y, scores = labels.numpy().astype(int), probs.numpy()
    variable = [j for j in range(6) if len(np.unique(y[:,j])) == 2]
    if not variable:
        raise ValueError("No variable labels for macro ranking metrics")
    pred = torch.zeros_like(labels, dtype=torch.bool)
    pred.scatter_(1, probs.topk(3, dim=1).indices, True)
    truth = labels.bool()
    intersection = (pred & truth).sum(1).double()
    union = (pred | truth).sum(1).double()
    if not torch.all(truth.sum(1) == 3):
        raise ValueError("Fixed Top-3 protocol requires three reference entities")
    return {
        "macro_ap": float(np.mean([average_precision_score(y[:,j],scores[:,j]) for j in variable])),
        "micro_ap": float(average_precision_score(y.ravel(),scores.ravel())),
        "macro_auc": float(np.mean([roc_auc_score(y[:,j],scores[:,j]) for j in variable])),
        "micro_auc": float(roc_auc_score(y.ravel(),scores.ravel())),
        "top3_f1": float((intersection/3).mean()),
        "set_accuracy": float((pred == truth).all(1).double().mean()),
        "jaccard": float((intersection/union).mean()),
        "macro_entities": [OBJECTS[j] for j in variable],
        "micro_and_set_entities": list(OBJECTS),
    }


def aggregate(rows):
    summary = []
    for setting, protocols in (("Session-Average", ("session_average",)),
                               ("Cross-Session", ("cs_s1", "cs_s2", "cs_s3"))):
        for variant in ("original", "object_only"):
            selected = [r for r in rows if r["variant"] == variant and r["protocol"] in protocols]
            if len(selected) != len(protocols) or {r["protocol"] for r in selected} != set(protocols):
                raise ValueError("Incomplete/duplicate protocol rows")
            summary.append({"setting":setting, "variant":variant,
                            "directions":len(selected),
                            "metrics":{k:float(np.mean([r["metrics"][k] for r in selected])) for k in KEYS}})
    return summary


def collect(root, subject, seed):
    rows, fingerprints = [], set()
    for variant, protocol, mode, audit_path, pred_path in sources(root, subject, seed):
        if mode != "6s":
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        sig = audit["signature"]
        if audit["development_only"] or (sig["protocol"],sig["variant"],sig["seed"]) != (protocol,variant,seed):
            raise ValueError(f"Unexpected audit protocol: {audit_path}")
        if sig["selection"] != "01-06 validation allowed-session mean object macro AP":
            raise ValueError("Unexpected checkpoint selection rule")
        fingerprints.add(sig["prepared_sha256"])
        ids, logits, labels = validate_prediction(torch.load(pred_path, map_location="cpu", weights_only=True))
        rows.append({"variant":variant,"protocol":protocol,"n":len(ids),
                     "checkpoint_epoch":audit["checkpoint_epoch"],
                     "checkpoint_sha256":audit["checkpoint_sha256"],
                     "prediction_sha256":sha256(pred_path),"prediction_path":str(pred_path),
                     "audit_sha256":sha256(audit_path),"metrics":paper_metrics(logits,labels)})
    if len(fingerprints) != 1:
        raise ValueError("Prepared dataset mismatch")
    return {"subject":subject,"seed":seed,"input":"full 6s; no window fusion",
            "prepared_sha256":fingerprints.pop(),"threshold_used":False,"exploratory":True,
            "independent_stimulus_alignment":"NOT_VERIFIED",
            "metric_scope":"Macro AP/AUC: dog, ball, flower, bird only. Micro AP/AUC and set metrics: all six entities.",
            "aggregation":"Compute each held-session metric first, then arithmetic mean over three directions. Not pooled predictions, not between-subject mean/std.",
            "rows":rows,"summary":aggregate(rows)}


def latex(report):
    lines = [r"\begin{table}[t]",r"\centering",r"\small",
             r"\caption{Recognition of unseen three-entity scenes from full six-second EEG (one participant, one seed). Cross-session results average three held-out directions.}",
             r"\label{tab:three_entity_recognition}",r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}lccccccc@{}}",r"\toprule",
             r"Objective & \shortstack{Macro\\AP$^{\dagger}$} & \shortstack{Micro\\AP} & \shortstack{Macro\\AUC$^{\dagger}$} & \shortstack{Micro\\AUC} & \shortstack{Top-3\\F1} & \shortstack{Set\\Acc.} & Jaccard \\",
             r"\midrule"]
    for setting in ("Session-Average", "Cross-Session"):
        lines += [r"\multicolumn{8}{c}{\textbf{"+setting+r" Evaluation}} \\",r"\midrule"]
        rows = [r for r in report["summary"] if r["setting"] == setting]
        best = {k:max(r["metrics"][k] for r in rows) for k in KEYS}
        for row in rows:
            cells = []
            for k in KEYS:
                value = row["metrics"][k]
                cell = f"{value:.4f}"
                if round(value,4) == round(best[k],4): cell = r"\textbf{"+cell+"}"
                cells.append(cell)
            name = "Original" if row["variant"] == "original" else "Object-only"
            lines.append(name+" & "+" & ".join(cells)+r" \\")
        if setting == "Session-Average": lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}",r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\footnotesize",
              r"$^{\dagger}$Macro AP/AUC exclude person (all positive) and car (all negative); all other metrics include all six entities. Session averaging operates on logits, not raw EEG. Higher is better for all metrics.",
              r"\end{minipage}",r"\end{table}"]
    return "\n".join(lines)+"\n"


def section(report):
    return (r"\section{Extension to Three-Entity Scenes}"+"\n"
            +r"\label{app:three_entity_scenes}"+"\n\n"
            +"We further examine entity recognition on unseen three-entity combinations. "
            "Using the EEG-Caption compact encoder, we train on four-second EEG from "
            "two-entity categories 01--06 (420 training and 48 validation videos) and "
            "test on complete six-second recordings from categories 07--08 (156 videos). "
            "The object-only variant removes the auxiliary combination-classification loss. "
            "Both variants select the three highest-scoring entities from all six labels. "
            "Session-average evaluation averages logits across three sessions; cross-session "
            "evaluation trains on two sessions and tests on the remaining session.\n\n"
            +latex(report)+"\n"
            +r"Table~\ref{tab:three_entity_recognition} summarizes ranking and fixed-cardinality "
            "recognition metrics. Cross-session scores are averaged over the three held-out "
            "directions rather than pooled across predictions. These single-participant, "
            "single-seed results are exploratory and do not establish statistically significant "
            "or robust compositional generalization.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("outputs/eeg_composition_v2"))
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/eeg_composition_paper/seed42"))
    parser.add_argument("--subject",default="chentianlin")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    for _,_,mode,audit,pred in sources(args.root,args.subject,args.seed):
        if mode == "6s":
            for path in (audit,pred):
                if not path.is_file(): raise FileNotFoundError(path)
    if args.output_dir.exists(): raise FileExistsError("Choose a new output directory; old results are preserved")
    if args.dry_run:
        print("Preflight PASS: eight full-6s predictions; no training or files changed")
        return
    torch.set_num_threads(1)
    report = collect(args.root,args.subject,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    (args.output_dir / "table.tex").write_text(latex(report),encoding="utf-8")
    (args.output_dir / "section.tex").write_text(section(report),encoding="utf-8")
    for row in report["summary"]:
        print(row["setting"],row["variant"]," ".join(f"{k}={row['metrics'][k]:.4f}" for k in KEYS))
    print(f"COMPLETE: {args.output_dir}")


if __name__ == "__main__":
    main()
