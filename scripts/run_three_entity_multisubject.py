"""Independent per-subject pair-to-triple training; subject-level mean/sample SD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_eeg_composition as base
from scripts.build_three_entity_paper_table import paper_metrics, aggregate, KEYS
from scripts.evaluate_eeg_triple_top3 import validate_prediction

DEFAULT_SUBJECTS = ["chentianlin", "duzhuoxuan", "fangzikai", "luotianming", "niezhiheng"]
VARIANTS = ("original", "object_only")


def group_statistics(reports, subjects):
    if len(subjects) < 2 or len(set(subjects)) != len(subjects):
        raise ValueError("At least two distinct, pre-specified subjects required")
    if set(reports) != set(subjects):
        raise ValueError("Incomplete subject reports; never silently drop subjects")
    summaries = {s:aggregate(reports[s]) for s in subjects}
    output = []
    for setting in ("Session-Average", "Cross-Session"):
        for variant in VARIANTS:
            values = [next(r["metrics"] for r in summaries[s] if r["setting"] == setting and r["variant"] == variant) for s in subjects]
            output.append({"setting":setting,"variant":variant,"n_subjects":len(subjects),
                           "mean":{k:float(np.mean([v[k] for v in values])) for k in KEYS},
                           "std":{k:float(np.std([v[k] for v in values],ddof=1)) for k in KEYS}})
    return summaries, output


def table(rows, n):
    lines = [r"\begin{table}[t]",r"\centering",r"\small",
             r"\caption{Recognition of unseen three-entity scenes from full six-second EEG. Results are mean $\pm$ sample standard deviation across "+str(n)+" participants. Cross-session directions are averaged within each participant before aggregation.}",
             r"\label{tab:three_entity_recognition}",r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{@{}lccccccc@{}}",r"\toprule",
             r"Objective & Macro AP & Micro AP & Macro AUC & Micro AUC & Top-3 F1 & Set Acc. & Jaccard \\",r"\midrule",
             r"Chance Level & 0.5000 & 0.5000 & 0.5000 & 0.5000 & 0.5000 & 0.0500 & 0.3650 \\",r"\midrule"]
    for setting in ("Session-Average", "Cross-Session"):
        lines += [r"\multicolumn{8}{c}{\textbf{"+setting+r" Evaluation}} \\",r"\midrule"]
        for row in (r for r in rows if r["setting"] == setting):
            name = "Original" if row["variant"] == "original" else "Object-only"
            lines.append(name+" & "+" & ".join(f"{row['mean'][k]:.4f} $\\pm$ {row['std'][k]:.4f}" for k in KEYS)+r" \\")
        if setting == "Session-Average": lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)+"\n"


def worker(args):
    subject = args.subjects[0]
    prepared = args.output_root / "prepared" / f"{subject}.pt"
    p = base.load(prepared)
    if p["protocol"]["subject"] != subject or p["protocol"]["split_seed"] != 42:
        raise ValueError("Subject/split mismatch")
    if p["splits"] != base.split_videos(p["ids"],42) or not torch.equal(p["labels"],base.labels_for(p["ids"])):
        raise ValueError("Invalid prepared split/labels")
    output = args.output_root / "runs" / subject / args.protocol / args.variant / f"seed{args.seed}"
    fingerprint = base.digest(prepared)
    base.train(args,p,fingerprint,output)
    ckpt = base.load(output / "best.pt")
    if ckpt["signature"]["prepared_sha256"] != fingerprint:
        raise ValueError("Checkpoint input mismatch")
    complete = json.loads((output / "completed.json").read_text())
    if ckpt["signature"] != complete["signature"]:
        raise ValueError("Checkpoint/completion mismatch")
    model = base.CompactEEGClassifier(num_pairs=6,dropout=.35).to(args.device)
    model.load_state_dict(ckpt["model"])
    sessions = [0,1,2] if args.protocol == "session_average" else [int(args.protocol[-1])-1]
    ix = p["splits"]["test"]
    logits = base.infer(model,p["eeg"],ix,sessions,1200,ckpt["mean"],ckpt["std"],args.device,args.batch_size).mean(1)
    data = {"video_ids":[p["ids"][i] for i in ix],"logits":logits,"labels":p["labels"][ix],"object_names":base.OBJECTS}
    ids, logits, labels = validate_prediction(data)
    report = {"subject":subject,"variant":args.variant,"protocol":args.protocol,"seed":args.seed,
              "signature":ckpt["signature"],"checkpoint_sha256":base.digest(output / "best.pt"),
              "epoch":ckpt["epoch"],"n":len(ids),"metrics":paper_metrics(logits,labels),
              "input":"full 6s", "threshold_used":False}
    base.atomic_save(data,output / "test_top3_predictions.pt")
    base.atomic_json(report,output / "test_top3_metrics.json")
    print(f"[multi-triple] {subject}/{args.protocol}/{args.variant}: {report['metrics']}",flush=True)


def summarize(args, plan):
    reports = {}
    for subject in args.subjects:
        reports[subject] = []
        for protocol in base.PROTOCOLS:
            for variant in VARIANTS:
                path = args.output_root / "runs" / subject / protocol / variant / f"seed{args.seed}/test_top3_metrics.json"
                r = json.loads(path.read_text())
                if (r["subject"],r["protocol"],r["variant"],r["seed"],r["n"]) != (subject,protocol,variant,args.seed,156):
                    raise ValueError(f"Unexpected report identity: {path}")
                sig = r["signature"]
                if (sig["epochs"],sig["batch_size"],sig["lr"]) != (args.epochs,args.batch_size,args.lr):
                    raise ValueError("Training budget mismatch")
                reports[subject].append(r)
    per_subject, rows = group_statistics(reports,args.subjects)
    output = args.output_root / "report"
    output.mkdir(parents=True,exist_ok=True)
    base.atomic_json({"plan":plan,"per_subject":per_subject,"mean_std":rows,"std_ddof":1,
                      "unit":"subject, not session/video/seed", "exploratory":True,
                      "independent_stimulus_alignment":"NOT_VERIFIED"},output / "metrics.json")
    (output / "table.tex").write_text(table(rows,len(args.subjects)),encoding="utf-8")
    (output / "notes.txt").write_text(
        "Macro AP/AUC use dog, ball, flower, bird; micro/set metrics include all six entities.\n"
        "Session-average means logit fusion. One seed per subject; SD is across subjects (ddof=1).\n"
        "AP chance uses prevalence, not exact finite-sample random AP expectation.\n"
        "Known cardinality three. Full 6s only. Subject-specific training, NOT cross-subject transfer.\n"
        "Current participant already inspected; exploratory extension. No automatic significance claims.\n",encoding="utf-8")
    print(json.dumps(rows,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("run","worker","summarize"),default="run")
    parser.add_argument("--subjects",nargs="+",default=DEFAULT_SUBJECTS)
    parser.add_argument("--data-root",type=Path,default=Path("data/EEG_and_EYE"))
    parser.add_argument("--output-root",type=Path,default=Path("outputs/eeg_composition_multisubject"))
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--protocol",choices=base.PROTOCOLS,default="cs_s3")
    parser.add_argument("--variant",choices=VARIANTS,default="original")
    args = parser.parse_args()
    if min(args.epochs,args.batch_size,args.threads) < 1 or args.lr <= 0:
        parser.error("Positive budgets required")
    if len(set(args.subjects)) != len(args.subjects) or any(Path(s).name != s or s in (".","..") for s in args.subjects):
        parser.error("Distinct subject directory names required")
    torch.set_num_threads(args.threads)
    if args.stage == "worker":
        if len(args.subjects) != 1 or args.dry_run: parser.error("Worker requires one subject and no dry-run")
        worker(args)
        return
    if len(args.subjects) < 2: parser.error("At least two subjects required")
    plan = {"schema_version":1,"subjects":args.subjects,"seed":args.seed,"split_seed":42,
            "protocols":list(base.PROTOCOLS),"variants":list(VARIANTS),"epochs":args.epochs,
            "batch_size":args.batch_size,"lr":args.lr,"input":"full 6s; training 4s",
            "data_root":str(args.data_root.resolve()),"selection":"pair validation macro AP"}
    path = args.output_root / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen cohort/budget changed; use a different output-root")
    if args.stage == "summarize":
        if args.dry_run: parser.error("Use --stage run --dry-run")
        if not path.is_file(): raise FileNotFoundError(path)
        summarize(args,plan)
        return
    commands = []
    for subject in args.subjects:
        for session in range(1,4):
            raw = args.data_root / subject / f"session{session}/EEG/eeg_data.npz"
            if not raw.is_file(): raise FileNotFoundError(f"Missing source; no subject silently skipped: {raw}")
        prepared = args.output_root / "prepared" / f"{subject}.pt"
        if not prepared.exists():
            commands.append([sys.executable,str(ROOT / "scripts/run_eeg_composition.py"),"--stage","prepare",
                             "--subject",subject,"--data-root",str(args.data_root),"--prepared",str(prepared)])
        for protocol in base.PROTOCOLS:
            for variant in VARIANTS:
                command = [sys.executable,str(Path(__file__).resolve()),"--stage","worker","--subjects",subject,
                           "--protocol",protocol,"--variant",variant]
                for key in ("output_root","seed","epochs","batch_size","lr","threads","device"):
                    command += ["--"+key.replace("_","-"),str(getattr(args,key))]
                if args.resume: command.append("--resume")
                commands.append(command)
    print(f"[multi-triple] subjects={len(args.subjects)} training_jobs={len(args.subjects)*8}",flush=True)
    for command in commands: print("[multi-triple] "+" ".join(command),flush=True)
    if args.dry_run:
        print("Dry-run: source path checks only; no files changed, full data validation occurs during prepare")
        return
    args.output_root.mkdir(parents=True,exist_ok=True)
    if not path.exists(): base.atomic_json(plan,path)
    for command in commands: subprocess.run(command,check=True)
    summarize(args,plan)
    print("[multi-triple] COMPLETE",flush=True)


if __name__ == "__main__": main()
