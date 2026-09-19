"""EEG2Caption: train 01--06, test 07--08 under cross-session / logit-average evaluation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))
from EEG2Caption.src.common import CompactEEGClassifier
from ms_video_eval.eeg_composition import (OBJECTS, PROTOCOLS, labels_for, split_videos,
    fit_normalization, select_threshold, metrics, set_metrics)
from scripts.build_eeg_video_manifest import load_session
from scripts.run_c2_residual import digest
from scripts.train_compact_tora_alignment import atomic_json, atomic_save


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare(args):
    if args.prepared.exists():
        raise FileExistsError("Prepared cache exists; reuse it or choose a new --prepared path")
    ids = [f"{c:02d}-{v:03d}" for c in range(1,9) for v in range(1,79)]
    splits = split_videos(ids, args.split_seed)
    eeg = torch.zeros(624,3,62,1200)
    channels, hashes = None, {}
    for s in range(3):
        directory = args.data_root / args.subject / f"session{s+1}"
        _, rows = load_session(directory)
        if set(rows) != set(ids):
            raise ValueError(f"Incomplete/unexpected video IDs: {directory}")
        path = directory / "EEG/eeg_data.npz"
        metadata = Path(rows[ids[0]]["metadata_path"])
        hashes[f"session{s+1}"] = {"npz": digest(path), "metadata": digest(metadata)}
        with np.load(path, allow_pickle=False) as data:
            names = data["channel_names"].tolist()
            if len(names) != 62 or len(set(names)) != 62 or (channels is not None and names != channels):
                raise ValueError("Channel names/order mismatch")
            channels = names
            values, mask = data["eeg"], data["mask"]
            for i, video in enumerate(ids):
                row = rows[video]
                n = 800 if int(video[:2]) <= 6 else 1200
                if row["length_samples"] != n or row["sfreq"] != 200.:
                    raise ValueError(f"Unexpected duration/sfreq: {directory}/{video}")
                j = row["trial_index"]
                if not mask[j,:n].all():
                    raise ValueError(f"Padding in valid interval: {directory}/{video}")
                signal = torch.from_numpy(np.asarray(values[j,:,:n],dtype=np.float32).copy())*1e6
                if signal.shape != (62,n) or not torch.isfinite(signal).all():
                    raise ValueError(f"Invalid EEG: {directory}/{video}")
                eeg[i,s,:,:n] = signal
        print(f"[composition] prepared session{s+1}", flush=True)
    protocol = {"subject": args.subject, "split_seed": args.split_seed, "source_hashes": hashes,
                "channel_names": channels, "object_names": OBJECTS, "scale": 1e6,
                "splits": {k:[ids[i] for i in indices] for k,indices in splits.items()},
                "train_samples": 800, "test_samples": 1200,
                "independent_stimulus_alignment": "NOT_VERIFIED"}
    args.prepared.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({"eeg": eeg, "ids": ids, "labels": labels_for(ids), "splits": splits, "protocol": protocol}, args.prepared)
    atomic_json(protocol, args.prepared.with_suffix(".json"))


@torch.no_grad()
def infer(model, eeg, ids, sessions, samples, mean, std, device, batch_size):
    model.eval()
    outputs = []
    for start in range(0,len(ids),batch_size):
        x = eeg[ids[start:start+batch_size]][:,sessions,:,:samples]
        x = (x-mean[None,None,:,None])/std[None,None,:,None]
        outputs.append(model(x.to(device))["session_object_logits"].cpu())
    return torch.cat(outputs)


def train(args, p, fingerprint, output):
    sessions = PROTOCOLS[args.protocol]
    signature = {"schema_version": 1, "prepared_sha256": fingerprint, "protocol": args.protocol, "variant": args.variant,
                 "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                 "normalization": "shared_channel_train_allowed_sessions_first800", "dropout": .35,
                 "weight_decay": 1e-4, "pair_weight": .5 if args.variant == "original" else 0.,
                 "fused_weight": 1., "consistency_weight": .05, "noise_std": .02, "time_mask_samples": 20,
                 "selection": "01-06 validation allowed-session mean object macro AP", "precision": "float32"}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        if json.loads((output / "completed.json").read_text())["signature"] != signature:
            raise ValueError("Completed run configuration changed")
        print(f"[composition] skip completed {output}", flush=True)
        return
    torch.manual_seed(args.seed)
    model = CompactEEGClassifier(num_pairs=6, dropout=.35).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.lr*.05)
    generator = torch.Generator().manual_seed(args.seed)
    ti, vi = p["splits"]["train"], p["splits"]["validation"]
    mean, std = fit_normalization(p["eeg"], ti, sessions)
    x = (p["eeg"][ti][:,sessions,:,:800]-mean[None,None,:,None])/std[None,None,:,None]
    y = p["labels"][ti]
    pairs = torch.tensor([int(p["ids"][i][:2])-1 for i in ti])
    weight = ((len(y)-y.sum(0))/y.sum(0).clamp_min(1)).to(args.device)
    start, best, history, updates = 0, -1., [], 0
    if (output / "last.pt").exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a new output-root")
        last = load(output / "last.pt")
        if last["signature"] != signature:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        generator.set_state(last["sampler_rng"])
        torch.set_rng_state(last["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        start, best, history, updates = last["epoch"], last["best"], last["history"], last["updates"]
    for epoch in range(start+1,args.epochs+1):
        model.train()
        total = 0.
        for batch in torch.randperm(len(ti),generator=generator).split(args.batch_size):
            signal, target, category = x[batch].to(args.device).clone(), y[batch].to(args.device), pairs[batch].to(args.device)
            signal += torch.randn_like(signal)*.02
            mask_start = int(torch.randint(781,(),generator=generator))
            signal[:,:,:,mask_start:mask_start+20] = 0
            out = model(signal)
            loss = F.binary_cross_entropy_with_logits(out["session_object_logits"],target[:,None,:].expand(-1,len(sessions),-1),pos_weight=weight)
            loss += F.binary_cross_entropy_with_logits(out["fused_object_logits"],target,pos_weight=weight)
            if args.variant == "original":
                loss += .5*(F.cross_entropy(out["session_pair_logits"].flatten(0,1),category[:,None].expand(-1,len(sessions)).flatten())
                            + F.cross_entropy(out["fused_pair_logits"],category))
            feature = F.normalize(out["features"],dim=-1)
            loss += .05*(feature-feature.mean(1,keepdim=True)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            updates += 1
            total += float(loss.detach())*len(batch)
        scheduler.step()
        logits = infer(model,p["eeg"],vi,sessions,800,mean,std,args.device,args.batch_size).mean(1)
        score = metrics(logits,p["labels"][vi],.5,cardinality=2)["informative_macro_ap"]
        improved = score > best
        if improved:
            best = score
        history.append({"epoch":epoch,"optimizer_updates":updates,"train_loss":total/len(ti),"validation_macro_ap":score})
        state = {"signature":signature,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
                 "epoch":epoch,"updates":updates,"best":best,"history":history,"mean":mean,"std":std,
                 "sampler_rng":generator.get_state(),"torch_rng":torch.get_rng_state(),
                 "cuda_rng":torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []}
        if improved:
            atomic_save(state,output / "best.pt")
        atomic_save(state,output / "last.pt")
        atomic_json(history,output / "history.json")
        print(f"[composition] {args.protocol}/{args.variant} epoch={epoch}/{args.epochs} updates={updates} loss={total/len(ti):.4f} val_AP={score:.4f}",flush=True)
    atomic_json({"signature":signature,"optimizer_updates":updates,"status":"COMPLETE"},output / "completed.json")


def evaluate(args,p,fingerprint,output):
    ckpt = load(output / "best.pt")
    if (ckpt["signature"]["prepared_sha256"] != fingerprint or ckpt["signature"]["protocol"] != args.protocol
            or ckpt["signature"]["variant"] != args.variant or ckpt["signature"]["seed"] != args.seed):
        raise ValueError("Checkpoint does not match evaluation protocol/data")
    if json.loads((output / "completed.json").read_text())["signature"] != ckpt["signature"]:
        raise ValueError("Incomplete/mismatched training")
    model = CompactEEGClassifier(num_pairs=6,dropout=.35).to(args.device)
    model.load_state_dict(ckpt["model"])
    allowed = PROTOCOLS[args.protocol]
    vi, ti = p["splits"]["validation"], p["splits"]["test"]
    mean,std = ckpt["mean"],ckpt["std"]
    validation = infer(model,p["eeg"],vi,allowed,800,mean,std,args.device,args.batch_size).mean(1)
    threshold = select_threshold(validation,p["labels"][vi])
    sessions = [0,1,2] if args.protocol == "session_average" else [s for s in range(3) if s not in allowed]
    report = {"signature":ckpt["signature"],"checkpoint_epoch":ckpt["epoch"],"threshold":threshold,
              "threshold_fit":"01-06 validation only, global micro-F1 grid .10:.05:.90",
              "test_ids":[p["ids"][i] for i in ti], "reports":{}, "diagnostic_only":False,
              "independent_stimulus_alignment":"NOT_VERIFIED",
              "chance_uniform_top3_exact":.05,"chance_uniform_top3_recall":.5}
    for samples in (800,1200):
        logits = infer(model,p["eeg"],ti,sessions,samples,mean,std,args.device,args.batch_size)
        groups = {f"session{s+1}":logits[:,j] for j,s in enumerate(sessions)}
        if len(sessions)==3:
            groups["session_average"] = logits.mean(1)
        for group,prediction in groups.items():
            key = f"{samples//200}s/{group}"
            result = metrics(prediction,p["labels"][ti],threshold)
            result["by_category"] = {}
            for category in ("07","08"):
                subset = [j for j,i in enumerate(ti) if p["ids"][i].startswith(category+"-")]
                result["by_category"][category] = metrics(prediction[subset],p["labels"][ti][subset],threshold)
            report["reports"][key] = result
            probabilities = prediction.sigmoid()
            records = [{"video_id":p["ids"][i],"truth":[OBJECTS[k] for k in range(6) if p["labels"][i,k]],
                        "probabilities":probabilities[j].tolist(),"top3":[OBJECTS[k] for k in prediction[j].topk(3).indices.tolist()],
                        "threshold_prediction":[OBJECTS[k] for k in range(6) if probabilities[j,k]>=threshold]} for j,i in enumerate(ti)]
            atomic_json(records,output / "predictions" / f"{samples//200}s_{group}.json")
            atomic_save({"video_ids":report["test_ids"],"logits":prediction,"labels":p["labels"][ti]},output / "predictions" / f"{samples//200}s_{group}.pt")
            print(f"[composition] {key} exact={result['topk_metrics']['exact_set_accuracy']:.4f} informative_AP={result['informative_macro_ap']:.4f}",flush=True)
    report["fixed_set_baselines"] = {c:set_metrics(labels_for([c+"-001"]).expand(len(ti),-1),p["labels"][ti]) for c in ("07","08")}
    atomic_json(report,output / "report.json")


def summarize(args):
    rows=[]
    for protocol in args.protocols:
        for variant in args.variants:
            for seed in args.seeds:
                file=args.output_root / args.subject / protocol / variant / f"seed{seed}" / "report.json"
                report=json.loads(file.read_text())
                for group,r in report["reports"].items():
                    rows.append({"protocol":protocol,"variant":variant,"seed":seed,"group":group,"n":r["video_count"],
                                 "epoch":report["checkpoint_epoch"],"threshold":report["threshold"],
                                 "top3_exact":r["topk_metrics"]["exact_set_accuracy"],"top3_recall":r["topk_metrics"]["micro_recall"],
                                 "threshold_exact":r["threshold_metrics"]["exact_set_accuracy"],"threshold_micro_f1":r["threshold_metrics"]["micro_f1"],
                                 "informative_macro_ap":r["informative_macro_ap"],"person_recall":r["per_object"]["person"]["recall"],
                                 "car_false_positive_rate":r["per_object"]["car"]["false_positive_rate"]})
    path=args.output_root / args.subject / "summary.csv"
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,indent=2),flush=True)
    print(f"[composition] summary={path}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("prepare","train","evaluate","wave","summarize"))
    parser.add_argument("--data-root",type=Path,default=Path("data/EEG_and_EYE"))
    parser.add_argument("--subject",default="chentianlin")
    parser.add_argument("--prepared",type=Path)
    parser.add_argument("--output-root",type=Path,default=Path("outputs/eeg_composition"))
    parser.add_argument("--protocol",choices=PROTOCOLS,default="cs_s3")
    parser.add_argument("--protocols",choices=PROTOCOLS,nargs="+",default=list(PROTOCOLS))
    parser.add_argument("--variant",choices=("original","object_only"),default="original")
    parser.add_argument("--variants",choices=("original","object_only"),nargs="+",default=["original"])
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42])
    parser.add_argument("--split-seed",type=int,default=42)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    args.prepared=args.prepared or args.output_root / args.subject / "prepared.pt"
    if min(args.epochs,args.batch_size,args.threads)<1 or args.lr<=0:
        parser.error("Positive budgets required")
    if args.stage=="wave":
        for protocol in args.protocols:
            for variant in args.variants:
                for seed in args.seeds:
                    for stage in ("train","evaluate"):
                        command=[sys.executable,str(Path(__file__).resolve()),"--stage",stage,"--protocol",protocol,"--variant",variant,"--seed",str(seed)]
                        for field in ("prepared","output_root","subject","epochs","batch_size","lr","threads","device"):
                            command += ["--"+field.replace("_","-"),str(getattr(args,field))]
                        if args.resume: command.append("--resume")
                        print("[composition] "+" ".join(command),flush=True)
                        if not args.dry_run: subprocess.run(command,check=True)
        if not args.dry_run: summarize(args)
        return
    if args.dry_run: parser.error("--dry-run requires --stage wave")
    if args.stage=="summarize": summarize(args); return
    torch.set_num_threads(args.threads)
    if args.stage=="prepare": prepare(args); return
    p=load(args.prepared)
    if p["protocol"]["subject"] != args.subject:
        raise ValueError("Prepared subject mismatch")
    expected=split_videos(p["ids"],p["protocol"]["split_seed"])
    if expected!=p["splits"] or not torch.equal(labels_for(p["ids"]),p["labels"]):
        raise ValueError("Prepared labels/splits inconsistent")
    output=args.output_root / args.subject / args.protocol / args.variant / f"seed{args.seed}"
    {"train":train,"evaluate":evaluate}[args.stage](args,p,digest(args.prepared),output)


if __name__=="__main__":
    main()
