"""C2 mean / EEG-predicted category centers / centers plus within-class residual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(value))
from ms_video_eval.c2_residual import normalize_sessions, within_category_permutation, decode
from ms_video_eval.tora_conditioning import ToraPCAProjector
from scripts.run_c2_residual import digest, measure, atomic_json, atomic_save
from scripts.run_c2_sessions import ridge_fit, ridge_predict, features


def train_centers(target, categories):
    if set(categories.tolist()) != set(range(6)):
        raise ValueError("Training set must contain all six categories")
    return torch.stack([target[categories == category].mean(0) for category in range(6)])


def predict_parts(eeg, checkpoint):
    """Inference has no ground-truth categories/captions in its interface."""
    scores = ridge_predict(checkpoint["classifier"], eeg)
    probabilities = F.softmax(scores / checkpoint["temperature"], dim=-1)
    coarse = probabilities @ checkpoint["centers"]
    residual = ridge_predict(checkpoint["residual_fit"], eeg)
    return coarse, residual, probabilities


def fit(package, classifier, sessions, fingerprint, classifier_hash):
    ti, vi = package["splits"]["train"], package["splits"]["validation"]
    eeg, z, categories = package["eeg"], package["z"], package["categories"]
    centers = train_centers(z[ti], categories[ti])
    valid_scores = ridge_predict(classifier, eeg[vi][:, sessions])
    temperatures = [0.01, 0.03, 0.1, 0.3, 1., 3.]
    calibration = [{"temperature": t, "validation_nll": float(F.cross_entropy(valid_scores/t, categories[vi]))}
                   for t in temperatures]
    temperature = min(calibration, key=lambda r: r["validation_nll"])["temperature"]
    coarse = F.softmax(valid_scores / temperature, dim=-1) @ centers
    residual_target = z[ti] - centers[categories[ti]]
    x = features(eeg[ti][:, sessions])
    y = residual_target[:, None].expand(-1, len(sessions), -1).flatten(0, 1).double()
    best, chosen, grid = -float("inf"), None, []
    # Include zero first: exact ties favor the simpler center-only predictor.
    for alpha in (0.001, 0.01, 0.1, 1., 10., 100.):
        residual_fit = ridge_fit(x, y, alpha)
        residual = ridge_predict(residual_fit, eeg[vi][:, sessions])
        for weight in (0., 0.25, 0.5, 1.):
            metrics = measure(coarse + weight * residual, package, vi)
            grid.append({"alpha": alpha, "residual_weight": weight, "validation": metrics})
            score = metrics["within_category_mrr"]
            if score > best + 1e-6:
                best = score
                chosen = {"classifier": classifier, "centers": centers, "temperature": temperature,
                          "residual_fit": residual_fit, "residual_weight": weight,
                          "selection_metric": "validation within-category MRR",
                          "prepared_sha256": fingerprint, "classifier_sha256": classifier_hash,
                          "sessions": sessions, "validation": metrics,
                          "grid_version": 1}
    return chosen, {"calibration": calibration, "residual_grid": grid}


def evaluate(package, checkpoint, output, shuffles, seed, export):
    reports = {}
    sessions = checkpoint["sessions"]
    for partition in ("train", "validation", "test"):
        indices = package["splits"][partition]
        groups = {"training_sessions_mean": sessions, "session3": [2]}
        for group, selected in groups.items():
            coarse, residual, probabilities = predict_parts(package["eeg"][indices][:, selected], checkpoint)
            predictions = {"mean": torch.zeros_like(coarse), "predicted_centers": coarse,
                           "centers_plus_residual": coarse + checkpoint["residual_weight"] * residual}
            baseline = measure(predictions["mean"], package, indices)
            for variant, pred in predictions.items():
                metrics = measure(pred, package, indices)
                generator = torch.Generator().manual_seed(seed + 10000)
                shuffle_values = []
                if variant == "centers_plus_residual":
                    for _ in range(shuffles):
                        permutation = within_category_permutation(package["categories"][indices], generator)
                        shuffled = coarse + checkpoint["residual_weight"] * residual[permutation]
                        shuffle_values.append(measure(shuffled, package, indices)["within_category_mrr"])
                key = f"{partition}/{group}/{variant}"
                reports[key] = {"video_count": len(indices), "metrics": metrics, "mean_baseline": baseline,
                                "predicted_category_accuracy": float(probabilities.argmax(-1).eq(package["categories"][indices]).float().mean()),
                                "residual_shuffle_mrr": shuffle_values,
                                "residual_shuffle_mean_mrr": sum(shuffle_values)/len(shuffle_values) if shuffle_values else None,
                                "residual_weight": checkpoint["residual_weight"] if variant == "centers_plus_residual" else 0.,
                                "raw_residual_between_video_rms": float((residual-residual.mean(0)).square().mean().sqrt()),
                                "heldout_session": sessions == [0,1] and group == "session3"}
                destination = output / partition / group / variant
                destination.mkdir(parents=True, exist_ok=True)
                atomic_save({"predicted": pred, "video_ids": [package["ids"][i] for i in indices]}, destination / "predictions.pt")
                if export and partition == "test" and group == "training_sessions_mean":
                    projector = ToraPCAProjector(**package["token_projector"])
                    rows = []
                    for position, i in enumerate(indices):
                        hidden = projector.decode(decode(pred[position:position+1], package["pca"])[0])
                        if hidden.shape != (226,4096) or not torch.isfinite(hidden).all():
                            raise ValueError("Invalid exported Tora condition")
                        path = destination / f"{package['ids'][i]}.pt"
                        atomic_save({"schema_version":2, "video_id":package["ids"][i], "hidden_state":hidden,
                                     "caption":"EEG hierarchical continuous condition", "source_method":variant,
                                     "source_checkpoint":str((output/'best.pt').resolve())},path)
                        rows.append({"video_id":package["ids"][i],"condition_path":str(path.resolve()),"trial_count":len(sessions)})
                    (destination/'video_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
            print(f"[c2-hierarchical] evaluated {partition}/{group}",flush=True)
    result = {"schema_version":1,"prepared_sha256":checkpoint["prepared_sha256"],
              "classifier_sha256":checkpoint["classifier_sha256"],"sessions":sessions,
              "temperature":checkpoint["temperature"],"alpha":checkpoint["residual_fit"]["alpha"],
              "selected_residual_weight":checkpoint["residual_weight"],"reports":reports,
              "shuffle_note":"Only residuals are permuted within true category; coarse predictions remain fixed. Diagnostic, not a calibrated p-value."}
    atomic_json(result,output/'report.json')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt'))
    parser.add_argument('--category-root',type=Path,default=Path('outputs/eeg_semantic/session_category_control'))
    parser.add_argument('--output-root',type=Path,default=Path('outputs/eeg_semantic/c2_hierarchical_v4'))
    parser.add_argument('--protocol',choices=('all_sessions','holdout_s3','both'),default='both')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--shuffles',type=int,default=20)
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    protocols=('all_sessions','holdout_s3') if args.protocol=='both' else (args.protocol,)
    if args.dry_run:
        for protocol in protocols:
            print(f"[c2-hierarchical] {protocol}: reuse {args.category_root/protocol/'ridge'/f'seed{args.seed}'/'best.pt'}; fit train centers/residual ridge; select on validation; evaluate three variants")
        print('[c2-hierarchical] dry-run COMPLETE')
        return
    torch.set_num_threads(args.threads)
    fingerprint=digest(args.prepared)
    source=torch.load(args.prepared,map_location='cpu',weights_only=False)
    for protocol in protocols:
        classifier_path=args.category_root/protocol/'ridge'/f'seed{args.seed}'/'best.pt'
        classifier_hash=digest(classifier_path)
        classifier=torch.load(classifier_path,map_location='cpu',weights_only=False)
        signature=classifier['signature']
        if signature['prepared_sha256']!=fingerprint or signature['protocol']!=protocol or signature['model']!='ridge':
            raise ValueError('Classifier/prepared/protocol mismatch')
        sessions=[0,1,2] if protocol=='all_sessions' else [0,1]
        package=normalize_sessions(source,sessions)
        output=args.output_root/protocol/f'seed{args.seed}'
        output.mkdir(parents=True,exist_ok=True)
        if (output/'best.pt').exists():
            checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
            if checkpoint['prepared_sha256']!=fingerprint or checkpoint['classifier_sha256']!=classifier_hash or checkpoint['sessions']!=sessions or checkpoint['grid_version']!=1:
                raise ValueError('Existing run differs; use a new --output-root')
        else:
            checkpoint,grid=fit(package,classifier['fit'],sessions,fingerprint,classifier_hash)
            atomic_json(grid,output/'validation_grid.json')
            atomic_save(checkpoint,output/'best.pt')
        evaluate(package,checkpoint,output,args.shuffles,args.seed,args.export)
        atomic_json({'status':'complete','prepared_sha256':fingerprint},output/'completed.json')
        print(f"[c2-hierarchical] {protocol} residual_weight={checkpoint['residual_weight']} report={output/'report.json'}",flush=True)
    print('[c2-hierarchical] COMPLETE')


if __name__=='__main__':
    main()
