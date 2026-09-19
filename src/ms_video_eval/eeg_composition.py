"""Train-on-pairs/test-on-triples protocol and leakage-safe metrics."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .semantic_schema import CORE_ENTITIES_BY_CATEGORY

OBJECTS = ("person", "dog", "car", "ball", "flower", "bird")
PROTOCOLS = {"cs_s1": [1, 2], "cs_s2": [0, 2], "cs_s3": [0, 1], "session_average": [0, 1, 2]}


def labels_for(ids):
    return torch.tensor([[float(o in CORE_ENTITIES_BY_CATEGORY[v[:2]]) for o in OBJECTS] for v in ids])


def split_videos(ids, seed=42, val_per_class=8):
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs")
    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    for c in range(1, 9):
        positions = sorted((i for i,v in enumerate(ids) if v.startswith(f"{c:02d}-")), key=lambda i: ids[i])
        if len(positions) != 78:
            raise ValueError(f"Category {c:02d} requires 78 videos, found {len(positions)}")
        if c <= 6:
            shuffled = rng.permutation(positions).tolist()
            splits["validation"].extend(shuffled[:val_per_class])
            splits["train"].extend(shuffled[val_per_class:])
        else:
            splits["test"].extend(positions)
    if not 0 < val_per_class < 78 or sum(map(len,splits.values())) != len(ids):
        raise ValueError("Invalid category split")
    return splits


def fit_normalization(eeg, indices, sessions):
    # Shared channel statistics fitted ONLY on allowed train sessions, first 4s.
    x = eeg[indices][:, sessions, :, :800]
    return x.mean((0,1,3)), x.std((0,1,3)).clamp_min(1e-6)


def set_metrics(pred, target):
    p, y = pred.bool(), target.bool()
    tp = (p & y).sum(0).float()
    fp = (p & ~y).sum(0).float()
    fn = (~p & y).sum(0).float()
    return {"exact_set_accuracy": float((p == y).all(1).float().mean()),
            "micro_precision": float(tp.sum()/(tp+fp).sum().clamp_min(1)),
            "micro_recall": float(tp.sum()/(tp+fn).sum().clamp_min(1)),
            "micro_f1": float(2*tp.sum()/(2*tp+fp+fn).sum().clamp_min(1)),
            "macro_f1_six_labels": float((2*tp/(2*tp+fp+fn).clamp_min(1)).mean()),
            "predicted_count_mean": float(p.sum(1).float().mean()),
            "predicted_count_histogram": torch.bincount(p.sum(1), minlength=7).tolist()}


def select_threshold(logits, labels):
    probabilities = logits.sigmoid()
    grid = [.1+.05*i for i in range(17)]
    # Micro-F1 on 01--06 validation only; ties prefer proximity to .5.
    return max(grid, key=lambda t: (set_metrics(probabilities >= t, labels)["micro_f1"], -abs(t-.5)))


def metrics(logits, labels, threshold, cardinality=3):
    probs = logits.sigmoid()
    top = torch.zeros_like(labels, dtype=torch.bool)
    top.scatter_(1, probs.topk(cardinality, dim=1).indices, True)
    per_object = {}
    informative = []
    predicted = probs >= threshold
    for j, name in enumerate(OBJECTS):
        y = labels[:,j].bool()
        variable = bool(y.any() and (~y).any())
        ap = float(average_precision_score(y.numpy(), probs[:,j].numpy())) if variable else None
        if ap is not None:
            informative.append(ap)
        per_object[name] = {"positive_count": int(y.sum()), "ap": ap,
                            "recall": float(predicted[y,j].float().mean()) if y.any() else None,
                            "false_positive_rate": float(predicted[~y,j].float().mean()) if (~y).any() else None}
    return {"video_count": len(labels), "topk": cardinality,
            "topk_uses_known_cardinality": True, "topk_metrics": set_metrics(top, labels),
            "threshold": threshold, "threshold_metrics": set_metrics(predicted, labels),
            "informative_macro_ap": float(np.mean(informative)) if informative else None,
            "per_object": per_object}
