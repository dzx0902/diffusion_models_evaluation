"""Video-level residual PCA and anti-collapse objectives for C2-v2."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from EEG2Caption.src.common import CompactEEGClassifier


def fit_pca(train: torch.Tensor, dim: int, floor: float = 0.1) -> dict:
    """Fit ONLY on train videos; sample-space eigendecomposition avoids a D x D matrix."""
    x = train.flatten(1).float()
    if not 0 < dim < len(x) or not 0 < floor <= 1:
        raise ValueError("Require 0 < dim < train videos and 0 < floor <= 1")
    mean = x.mean(0)
    centered = x - mean
    eigenvalues, u = torch.linalg.eigh((centered @ centered.T).double())
    eigenvalues = eigenvalues.flip(0)[:dim]
    u = u.flip(1)[:, :dim].float()
    if eigenvalues[-1] <= eigenvalues[0] * 1e-7:
        raise ValueError("Requested dimension exceeds stable train residual rank; use fewer dimensions")
    singular = eigenvalues.sqrt().float()
    basis = (u.T @ centered) / singular[:, None]
    # Re-orthogonalize after sample-space numerical arithmetic.
    basis = torch.linalg.qr(basis.T, mode="reduced").Q.T.contiguous()
    scores = centered @ basis.T
    scale = scores.std(0, unbiased=True)
    scale = scale.clamp_min(scale.max() * floor)
    return {"mean": mean, "basis": basis, "scale": scale,
            "shape": tuple(train.shape[1:]), "fit_video_count": len(train)}


def encode(x: torch.Tensor, pca: dict) -> torch.Tensor:
    return ((x.flatten(1) - pca["mean"]) @ pca["basis"].T) / pca["scale"]


def decode(z: torch.Tensor, pca: dict) -> torch.Tensor:
    return (pca["mean"] + (z * pca["scale"]) @ pca["basis"]).reshape(len(z), *pca["shape"])


class ResidualEEG(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.encoder = CompactEEGClassifier(feature_dim=128, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128),
                                  nn.GELU(), nn.Linear(128, dim))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(eeg)["fused_feature"])


def contrastive_loss(pred: torch.Tensor, bank: torch.Tensor, positives: torch.Tensor,
                     temperature: float = 0.1) -> torch.Tensor:
    if not positives.any(1).all():
        raise ValueError("Every EEG needs at least one positive caption")
    logits = F.normalize(pred, dim=-1) @ F.normalize(bank, dim=-1).T / temperature
    return (torch.logsumexp(logits, 1) -
            torch.logsumexp(logits.masked_fill(~positives, -torch.inf), 1)).mean()


def variance_covariance(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = pred - pred.mean(0)
    variance = F.relu(1 - (centered.square().mean(0) + 1e-4).sqrt()).mean()
    cov = centered.T @ centered / max(1, len(pred) - 1)
    off = cov - torch.diag_embed(cov.diagonal())
    return variance, off.square().sum() / pred.shape[1]


def retrieval(pred: torch.Tensor, target: torch.Tensor, positives: torch.Tensor,
              categories: torch.Tensor) -> dict[str, float]:
    """Multi-positive retrieval, with expected scores under uniformly broken ties."""
    similarities = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T
    result = {}
    for prefix, allowed in (("global", torch.ones_like(positives)),
                            ("within_category", categories[:, None] == categories[None, :])):
        scores = similarities.masked_fill(~allowed, -torch.inf)
        correct = positives & allowed
        best = scores.masked_fill(~correct, -torch.inf).max(1).values
        greater = ((scores > best[:, None] + 1e-7) & allowed).sum(1)
        ties = ((scores - best[:, None]).abs() <= 1e-7) & allowed
        tied_count = ties.sum(1)
        tied_positive = (ties & correct).sum(1)
        # Distribution of the first positive in the tied block, without replacement.
        mrr = torch.zeros(len(pred))
        r1 = torch.zeros(len(pred))
        r5 = torch.zeros(len(pred))
        for i in range(len(pred)):
            survival = 1.0
            n, m, g = int(tied_count[i]), int(tied_positive[i]), int(greater[i])
            for position in range(1, n - m + 2):
                probability = survival * m / (n - position + 1)
                rank = g + position
                mrr[i] += probability / rank
                r1[i] += probability * (rank <= 1)
                r5[i] += probability * (rank <= 5)
                survival *= (n - position + 1 - m) / (n - position + 1)
        result.update({f"{prefix}_r1": float(r1.mean()), f"{prefix}_r5": float(r5.mean()),
                       f"{prefix}_mrr": float(mrr.mean())})
    result["residual_mse"] = float(F.mse_loss(pred, target))
    result["residual_cosine"] = float(F.cosine_similarity(pred, target, dim=-1).mean())
    result["between_video_rms"] = float((pred - pred.mean(0)).square().mean().sqrt())
    result["target_between_video_rms"] = float((target - target.mean(0)).square().mean().sqrt())
    return result


def within_category_permutation(categories: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    permutation = torch.arange(len(categories))
    for category in categories.unique():
        indices = torch.where(categories == category)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        permutation[indices] = indices.roll(1)
    return permutation
