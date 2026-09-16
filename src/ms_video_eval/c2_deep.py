"""Ordered EEG encoder for fixed-update C2 experiments (no video input at inference)."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class DeepC2(nn.Module):
    def __init__(self, dim=64, subjects=1, width=256, layers=4, heads=8,
                 tokens=50, frames=8, visual_dim=512, dropout=0.15):
        super().__init__()
        self.tokens, self.frames = tokens, frames
        self.stem = nn.Sequential(
            nn.Conv1d(62, width, 25, stride=4, padding=12), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width, 15, stride=4, padding=7, groups=width),
            nn.Conv1d(width, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True,
            norm_first=True), layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.residual = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(width, dim))
        self.visual = nn.Linear(width, visual_dim)
        self.text = nn.Linear(width, visual_dim)
        # Keep shared-weight initialization identical across donor counts.
        self.subject_scale = nn.Embedding(subjects, 62)
        self.subject_bias = nn.Embedding(subjects, 62)
        nn.init.zeros_(self.subject_scale.weight)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(self, eeg, subject):
        b, s, c, t = eeg.shape
        scale = 1 + 0.1 * self.subject_scale(subject).tanh()
        bias = 0.1 * self.subject_bias(subject).tanh()
        x = eeg * scale[:, None, :, None] + bias[:, None, :, None]
        x = F.adaptive_avg_pool1d(self.stem(x.reshape(b*s, c, t)), self.tokens)
        x = self.norm(self.encoder(x.transpose(1, 2) + self.position))
        pooled = x.mean(1)
        sessions = self.residual(pooled).reshape(b, s, -1)
        ordered = F.adaptive_avg_pool1d(x.transpose(1, 2), self.frames).transpose(1, 2)
        return {"z": sessions.mean(1), "sessions": sessions,
                "visual": self.visual(ordered).reshape(b, s, self.frames, -1).mean(1),
                "text": self.text(pooled).reshape(b, s, -1).mean(1)}


def temporal_visual_loss(pred, target):
    """Ordered normalized frame alignment plus consecutive-frame differences."""
    p, t = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    appearance = (1 - (p*t).sum(-1)).mean()
    # Do not normalize differences: identical frames should have zero motion target.
    dynamics = (p.diff(dim=1) - t.diff(dim=1)).square().sum(-1).mean()
    return appearance + dynamics


def learning_rate(step, updates, warmup, peak):
    if not 1 <= step <= updates:
        raise ValueError("Step outside optimizer-update budget")
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, updates - warmup)
    return peak * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)


def validate_splits(package):
    ids, splits = package["ids"], package["splits"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video identifiers")
    groups = [set(splits[k]) for k in ("train", "validation", "test")]
    if any(not g for g in groups) or sum(map(len, groups)) != len(set.union(*groups)):
        raise ValueError("Empty or overlapping video partitions")
    if set.union(*groups) != set(range(len(ids))):
        raise ValueError("Partitions must cover exactly the prepared videos")
    if any(v.split('-')[0] not in {"01", "02", "03", "04", "05", "06"} for v in ids):
        raise ValueError("This protocol is first-six 4-second videos only")


def validate_donor_ids(donor_ids, package):
    allowed = {package["ids"][i] for i in package["splits"]["train"]}
    if len(donor_ids) != len(set(donor_ids)) or set(donor_ids) != allowed:
        raise ValueError("Donor cache must contain exactly global TRAIN video IDs, never validation/test")
