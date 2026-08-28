"""Tora text-condition cache and exact cross-attention injection helpers.

The official Alibaba Tora 5B SAT model config uses ``T5Tokenizer`` and
``T5EncoderModel`` with ``max_length=226`` and forwards the resulting rank-3
tensor as ``conditioning['crossattn']``.  The reference embedder does not pass
the tokenizer attention mask into T5, so compatibility targets retain all 226
padded positions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import numpy as np


TORA_TEXT_TOKENS = 226
TORA_T5_HIDDEN_DIM = 4096


@dataclass(frozen=True)
class ToraPCAProjector:
    mean: torch.Tensor
    components: torch.Tensor

    @classmethod
    def load(cls, path: Path, dim: int | None = None) -> "ToraPCAProjector":
        package = np.load(path, allow_pickle=False)
        mean = torch.from_numpy(package["mean"]).float()
        components = torch.from_numpy(package["components"]).float()
        if dim is not None:
            if not 1 <= dim <= components.shape[0]:
                raise ValueError(f"PCA dim must be in [1, {components.shape[0]}]")
            components = components[:dim]
        if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != mean.shape[0]:
            raise ValueError("Invalid Tora PCA projector shapes")
        return cls(mean=mean, components=components)

    def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.shape[-1] != self.mean.shape[0]:
            raise ValueError("Tora state/projector dimension mismatch")
        return (hidden_state - self.mean.to(hidden_state)) @ self.components.to(hidden_state).t()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.components.shape[0]:
            raise ValueError("Tora latent/projector dimension mismatch")
        return latent @ self.components.to(latent) + self.mean.to(latent)


@dataclass(frozen=True)
class ToraTextCondition:
    video_id: str
    caption: str
    hidden_state: torch.Tensor
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    def validate(
        self,
        tokens: int = TORA_TEXT_TOKENS,
        hidden_dim: int | None = TORA_T5_HIDDEN_DIM,
    ) -> None:
        if self.hidden_state.ndim != 2 or self.hidden_state.shape[0] != tokens:
            raise ValueError(
                f"Expected Tora hidden state [{tokens}, hidden], got {tuple(self.hidden_state.shape)}"
            )
        if hidden_dim is not None and self.hidden_state.shape[1] != hidden_dim:
            raise ValueError(
                f"Expected Tora hidden dimension {hidden_dim}, got {self.hidden_state.shape[1]}"
            )
        for name, value in (("input_ids", self.input_ids), ("attention_mask", self.attention_mask)):
            if value is not None and tuple(value.shape) != (tokens,):
                raise ValueError(f"{name} must have shape [{tokens}], got {tuple(value.shape)}")


def load_tora_condition(path: Path, hidden_dim: int | None = TORA_T5_HIDDEN_DIM) -> ToraTextCondition:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    condition = ToraTextCondition(
        video_id=str(payload["video_id"]),
        caption=str(payload["caption"]),
        hidden_state=payload["hidden_state"].float(),
        input_ids=payload.get("input_ids"),
        attention_mask=payload.get("attention_mask"),
    )
    condition.validate(hidden_dim=hidden_dim)
    return condition


def read_tora_condition_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row["video_id"])
        if video_id in records:
            raise ValueError(f"Duplicate Tora target video_id: {video_id}")
        records[video_id] = row
    return records


def inject_tora_crossattn(
    conditioning: Mapping[str, Any],
    hidden_state: torch.Tensor,
) -> dict[str, Any]:
    """Return a shallow condition copy with only semantic cross-attention replaced."""

    if "crossattn" not in conditioning:
        raise KeyError("Tora conditioning has no 'crossattn' entry")
    native = conditioning["crossattn"]
    if not isinstance(native, torch.Tensor) or native.ndim != 3:
        raise ValueError("Tora native crossattn must be a [batch, tokens, hidden] tensor")
    if hidden_state.ndim == 2:
        hidden_state = hidden_state.unsqueeze(0)
    if hidden_state.ndim != 3 or hidden_state.shape[1:] != native.shape[1:]:
        raise ValueError(
            f"Injected condition shape {tuple(hidden_state.shape)} is incompatible with "
            f"native {tuple(native.shape)}"
        )
    if hidden_state.shape[0] not in {1, native.shape[0]}:
        raise ValueError("Injected condition batch must be 1 or match native batch")
    if hidden_state.shape[0] == 1 and native.shape[0] > 1:
        hidden_state = hidden_state.expand(native.shape[0], -1, -1)
    result = dict(conditioning)
    result["crossattn"] = hidden_state.to(device=native.device, dtype=native.dtype)
    return result
