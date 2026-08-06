"""Helpers for injecting a reconstructed positive Wan text condition."""

from __future__ import annotations

from collections.abc import Callable

import torch


def first_call_context_injector(
    original_call: Callable[..., list[torch.Tensor]],
    context: torch.Tensor,
) -> Callable[..., list[torch.Tensor]]:
    """Inject ``context`` once, then preserve native T5 calls.

    Wan TI2V encodes the positive prompt first and the negative prompt second.
    Replacing every call would make both CFG branches use the same condition.
    """

    positive_pending = True

    def patched_call(self, texts, device):  # type: ignore[no-untyped-def]
        nonlocal positive_pending
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Wan T5 expects a non-empty batch of prompts.")
        if positive_pending:
            positive_pending = False
            condition = context.to(device=torch.device(device), dtype=torch.bfloat16)
            return [condition.clone() for _ in texts]
        return original_call(self, texts, device)

    return patched_call
