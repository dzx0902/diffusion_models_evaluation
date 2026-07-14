"""Run Wan generation with PCA-compressed text states.

The wrapper patches Wan's native T5EncoderModel at runtime:

    hidden_4096 -> PCA down to k -> PCA inverse back to 4096

Then it executes Wan's original generate.py with the remaining arguments.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Wrap Wan generate.py with token-level PCA reconstruction.",
        add_help=True,
    )
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--project-dim", type=int, required=True)
    parser.add_argument(
        "--disable-projection",
        action="store_true",
        help="Run Wan generate.py without patching, useful for paired baseline commands.",
    )
    parser.add_argument(
        "--report-error",
        action="store_true",
        help="Print per-prompt reconstruction MSE/cosine diagnostics.",
    )
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    return args, remaining


class PcaProjector:
    def __init__(self, path: Path, dim: int) -> None:
        payload = np.load(path)
        components = payload["components"]
        if dim > components.shape[0]:
            raise ValueError(
                f"Requested project dim {dim}, but projector only has {components.shape[0]} components."
            )
        self.dim = dim
        self.mean_np = payload["mean"].astype(np.float32)
        self.components_np = components[:dim].astype(np.float32)
        self._cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def tensors_for(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        compute_dtype = torch.float32
        key = (str(device), compute_dtype)
        if key not in self._cache:
            mean = torch.from_numpy(self.mean_np).to(device=device, dtype=compute_dtype)
            components = torch.from_numpy(self.components_np).to(device=device, dtype=compute_dtype)
            self._cache[key] = (mean, components)
        return self._cache[key]

    def reconstruct(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        original_dtype = hidden.dtype
        mean, components = self.tensors_for(hidden.device, hidden.dtype)
        x = hidden.float()
        centered = x - mean
        low = centered @ components.t()
        recon = low @ components + mean
        diff = recon - x
        mse = float(diff.pow(2).mean().detach().cpu().item())
        denom = torch.linalg.norm(x, dim=-1) * torch.linalg.norm(recon, dim=-1)
        cosine = float(((x * recon).sum(dim=-1) / denom.clamp_min(1e-8)).mean().detach().cpu().item())
        return recon.to(dtype=original_dtype), {"mse": mse, "cosine": cosine}


def patch_wan_t5(projector: PcaProjector, report_error: bool) -> None:
    from wan.modules.t5 import T5EncoderModel

    original_call = T5EncoderModel.__call__

    def patched_call(self, texts, device):  # type: ignore[no-untyped-def]
        contexts = original_call(self, texts, device)
        patched = []
        for idx, context in enumerate(contexts):
            recon, stats = projector.reconstruct(context)
            if report_error:
                print(
                    f"[wan-projector] text_index={idx} dim={projector.dim} "
                    f"mse={stats['mse']:.6e} cosine={stats['cosine']:.6f}",
                    flush=True,
                )
            patched.append(recon)
        return patched

    T5EncoderModel.__call__ = patched_call


def main() -> None:
    args, wan_args = parse_args()
    repo = args.wan_repo.resolve()
    generate_py = repo / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(generate_py)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if not args.disable_projection:
        projector = PcaProjector(args.projector, args.project_dim)
        patch_wan_t5(projector, args.report_error)
        print(
            f"[wan-projector] enabled projector={args.projector} dim={args.project_dim}",
            flush=True,
        )
    else:
        print("[wan-projector] projection disabled; running baseline", flush=True)

    old_argv = sys.argv
    try:
        sys.argv = [str(generate_py), *wan_args]
        runpy.run_path(str(generate_py), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
