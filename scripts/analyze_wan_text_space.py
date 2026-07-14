"""Analyze Wan2.2 text-encoder embedding dimensionality.

This script measures whether prompt embeddings occupy a lower-dimensional
subspace than the raw UMT5 hidden size. It does not generate videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, UMT5EncoderModel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.prompt_builder import build_prompt
from ms_video_eval.task_schema import load_tasks
from ms_video_eval.utils import ensure_dir, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA/effective-rank analysis on Wan2.2 text embeddings."
    )
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs" / "ms_eval_tasks.yaml")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional txt/jsonl file. txt uses one prompt per line; jsonl reads the 'prompt' field.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "ms_eval" / "metrics" / "generation_manifest.jsonl",
        help="Optional generation manifest; unique prompt fields are included when present.",
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="",
        help=(
            "Local text encoder directory or HF id. If omitted, the script tries "
            "$WAN_TEXT_ENCODER, local Wan diffusers paths, then google/umt5-xxl."
        ),
    )
    parser.add_argument(
        "--encoder-backend",
        choices=["auto", "hf", "wan"],
        default="auto",
        help="Use HuggingFace from_pretrained or Wan's native T5 .pth loader.",
    )
    parser.add_argument(
        "--wan-repo",
        type=Path,
        default=None,
        help="Path to the Wan2.2 repository when --encoder-backend wan is used.",
    )
    parser.add_argument(
        "--wan-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing models_t5_umt5-xxl-enc-bf16.pth and google/umt5-xxl.",
    )
    parser.add_argument(
        "--wan-t5-checkpoint",
        type=Path,
        default=None,
        help="Explicit path to models_t5_umt5-xxl-enc-bf16.pth.",
    )
    parser.add_argument(
        "--wan-tokenizer-dir",
        type=Path,
        default=None,
        help="Explicit path to Wan's local google/umt5-xxl tokenizer directory.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-token-samples",
        type=int,
        default=20000,
        help="Maximum valid token vectors to keep for token-level PCA.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 768, 1024, 1536, 2048],
        help="Candidate projection dimensions to report.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.9, 0.95, 0.99],
        help="Explained-variance thresholds.",
    )
    parser.add_argument(
        "--near-zero-eps",
        type=float,
        default=1e-3,
        help="Absolute-value threshold for reporting activation near-zero rates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "text_space" / "wan2_2_ti2v_5b",
    )
    return parser.parse_args()


class WanNativeTextEncoder:
    """Small adapter around Wan's native T5EncoderModel."""

    def __init__(
        self,
        repo: Path,
        checkpoint_path: Path,
        tokenizer_path: Path,
        text_len: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from wan.modules.t5 import T5EncoderModel

        self.device = device
        self.model = T5EncoderModel(
            text_len=text_len,
            dtype=dtype,
            device=torch.device(device),
            checkpoint_path=str(checkpoint_path),
            tokenizer_path=str(tokenizer_path),
        )

    def encode(self, prompts: list[str]) -> list[torch.Tensor]:
        return self.model(prompts, torch.device(self.device))


def _read_prompt_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        return [str(row.get("prompt", "")).strip() for row in rows if row.get("prompt")]
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.tasks.exists():
        prompts.extend(build_prompt(task) for task in load_tasks(args.tasks))
    if args.prompt_file:
        prompts.extend(_read_prompt_file(args.prompt_file))
    if args.manifest.exists():
        prompts.extend(str(row.get("prompt", "")).strip() for row in read_jsonl(args.manifest))

    seen: set[str] = set()
    unique: list[str] = []
    for prompt in prompts:
        if prompt and prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    if not unique:
        raise ValueError("No prompts found. Provide --tasks, --prompt-file, or --manifest.")
    return unique


def resolve_text_encoder(user_value: str) -> str:
    if user_value:
        return user_value
    env_value = os.environ.get("WAN_TEXT_ENCODER", "").strip()
    if env_value:
        return env_value

    models_root = os.environ.get("MS_MODELS_ROOT", "").strip()
    candidates: list[Path] = []
    if models_root:
        wan_root = Path(models_root) / "Wan2.2" / "Wan2.2-TI2V-5B"
        candidates.extend(
            [
                wan_root / "text_encoder",
                wan_root / "google" / "umt5-xxl",
                wan_root / "umt5-xxl",
            ]
        )
        diffusers_root = Path(models_root) / "Wan2.2-TI2V-5B-Diffusers"
        candidates.append(diffusers_root / "text_encoder")

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)
    return "google/umt5-xxl"


def resolve_wan_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    models_root = Path(os.environ.get("MS_MODELS_ROOT", "")).expanduser()
    repo = args.wan_repo
    checkpoint_dir = args.wan_checkpoint_dir

    if repo is None:
        repo = models_root / "Wan2.2"
    if checkpoint_dir is None:
        checkpoint_dir = repo / "Wan2.2-TI2V-5B"

    checkpoint_path = args.wan_t5_checkpoint or checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = args.wan_tokenizer_dir or checkpoint_dir / "google" / "umt5-xxl"

    missing = [
        str(path)
        for path in [repo, checkpoint_path, tokenizer_path]
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Wan native text encoder paths missing: " + ", ".join(missing))
    return Path(repo), Path(checkpoint_path), Path(tokenizer_path)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_text_model(model_id: str, args: argparse.Namespace):
    backend = args.encoder_backend
    if backend == "auto" and (args.wan_repo or args.wan_checkpoint_dir or args.wan_t5_checkpoint):
        backend = "wan"
    if backend == "wan":
        repo, checkpoint_path, tokenizer_path = resolve_wan_paths(args)
        return (
            "wan",
            WanNativeTextEncoder(
                repo=repo,
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                text_len=args.max_length,
                dtype=torch_dtype(args.dtype),
                device=args.device,
            ),
            None,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    try:
        model = UMT5EncoderModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype(args.dtype),
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype(args.dtype),
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
    model.to(args.device)
    model.eval()
    return "hf", tokenizer, model


def encode_prompts(
    prompts: list[str],
    backend: str,
    tokenizer_or_encoder: Any,
    model: torch.nn.Module | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prompt_vectors: list[np.ndarray] = []
    token_chunks: list[np.ndarray] = []
    valid_lengths: list[int] = []
    near_zero_rates: list[float] = []
    token_budget = args.max_token_samples

    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            if backend == "wan":
                contexts = tokenizer_or_encoder.encode(batch)
                for valid in contexts:
                    valid = valid.float()
                    valid_lengths.append(int(valid.shape[0]))
                    prompt_vectors.append(valid.mean(dim=0).cpu().numpy())
                    near_zero_rates.append(float((valid.abs() < args.near_zero_eps).float().mean().item()))
                    if token_budget > 0:
                        take = min(token_budget, int(valid.shape[0]))
                        token_chunks.append(valid[:take].cpu().numpy())
                        token_budget -= take
                continue

            tokenizer = tokenizer_or_encoder
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded["attention_mask"].to(args.device)
            if model is None:
                raise ValueError("HuggingFace backend requires a model.")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state.float()
            mask = attention_mask.bool()

            for idx in range(hidden.shape[0]):
                valid = hidden[idx][mask[idx]]
                valid_lengths.append(int(valid.shape[0]))
                prompt_vectors.append(valid.mean(dim=0).cpu().numpy())
                near_zero_rates.append(float((valid.abs() < args.near_zero_eps).float().mean().item()))
                if token_budget > 0:
                    take = min(token_budget, int(valid.shape[0]))
                    token_chunks.append(valid[:take].cpu().numpy())
                    token_budget -= take

    prompt_matrix = np.stack(prompt_vectors).astype(np.float64)
    token_matrix = (
        np.concatenate(token_chunks, axis=0).astype(np.float64)
        if token_chunks
        else np.empty((0, prompt_matrix.shape[1]), dtype=np.float64)
    )
    stats = {
        "prompt_count": len(prompts),
        "embedding_dim": int(prompt_matrix.shape[1]),
        "max_length": args.max_length,
        "avg_valid_tokens": float(np.mean(valid_lengths)),
        "min_valid_tokens": int(np.min(valid_lengths)),
        "max_valid_tokens": int(np.max(valid_lengths)),
        "padding_fraction": float(1.0 - (np.mean(valid_lengths) / args.max_length)),
        "activation_near_zero_rate_mean": float(np.mean(near_zero_rates)),
        "activation_near_zero_eps": args.near_zero_eps,
        "token_vectors_analyzed": int(token_matrix.shape[0]),
    }
    return prompt_matrix, token_matrix, stats


def pca_stats(matrix: np.ndarray, dims: list[int], thresholds: list[float]) -> dict[str, Any]:
    if matrix.shape[0] < 2:
        return {"error": "Need at least two vectors for PCA."}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values**2) / max(matrix.shape[0] - 1, 1)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {"error": "Zero variance matrix."}
    ratio = eigenvalues / total
    cumulative = np.cumsum(ratio)
    entropy = -float(np.sum(ratio * np.log(ratio + 1e-30)))
    effective_rank = float(math.exp(entropy))
    participation_ratio = float((eigenvalues.sum() ** 2) / np.sum(eigenvalues**2))

    dim_rows = []
    max_rank = len(cumulative)
    for dim in dims:
        clipped = min(dim, max_rank)
        explained = float(cumulative[clipped - 1]) if clipped > 0 else 0.0
        dim_rows.append(
            {
                "dim": dim,
                "effective_dim_used": clipped,
                "explained_variance": explained,
                "reconstruction_mse_fraction": float(1.0 - explained),
            }
        )

    threshold_rows = []
    for threshold in thresholds:
        idx = int(np.searchsorted(cumulative, threshold, side="left"))
        threshold_rows.append(
            {
                "threshold": threshold,
                "dim": min(idx + 1, max_rank),
                "attainable": bool(cumulative[-1] >= threshold),
            }
        )

    return {
        "sample_count": int(matrix.shape[0]),
        "raw_dim": int(matrix.shape[1]),
        "rank_upper_bound": int(max_rank),
        "effective_rank_entropy": effective_rank,
        "effective_rank_participation": participation_ratio,
        "top_explained_variance": [float(x) for x in cumulative[: min(32, max_rank)]],
        "dims": dim_rows,
        "thresholds": threshold_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Wan2.2 Text-Space Dimensionality Analysis",
        "",
        "## Encoder",
        f"- Backend: `{result.get('encoder_backend', 'unknown')}`",
        f"- Text encoder: `{result['text_encoder']}`",
        f"- Raw embedding dim: {result['embedding_stats']['embedding_dim']}",
        f"- Prompt count: {result['embedding_stats']['prompt_count']}",
        f"- Avg valid tokens: {result['embedding_stats']['avg_valid_tokens']:.2f}",
        f"- Padding fraction at max length: {result['embedding_stats']['padding_fraction']:.3f}",
        f"- Activation near-zero rate: {result['embedding_stats']['activation_near_zero_rate_mean']:.4f}",
        "",
        "## Prompt-Level PCA",
    ]
    prompt_pca = result["prompt_pca"]
    if "error" in prompt_pca:
        lines.append(f"- {prompt_pca['error']}")
    else:
        lines.extend(
            [
                f"- Effective rank, entropy: {prompt_pca['effective_rank_entropy']:.2f}",
                f"- Effective rank, participation: {prompt_pca['effective_rank_participation']:.2f}",
                "",
                "| dim | effective_dim_used | explained_variance | reconstruction_mse_fraction |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prompt_pca["dims"]:
            lines.append(
                f"| {row['dim']} | {row['effective_dim_used']} | "
                f"{row['explained_variance']:.6f} | {row['reconstruction_mse_fraction']:.6f} |"
            )
        lines.extend(["", "Threshold dimensions:"])
        for row in prompt_pca["thresholds"]:
            lines.append(f"- {row['threshold']:.2f}: {row['dim']} dims")

    lines.extend(["", "## Token-Level PCA"])
    token_pca = result["token_pca"]
    if "error" in token_pca:
        lines.append(f"- {token_pca['error']}")
    else:
        lines.extend(
            [
                f"- Token vectors analyzed: {token_pca['sample_count']}",
                f"- Effective rank, entropy: {token_pca['effective_rank_entropy']:.2f}",
                f"- Effective rank, participation: {token_pca['effective_rank_participation']:.2f}",
                "",
                "| dim | effective_dim_used | explained_variance | reconstruction_mse_fraction |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in token_pca["dims"]:
            lines.append(
                f"| {row['dim']} | {row['effective_dim_used']} | "
                f"{row['explained_variance']:.6f} | {row['reconstruction_mse_fraction']:.6f} |"
            )
        lines.extend(["", "Threshold dimensions:"])
        for row in token_pca["thresholds"]:
            lines.append(f"- {row['threshold']:.2f}: {row['dim']} dims")

    lines.extend(
        [
            "",
            "## Practical Interpretation",
            "- Prompt-level PCA estimates whether whole-prompt semantics can be aligned in a lower-dimensional space.",
            "- Token-level PCA is stricter and closer to what cross-attention consumes.",
            "- A candidate dimension is promising only if generation ablation keeps object, count, layout, and motion scores stable.",
            "- Use 512, 768, and 1024 as the first generation-ablation targets unless PCA clearly supports a smaller value.",
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    prompts = load_prompts(args)
    text_encoder = resolve_text_encoder(args.text_encoder)
    backend, tokenizer_or_encoder, model = load_text_model(text_encoder, args)
    prompt_matrix, token_matrix, embedding_stats = encode_prompts(
        prompts, backend, tokenizer_or_encoder, model, args
    )

    result = {
        "encoder_backend": backend,
        "text_encoder": text_encoder,
        "embedding_stats": embedding_stats,
        "prompt_pca": pca_stats(prompt_matrix, args.dims, args.thresholds),
        "token_pca": pca_stats(token_matrix, args.dims, args.thresholds),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    if "dims" in result["prompt_pca"]:
        write_csv(args.output_dir / "prompt_pca_dims.csv", result["prompt_pca"]["dims"])
    if "dims" in result["token_pca"]:
        write_csv(args.output_dir / "token_pca_dims.csv", result["token_pca"]["dims"])
    with (args.output_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"prompt": prompt}, ensure_ascii=True))
            handle.write("\n")
    write_report(args.output_dir / "report.md", result)
    print(f"[wan-text-space] wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
