"""Cache native Wan T5 conditioning states for fixed-latent training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache variable-length Wan T5 states as CPU tensors.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, default=None)
    parser.add_argument("--wan-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fixed_latent" / "cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_prompts(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        prompts = [str(row.get("prompt", "")).strip() for row in read_jsonl(path)]
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(prompt for prompt in prompts if prompt))


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    models_root = Path(os.environ.get("MS_MODELS_ROOT", ""))
    repo = args.wan_repo or models_root / "Wan2.2"
    checkpoint_dir = args.wan_checkpoint_dir or repo / "Wan2.2-TI2V-5B"
    checkpoint = checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = checkpoint_dir / "google" / "umt5-xxl"
    for path in (repo, checkpoint, tokenizer):
        if not path.exists():
            raise FileNotFoundError(path)
    return repo, checkpoint, tokenizer


def main() -> None:
    args = parse_args()
    prompts = read_prompts(args.prompt_file)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    repo, checkpoint, tokenizer = resolve_paths(args)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from wan.modules.t5 import T5EncoderModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = args.output_dir / "states"
    states_dir.mkdir(exist_ok=True)
    encoder = T5EncoderModel(
        text_len=args.max_length,
        dtype=torch.bfloat16,
        device=torch.device(args.device),
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
    )
    index_path = args.output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as index:
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                state_path = states_dir / f"{idx:06d}.pt"
                if not (args.skip_existing and state_path.exists()):
                    context = encoder([prompt], torch.device(args.device))[0]
                    torch.save(context.detach().cpu().to(torch.float16), state_path)
                index.write(
                    json.dumps(
                        {"id": idx, "prompt": prompt, "tokens": int(torch.load(state_path).shape[0]), "path": str(state_path)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if (idx + 1) % 25 == 0 or idx + 1 == len(prompts):
                    print(f"[fixed-latent-cache] {idx + 1}/{len(prompts)}", flush=True)
    print(f"[fixed-latent-cache] wrote {index_path}")


if __name__ == "__main__":
    main()
