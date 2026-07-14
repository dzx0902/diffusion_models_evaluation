# Wan2.2 Text-Space Reduction Check

This note describes a practical way to test whether Wan2.2 prompt embeddings can be
projected to a lower-dimensional space for alignment with other modalities.

## Goal

Wan2.2 TI2V uses a large UMT5 text encoder. The raw text hidden size is 4096, and
the sequence length is usually capped at 512 tokens. Short prompts do not make the
4096-dimensional activation vector sparse in a strict zero-valued sense, but they
often occupy a lower-dimensional semantic subspace.

The first check is therefore:

- token padding sparsity: how much of the 512-token context is unused;
- prompt-level intrinsic dimension: PCA on mean-pooled prompt embeddings;
- token-level intrinsic dimension: PCA on all valid token embeddings;
- candidate projection sizes: 512, 768, 1024, and 2048.

## Run

Generate a larger prompt corpus first if the benchmark prompts are too few:

```bash
python scripts/generate_prompt_corpus.py \
  --count 1200 \
  --seed 20260714 \
  --output-jsonl outputs/text_space/prompt_corpus/wan_prompt_corpus.jsonl \
  --output-txt outputs/text_space/prompt_corpus/wan_prompt_corpus.txt
```

Use the Wan environment because it already has torch and transformers installed:

```bash
export MS_BENCHMARK_ROOT="$HOME/workspace/diffusion_models_evaluation"
export MS_MODELS_ROOT="$MS_BENCHMARK_ROOT/.ms_video_models"
cd "$MS_BENCHMARK_ROOT"

conda activate wan22
python scripts/analyze_wan_text_space.py \
  --text-encoder google/umt5-xxl \
  --tasks configs/ms_eval_tasks.yaml \
  --prompt-file outputs/text_space/prompt_corpus/wan_prompt_corpus.jsonl \
  --manifest outputs/ms_eval/metrics/generation_manifest.jsonl \
  --batch-size 1 \
  --device cuda \
  --dtype bf16 \
  --output-dir outputs/text_space/wan2_2_ti2v_5b
```

If the UMT5 encoder is available as a local Diffusers-style directory, pass it
instead:

```bash
python scripts/analyze_wan_text_space.py \
  --text-encoder "$MS_MODELS_ROOT/Wan2.2-TI2V-5B-Diffusers/text_encoder" \
  --local-files-only
```

If GPU memory is tight, use CPU for the text encoder:

```bash
python scripts/analyze_wan_text_space.py --device cpu --dtype fp32
```

## Outputs

The script writes:

```text
outputs/text_space/wan2_2_ti2v_5b/report.md
outputs/text_space/wan2_2_ti2v_5b/summary.json
outputs/text_space/wan2_2_ti2v_5b/prompt_pca_dims.csv
outputs/text_space/wan2_2_ti2v_5b/token_pca_dims.csv
outputs/text_space/wan2_2_ti2v_5b/prompts.jsonl
```

Read `report.md` first.

## Interpretation

Prompt-level PCA answers:

> Can a whole-prompt representation be compressed for retrieval/alignment?

Token-level PCA answers:

> Can the vectors consumed by cross-attention be compressed without obviously
> losing much variance?

Useful rules of thumb:

- 1024 dims is the conservative first target.
- 768 dims is a strong candidate for cross-modal alignment.
- 512 dims is attractive if token-level explained variance remains high.
- 256 dims is risky unless prompts are simple and generation ablation passes.

For a dimension `k`, the PCA table reports:

```text
explained_variance
reconstruction_mse_fraction = 1 - explained_variance
```

If token-level PCA at 512 dims explains little variance, do not compress
cross-attention tokens to 512 directly. Prompt-level alignment may still work.

## Required Functional Test

PCA is only a screening test. Generation must be tested because low-variance
directions can still control count, layout, or motion.

The recommended ablation is:

1. Encode prompts with the original text encoder.
2. Project hidden states from 4096 to `k`.
3. Reconstruct back to 4096.
4. Feed reconstructed prompt embeddings into Wan generation.
5. Re-run the benchmark on the same tasks/seeds.

Test these dimensions first:

```text
k = 512, 768, 1024
```

Accept a dimension only if:

- MS-VGS drop is less than about 5 percent relative;
- subject presence/count does not collapse;
- spatial relation and motion do not degrade disproportionately;
- manual review confirms the same main subjects and layout.

## Landing Plan

Phase 1: Measurement

- Run `scripts/analyze_wan_text_space.py` on benchmark prompts and real user prompts.
- Choose candidate dimensions using token-level and prompt-level PCA.

Phase 2: Offline Alignment

- Train or fit a projection from 4096 to `k`.
- For simple analysis use PCA.
- For learned alignment use a small projector such as `Linear(4096, k)` or
  `Linear(4096, k) + LayerNorm`.

Phase 3: Generation Ablation

- Add a Wan adapter path that accepts precomputed `encoder_hidden_states`.
- Compare original embeddings against projected/reconstructed embeddings.
- Run the existing benchmark at `seeds 0 1 2`.

Phase 4: Deployment

- If only retrieval/alignment needs low-dimensional vectors, store `k`-dim
  prompt-level vectors and leave generation untouched.
- If generation should consume compressed text states, add a trained
  decompressor `k -> 4096` and keep the Wan transformer interface unchanged.
