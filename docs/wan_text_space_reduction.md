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
  --encoder-backend wan \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --wan-checkpoint-dir "$MS_MODELS_ROOT/Wan2.2/Wan2.2-TI2V-5B" \
  --tasks configs/ms_eval_tasks.yaml \
  --prompt-file outputs/text_space/prompt_corpus/wan_prompt_corpus.jsonl \
  --manifest outputs/ms_eval/metrics/generation_manifest.jsonl \
  --batch-size 1 \
  --device cuda \
  --dtype bf16 \
  --save-token-pca \
  --pca-max-dim 2048 \
  --output-dir outputs/text_space/wan2_2_ti2v_5b
```

The native Wan backend reuses these files from the original Wan checkpoint:

```text
Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json
Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model
```

Alternatively, use a HuggingFace-format text encoder:

```bash
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

## Generation Ablation With PCA Reconstruction

### 48GB GPU high-throughput mode

For a 48GB RTX 4090D, keep the text encoder and both Wan TI2V DiT submodels resident
on the GPU. This avoids the CPU/GPU model transfers used by the conservative low-VRAM
configuration. `run_wan_pca_ablation.py` enables this mode by default and passes
`--offload_model False`; it also enables TF32 only for remaining FP32 operations.
The batch runner defaults to `--frame-num 65`, which exactly matches the benchmark's
4-second, 16-fps task duration and is about 20% less denoising work than 81 frames.
It keeps Wan's configured sampling-step count unless `--sample-steps` is explicitly set.

Before committing to a long run, test one representative video and monitor peak VRAM:

```bash
conda activate wan22
cd "$MS_BENCHMARK_ROOT"

python scripts/run_wan_pca_ablation.py \
  --task-ids dog_car_walk_static \
  --seeds 0 \
  --variants baseline \
  --output-root outputs/wan_high_vram_smoke \
  --generate-only

nvidia-smi
```

Keep the default `--gpu-resident-models` only if this finishes without OOM. For the
previous low-VRAM behavior, add `--no-gpu-resident-models --no-enable-tf32`.

For the complete generation-and-YOLO run, launch the controller from `wan22` and
explicitly point evaluation to the separate benchmark environment:

```bash
conda activate wan22
cd "$MS_BENCHMARK_ROOT"
# Set this once per shell. Do not leave EVAL_PY unset: an empty value resolves to '.'.
export EVAL_PY=/home/dzxy/miniconda3/envs/ms-video-eval/bin/python
"$EVAL_PY" --version

python scripts/run_wan_pca_ablation.py \
  --preset full \
  --output-root outputs/wan_text_compression_full \
  --eval-python "$EVAL_PY" \
  --settings "$MS_BENCHMARK_ROOT/configs/ms_eval_settings.wsl.yaml" \
  --report-error \
  --skip-existing
```

After running the analysis command with `--save-token-pca`, use the saved
projector:

```text
outputs/text_space/wan2_2_ti2v_5b/token_pca_projector.npz
```

The wrapper below runs Wan's original `generate.py`, but patches the native T5
encoder output at runtime:

```text
4096 -> k -> 4096
```

Run a baseline through the same wrapper:

```bash
conda activate wan22
cd "$MS_BENCHMARK_ROOT"

python scripts/adapters/wan_projected_generate.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --projector outputs/text_space/wan2_2_ti2v_5b/token_pca_projector.npz \
  --project-dim 1024 \
  --disable-projection \
  -- \
  --task ti2v-5B \
  --size "1280*704" \
  --ckpt_dir "$MS_MODELS_ROOT/Wan2.2/Wan2.2-TI2V-5B" \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --base_seed 0 \
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." \
  --save_file "$MS_MODELS_ROOT/Wan2.2/outputs/pca_baseline_seed0.mp4"
```

Run a projected version:

```bash
python scripts/adapters/wan_projected_generate.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --projector outputs/text_space/wan2_2_ti2v_5b/token_pca_projector.npz \
  --project-dim 1024 \
  --report-error \
  -- \
  --task ti2v-5B \
  --size "1280*704" \
  --ckpt_dir "$MS_MODELS_ROOT/Wan2.2/Wan2.2-TI2V-5B" \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --base_seed 0 \
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." \
  --save_file "$MS_MODELS_ROOT/Wan2.2/outputs/pca_1024_seed0.mp4"
```

Recommended first pass:

```text
k = 1536, 1024, 768
```

Only test `512` after `768` is acceptable, because token-level PCA retained
about 91 percent variance at 512 in the initial run.

To benchmark projected Wan outputs, add a temporary model entry whose
`command_template` calls `scripts/adapters/wan_projected_generate.py` with the
desired `--project-dim`. Keep the original Wan entry enabled as the baseline and
compare MS-VGS over the same tasks/seeds.

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
