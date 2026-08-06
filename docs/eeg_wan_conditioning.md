# EEG To Wan Conditioning

This workflow learns raw EEG to the validated fixed Wan text target: PCA-padded
`[128, 512]`. It uses only raw `session1`, `session2`, and `session3` data;
`session_average` is excluded because it aggregates all sessions.

## 1. Build The Subject Manifest

```bash
python scripts/build_eeg_video_manifest.py --subject chentianlin --overwrite
```

This produces ignored local files under `data/manifests/`. The video manifest
has 624 unique video records; the trial CSV has 1,872 EEG trials. The script
validates all caption, video, metadata, mask, length, and `video_id` joins.

## 2. Build The Six-Fold Plan

```bash
python scripts/build_eeg_split_plans.py \
  --video-manifest data/manifests/video_manifest.jsonl \
  --subject chentianlin \
  --output-dir outputs/eeg_wan/splits \
  --overwrite
```

Every fold contains 416 train, 104 validation, and 104 test videos. The plan
also exports caption JSONL files per partition under
`outputs/eeg_wan/splits/chentianlin_video_6fold_captions/`.

### Optional: Concise Caption Targets

The original captions have substantially different levels of detail across the
eight categories. In particular, categories `03`, `04`, `07`, and `08` contain
long scene, appearance, and multi-step descriptions that are absent from the
shorter categories. For an EEG semantic-conditioning experiment, create a
separate concise manifest instead of overwriting the source captions:

```bash
python scripts/build_simple_eeg_caption_manifest.py \
  --input data/manifests/video_manifest.jsonl \
  --output data/manifests/simple_v1_video_manifest.jsonl

python scripts/build_eeg_split_plans.py \
  --video-manifest data/manifests/simple_v1_video_manifest.jsonl \
  --subject chentianlin \
  --output-dir outputs/eeg_wan_simple_v1/splits
```

The concise scheme preserves the category subjects and their principal action
or relation, for example `A person throws a ball and a dog chases it.` It
removes identity, clothing, colour, weather, and background detail. The output
manifest keeps every original annotation in `source_caption` for traceability.
Use the concise manifest consistently for text-state caching, PCA fitting,
target export, and EEG training. Do not mix it with the original caption cache.

For the recommended structured simplification, use
`rewrite_eeg_captions_deepseek.py` with an API key stored only in the shell
environment. It asks a language model to retain all central entities, counts,
main actions, and relations while removing incidental visual detail. For
three-entity categories it preserves source-faithful connected interactions or
independent simultaneous actions, without inventing a connection.

```bash
export DEEPSEEK_API_KEY='set-this-in-your-shell-only'
export DEEPSEEK_MODEL=deepseek-v4-flash

python scripts/rewrite_eeg_captions_deepseek.py \
  --input data/manifests/video_manifest.jsonl \
  --output data/manifests/structured_v2_video_manifest.jsonl \
  --batch-size 8 \
  --thinking disabled
```

Inspect generated `caption`, `source_caption`, `caption_entities`, and
`caption_relations` before generating Wan targets. The script rejects a result
that omits a category's mandatory entities and writes no API key to disk. Each
batch is validated by `video_id`; use `--resume --batch-size 8` after an
interruption to process failed and not-yet-written records without repeating
successful ones.

### Structured V2 Pilot Training

`data/manifests/structured_v2_video_manifest.jsonl` is the checked-in,
completed 624-caption manifest. Keep its entire output tree separate from the
original-caption experiment so checkpoints and PCA projectors cannot mix.
Run the text steps once in `wan22`; the cache is shared by every subject, while
the PCA projector is fitted only to the selected fold's train captions.

```bash
STRUCTURED_MANIFEST=data/manifests/structured_v2_video_manifest.jsonl
RUN_ROOT=outputs/eeg_wan_structured_v2
FOLD=video_6fold_1

for SUBJECT in chentianlin duzhuoxuan; do
  python scripts/build_eeg_split_plans.py \
    --video-manifest "$STRUCTURED_MANIFEST" \
    --subject "$SUBJECT" \
    --output-dir "$RUN_ROOT/splits" \
    --overwrite
done

python scripts/cache_wan_text_states.py \
  --prompt-file "$STRUCTURED_MANIFEST" \
  --output-dir "$RUN_ROOT/text_cache"

python scripts/analyze_wan_text_space.py \
  --prompt-file "$RUN_ROOT/splits/chentianlin_video_6fold_captions/${FOLD}_train.jsonl" \
  --only-prompt-file --encoder-backend wan --save-token-pca --pca-max-dim 512 \
  --output-dir "$RUN_ROOT/$FOLD/text_space"

python scripts/export_wan_fixed_pca_latents.py \
  --cache-dir "$RUN_ROOT/text_cache" \
  --projector "$RUN_ROOT/$FOLD/text_space/token_pca_projector.npz" \
  --slots 128 --dim 512 \
  --output-dir "$RUN_ROOT/$FOLD/pca_128x512"

python scripts/build_eeg_wan_targets.py \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --latent-index "$RUN_ROOT/$FOLD/pca_128x512/index.jsonl" \
  --output "$RUN_ROOT/$FOLD/wan_targets.jsonl" \
  --overwrite
```

Train each subject separately. Use a new directory and 80 epochs; previous
original-caption checkpoints cannot be resumed because their target space is
different. Derive the classifier bounds from the new target set, rather than
reusing token bounds from an earlier caption scheme.

### Validate the PCA Decoder Before More EEG Video Generation

The weak `condition-source target` control means that the PCA text-space
round-trip must be measured separately from EEG prediction.  Run this once for
the current fold. It fits a larger projector only on the 416 train captions,
then evaluates reconstruction on the held-out 104 test captions. It does not
generate videos.

```bash
PCA_ROOT="$RUN_ROOT/$FOLD/pca_decoder_validation"

python scripts/analyze_wan_text_space.py \
  --prompt-file "$RUN_ROOT/splits/chentianlin_video_6fold_captions/${FOLD}_train.jsonl" \
  --only-prompt-file --encoder-backend wan \
  --save-token-pca --pca-max-dim 1536 \
  --dims 512 768 1024 1536 \
  --output-dir "$PCA_ROOT/text_space"

python scripts/evaluate_wan_pca_holdout.py \
  --cache-dir "$RUN_ROOT/text_cache" \
  --projector "$PCA_ROOT/text_space/token_pca_projector.npz" \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --split-plan "$RUN_ROOT/splits/chentianlin_video_6fold_plan.json" \
  --experiment "$FOLD" --partition test \
  --dims 512 768 1024 1536 \
  --output-dir "$PCA_ROOT/heldout_test"

cat "$PCA_ROOT/heldout_test/report.md"
```

`per_video.csv` exposes the difficult captions by dimension, while
`summary.csv` reports the overall reconstruction. This is a true held-out
check: the held-out captions never participate in the PCA fit. It evaluates
the deterministic PCA inverse only, so an error here cannot be attributed to
EEG.

Next, generate the native text control plus exact PCA round trips for only the
three previously selected held-out videos. This script does not load an EEG
checkpoint. Every variant uses the identical prompt, seed, Wan checkpoint, and
sampling configuration.

```bash
CONTROL_ROOT="$PCA_ROOT/video_controls"

python scripts/run_wan_pca_decoder_controls.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --projector "$PCA_ROOT/text_space/token_pca_projector.npz" \
  --manifest "$STRUCTURED_MANIFEST" \
  --video-ids 02-040 06-069 07-031 \
  --dims 512 768 1024 1536 \
  --output-dir "$CONTROL_ROOT" \
  --size "1280*704" --seed 0 --offload-model True \
  --enable-tf32 --skip-existing

conda activate ms-video-eval
python scripts/score_eeg_wan_probe_videos.py \
  --videos "$CONTROL_ROOT"/*.mp4 \
  --manifest "$STRUCTURED_MANIFEST" \
  --settings configs/ms_eval_settings.wsl.yaml \
  --output-dir "$CONTROL_ROOT/yolo" \
  --sample-every 4
cat "$CONTROL_ROOT/yolo/report.md"
```

Expected output names are `02-040_native_text_seed0.mp4` and
`02-040_pca_512_seed0.mp4` (and so on). Do not use the old 512-dimensional
EEG checkpoint with the new 768/1024/1536 projector: these video controls are
PCA-only. Select the smallest dimension whose generated object score and
manual semantics stay close to `native_text`; only then rebuild the fixed PCA
targets and retrain the EEG conditioner for that dimension.

### Learned Fixed Wan Condition Space

PCA is a linear baseline, not the final target space. The learned alternative
keeps the same fixed-slot interface but learns a contextual sequence encoder
and decoder:

```text
Wan T5 state H [N,4096] -> pad to [128,4096] -> encoder E -> Z [128,k]
Z [128,k] -> frozen decoder D -> H_hat [N,4096] -> Wan
```

The first implementation deliberately compresses only feature dimension; it
does not also compress token slots. Train `E/D` only on the fold-train captions
and select it using fold-validation captions. The test partition is used only
after the checkpoint is frozen. Start with `k=1024`; do not train EEG until the
exact-latent Wan controls are acceptable.

The structured captions can contain duplicate strings for distinct videos. The
default trainer rejects duplicates spanning train and validation because that
is not a caption-held-out validation. For a *codec feasibility pretraining*
pilot only, append `--allow-prompt-overlap`; the log will warn explicitly.
This is acceptable for answering whether `E/D` can decode the known condition
distribution, but its validation score must not be reported as strict unseen-
caption generalization. A video-level reconstruction objective also requires
more distinctive captions than the current simplified duplicate groups.

```bash
AE_ROOT="$RUN_ROOT/$FOLD/wan_condition_ae_k1024"
TRAIN_CAPTIONS="$RUN_ROOT/splits/chentianlin_video_6fold_captions/${FOLD}_train.jsonl"
VALID_CAPTIONS="$RUN_ROOT/splits/chentianlin_video_6fold_captions/${FOLD}_validation.jsonl"

python scripts/train_wan_condition_autoencoder.py \
  --cache-dir "$RUN_ROOT/text_cache" \
  --train-prompts "$TRAIN_CAPTIONS" \
  --validation-prompts "$VALID_CAPTIONS" \
  --output-dir "$AE_ROOT" \
  --slots 128 --latent-dim 1024 --hidden-dim 1024 \
  --encoder-layers 2 --decoder-layers 2 --heads 8 \
  --epochs 100 --batch-size 4 --workers 0

python scripts/evaluate_wan_condition_autoencoder.py \
  --cache-dir "$RUN_ROOT/text_cache" \
  --checkpoint "$AE_ROOT/best.pt" \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --split-plan "$RUN_ROOT/splits/chentianlin_video_6fold_plan.json" \
  --experiment "$FOLD" --partition test \
  --output-dir "$AE_ROOT/heldout_test"

cat "$AE_ROOT/heldout_test/report.md"
```

Validate the frozen decoder before any EEG training. The following creates only
two exact-latent videos; `--skip-native` avoids regenerating existing native
controls. Compare the result with the corresponding previously generated
native-text video, then score with the same YOLO utility.

```bash
AE_CONTROL_ROOT="$AE_ROOT/video_controls"

python scripts/run_wan_autoencoder_controls.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --autoencoder-checkpoint "$AE_ROOT/best.pt" \
  --cache-dir "$RUN_ROOT/text_cache" \
  --manifest "$STRUCTURED_MANIFEST" \
  --video-ids 02-040 07-031 \
  --output-dir "$AE_CONTROL_ROOT" \
  --size "1280*704" --seed 0 --offload-model True \
  --enable-tf32 --skip-native --skip-existing

conda activate ms-video-eval
python scripts/score_eeg_wan_probe_videos.py \
  --videos "$AE_CONTROL_ROOT"/*.mp4 \
  --manifest "$STRUCTURED_MANIFEST" \
  --settings configs/ms_eval_settings.wsl.yaml \
  --output-dir "$AE_CONTROL_ROOT/yolo" --sample-every 4
cat "$AE_CONTROL_ROOT/yolo/report.md"
```

Only if these exact-latent controls preserve native semantics, export the
frozen autoencoder targets and train a fresh EEG predictor. The existing
512-dimensional PCA checkpoint is incompatible with this coordinate space.

```bash
conda activate wan22

python scripts/export_wan_autoencoder_latents.py \
  --cache-dir "$RUN_ROOT/text_cache" \
  --checkpoint "$AE_ROOT/best.pt" \
  --output-dir "$AE_ROOT/latents"

python scripts/build_eeg_wan_targets.py \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --latent-index "$AE_ROOT/latents/index.jsonl" \
  --output "$AE_ROOT/wan_targets.jsonl" --overwrite

read MIN_TOKENS MAX_TOKENS < <(
  python - "$AE_ROOT/wan_targets.jsonl" <<'PY'
import json
import sys
values = [json.loads(line)["tokens"] for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(min(values), max(values))
PY
)

python scripts/train_eeg_wan_conditioner.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets "$AE_ROOT/wan_targets.jsonl" \
  --split-plan "$RUN_ROOT/splits/chentianlin_video_6fold_plan.json" \
  --experiment "$FOLD" \
  --output-dir "$AE_ROOT/eeg_chentianlin" \
  --epochs 80 --batch-size 8 --hidden-dim 128 \
  --encoder-layers 1 --decoder-layers 1 --workers 0 \
  --min-tokens "$MIN_TOKENS" --max-tokens "$MAX_TOKENS"
```

Use `scripts/adapters/wan_eeg_generate.py` with
`--autoencoder-checkpoint "$AE_ROOT/best.pt"` instead of `--projector` for
the final EEG-to-Wan probe. A Wan-aware frozen-DiT loss is intentionally not
part of this first stage: add it only when the autoencoder's exact-latent
controls fail despite good held-out reconstruction metrics.

### Validate Whether the PCA Basis Generalizes

The PCA inverse itself is deterministic. The remaining question is whether a
projector fitted to 416 fold-train captions (`W_fold`) is too narrow for test
caption semantics. Generate a short-caption calibration corpus that excludes
all 624 experiment captions, then fit a fixed global basis (`W_global`). This
is valid at deployment because no test prompt is used to fit either basis.

```bash
GLOBAL_ROOT="$PCA_ROOT/global_basis"

python scripts/generate_wan_pca_calibration_corpus.py \
  --count 1800 --seed 20260805 \
  --exclude-manifest "$STRUCTURED_MANIFEST" \
  --output "$GLOBAL_ROOT/calibration_prompts.jsonl"

python scripts/analyze_wan_text_space.py \
  --prompt-file "$GLOBAL_ROOT/calibration_prompts.jsonl" \
  --only-prompt-file --encoder-backend wan \
  --save-token-pca --pca-max-dim 1536 \
  --max-token-samples 20000 \
  --dims 512 768 1024 1536 \
  --output-dir "$GLOBAL_ROOT/text_space"

python scripts/evaluate_wan_pca_holdout.py \
  --cache-dir "$RUN_ROOT/text_cache" \
  --projector "$GLOBAL_ROOT/text_space/token_pca_projector.npz" \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --split-plan "$RUN_ROOT/splits/chentianlin_video_6fold_plan.json" \
  --experiment "$FOLD" --partition test \
  --dims 512 768 1024 1536 \
  --output-dir "$GLOBAL_ROOT/heldout_test"

cat "$GLOBAL_ROOT/heldout_test/report.md"
```

Compare this report with `$PCA_ROOT/heldout_test/report.md`: lower error for
the same dimension means the global calibration basis covers test semantics
better than the fold-specific basis. Then compare `W_fold` and `W_global` in
Wan directly at the selected candidate dimension. For example, use 768 after
the 768 control has been generated:

```bash
GLOBAL_CONTROL_ROOT="$GLOBAL_ROOT/video_controls_k768"

python scripts/run_wan_pca_decoder_controls.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --projector "$GLOBAL_ROOT/text_space/token_pca_projector.npz" \
  --manifest "$STRUCTURED_MANIFEST" \
  --video-ids 02-040 07-031 \
  --dims 768 \
  --output-dir "$GLOBAL_CONTROL_ROOT" \
  --size "1280*704" --seed 0 --offload-model True \
  --enable-tf32 --skip-native --skip-existing
```

Use `W_all` only as a leakage diagnostic upper bound: it may fit all 624
captions, including held-out ones, but must never be reported as a valid test
result or used for the EEG evaluation. If `W_global` does not improve over
`W_fold` at 768/1024, the next PCA-stage replacement is a learned decoder
rather than more PCA tuning.

```bash
read MIN_TOKENS MAX_TOKENS < <(
  python - "$RUN_ROOT/$FOLD/wan_targets.jsonl" <<'PY'
import json
import sys

lengths = [json.loads(line)["tokens"] for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(min(lengths), max(lengths))
PY
)

for SUBJECT in chentianlin duzhuoxuan; do
  python scripts/train_eeg_wan_conditioner.py \
    --trials "data/manifests/$SUBJECT/eeg_trials.csv" \
    --targets "$RUN_ROOT/$FOLD/wan_targets.jsonl" \
    --split-plan "$RUN_ROOT/splits/${SUBJECT}_video_6fold_plan.json" \
    --experiment "$FOLD" \
    --output-dir "$RUN_ROOT/$SUBJECT/$FOLD" \
    --epochs 80 --batch-size 8 --hidden-dim 128 --encoder-layers 1 --decoder-layers 1 \
    --workers 0 --min-tokens "$MIN_TOKENS" --max-tokens "$MAX_TOKENS"
done
```

### EEG-To-Wan Video Probe

`wan_eeg_generate.py` uses one EEG trial to predict `[128, 512]`, applies the
fold-specific PCA inverse transform, and patches Wan's T5 output at runtime.
Use a held-out test video. `target` length is an oracle-length diagnostic that
isolates the continuous EEG latent; `predicted` length is the fully EEG-driven
setting. The prompt passed after `--` is only required by Wan's CLI and is not
encoded into the generated condition.

```bash
VIDEO_ID=01-001  # Replace with a video_id from this fold's test partition.

python scripts/adapters/wan_eeg_generate.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --checkpoint "$RUN_ROOT/chentianlin/$FOLD/last.pt" \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets "$RUN_ROOT/$FOLD/wan_targets.jsonl" \
  --projector "$RUN_ROOT/$FOLD/text_space/token_pca_projector.npz" \
  --video-id "$VIDEO_ID" --session session3 --length-source target \
  --condition-output "$RUN_ROOT/probes/chentianlin_${VIDEO_ID}_target_length.pt" \
  --enable-tf32 \
  -- \
  --task ti2v-5B --size "1280*704" \
  --ckpt_dir "$MS_MODELS_ROOT/Wan2.2/Wan2.2-TI2V-5B" \
  --offload_model True --convert_model_dtype --t5_cpu --base_seed 0 \
  --prompt "EEG-conditioned video." \
  --save_file "$RUN_ROOT/probes/chentianlin_${VIDEO_ID}_target_length.mp4"
```

Repeat the same command with `--length-source predicted` and a distinct output
name. Compare the two videos before attributing a failure to the continuous
EEG latent rather than token-length prediction.

Before generating any videos, rank all held-out trials without loading Wan:

```bash
python scripts/evaluate_eeg_wan_predictions.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets "$RUN_ROOT/$FOLD/wan_targets.jsonl" \
  --checkpoint "$RUN_ROOT/chentianlin/$FOLD/last.pt" \
  --split-plan "$RUN_ROOT/splits/chentianlin_video_6fold_plan.json" \
  --experiment "$FOLD" --partition test \
  --output-dir "$RUN_ROOT/chentianlin/$FOLD/test_ranking"
```

This writes per-session `trial_metrics.csv`, per-video `video_ranking.csv`,
and `summary.json`. Select videos using high mean pooled cosine, low MSE, and
stable results over all three sessions; then use `wan_eeg_generate.py` only for
those candidates.

For `video_6fold_1`, the complete concise-target export is:

```bash
python scripts/cache_wan_text_states.py \
  --prompt-file data/manifests/simple_v1_video_manifest.jsonl \
  --output-dir outputs/eeg_wan_simple_v1/text_cache

python scripts/analyze_wan_text_space.py \
  --prompt-file outputs/eeg_wan_simple_v1/splits/chentianlin_video_6fold_captions/video_6fold_1_train.jsonl \
  --only-prompt-file \
  --encoder-backend wan \
  --save-token-pca \
  --pca-max-dim 512 \
  --output-dir outputs/eeg_wan_simple_v1/video_6fold_1/text_space

python scripts/export_wan_fixed_pca_latents.py \
  --cache-dir outputs/eeg_wan_simple_v1/text_cache \
  --projector outputs/eeg_wan_simple_v1/video_6fold_1/text_space/token_pca_projector.npz \
  --slots 128 --dim 512 \
  --output-dir outputs/eeg_wan_simple_v1/video_6fold_1/pca_128x512

python scripts/build_eeg_wan_targets.py \
  --video-manifest data/manifests/simple_v1_video_manifest.jsonl \
  --latent-index outputs/eeg_wan_simple_v1/video_6fold_1/pca_128x512/index.jsonl \
  --output outputs/eeg_wan_simple_v1/video_6fold_1/wan_targets.jsonl
```

Then train with the original EEG trial mapping and the new target manifest:

```bash
python scripts/train_eeg_wan_conditioner.py \
  --trials data/manifests/eeg_trials.csv \
  --targets outputs/eeg_wan_simple_v1/video_6fold_1/wan_targets.jsonl \
  --split-plan outputs/eeg_wan_simple_v1/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 \
  --output-dir outputs/eeg_wan_simple_v1/chentianlin/video_6fold_1 \
  --epochs 20 --batch-size 8 --hidden-dim 128 --encoder-layers 1 --decoder-layers 1 \
  --workers 0
```

## 3. Export Frozen Wan Targets

Run these commands in the `wan22` environment. Cache native T5 states once for
all videos, but fit a separate PCA projector for every fold using that fold's
training captions only.

```bash
python scripts/cache_wan_text_states.py \
  --prompt-file data/manifests/video_manifest.jsonl \
  --output-dir outputs/eeg_wan/text_cache

python scripts/analyze_wan_text_space.py \
  --prompt-file outputs/eeg_wan/splits/chentianlin_video_6fold_captions/video_6fold_1_train.jsonl \
  --only-prompt-file \
  --encoder-backend wan \
  --save-token-pca \
  --pca-max-dim 512 \
  --output-dir outputs/eeg_wan/chentianlin_video_6fold_1/text_space

python scripts/export_wan_fixed_pca_latents.py \
  --cache-dir outputs/eeg_wan/text_cache \
  --projector outputs/eeg_wan/chentianlin_video_6fold_1/text_space/token_pca_projector.npz \
  --slots 128 --dim 512 \
  --output-dir outputs/eeg_wan/chentianlin_video_6fold_1/pca_128x512

python scripts/build_eeg_wan_targets.py \
  --video-manifest data/manifests/video_manifest.jsonl \
  --latent-index outputs/eeg_wan/chentianlin_video_6fold_1/pca_128x512/index.jsonl \
  --output outputs/eeg_wan/chentianlin_video_6fold_1/wan_targets.jsonl
```

There are currently 623 unique captions among the 624 videos. This is valid:
the cache contains one target per unique caption, while `wan_targets.jsonl`
still contains one explicit target binding per `video_id`.

## 4. Subject-Dependent Six-Fold Video Training

Run in an environment with PyTorch and NumPy. Each subject is trained separately.
Every fold holds out 13 videos/category for validation and 13 videos/category
for test. The remaining 52 videos/category train on all three raw sessions.
All three sessions of a video remain in the same partition.

```bash
python scripts/train_eeg_wan_conditioner.py \
  --trials data/manifests/eeg_trials.csv \
  --targets outputs/eeg_wan/chentianlin_video_6fold_1/wan_targets.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 \
  --output-dir outputs/eeg_wan/chentianlin_video_6fold_1
```

The checkpoint predicts `Z_hat[128,512]` and a token-length class range inferred
from the training targets. Repeat for `video_6fold_1` through `video_6fold_6`.
To obtain the unbiased test metric after selecting the best validation epoch:

```bash
python scripts/train_eeg_wan_conditioner.py \
  --trials data/manifests/eeg_trials.csv \
  --targets outputs/eeg_wan/wan_targets.jsonl \
  --split-plan outputs/eeg_wan/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 \
  --validation-partition test \
  --checkpoint outputs/eeg_wan/chentianlin_video_6fold_1/best.pt
```

## 5. Four-Second Protocol And Condition Controls

The current dataset contains both four-second and six-second trials. Pass
`--duration-sec 4` to every training and ranking command so categories 07-08
cannot enter the four-second baseline. The checkpoint records the resulting
trial/video counts and refuses to resume from a checkpoint with a different
recorded protocol.

Audit the fold before training:

```bash
RUN_ROOT=outputs/eeg_wan_structured_v2
FOLD=video_6fold_1
TRIALS=data/manifests/chentianlin/eeg_trials.csv
TARGETS="$RUN_ROOT/$FOLD/wan_targets.jsonl"
SPLIT_PLAN="$RUN_ROOT/splits/chentianlin_video_6fold_plan.json"
FOUR_S_ROOT=outputs/eeg_wan_4s_v1

python scripts/audit_eeg_wan_protocol.py \
  --trials "$TRIALS" \
  --targets "$TARGETS" \
  --split-plan "$SPLIT_PLAN" \
  --experiment "$FOLD" \
  --duration-sec 4 \
  --output-dir "$FOUR_S_ROOT/protocol/$FOLD"
```

Train a diagnostic conditioner that focuses on the continuous `[128,512]`
condition and uses oracle/fixed token length during generation. Setting the
length loss to zero prevents the weak length classifier from dominating model
selection; this is an ablation, not the final prompt-free length solution.

```bash
python scripts/train_eeg_wan_conditioner.py \
  --trials "$TRIALS" \
  --targets "$TARGETS" \
  --split-plan "$SPLIT_PLAN" \
  --experiment "$FOLD" \
  --duration-sec 4 \
  --output-dir "$FOUR_S_ROOT/chentianlin/$FOLD/regression_only" \
  --epochs 20 --batch-size 8 \
  --hidden-dim 128 --encoder-layers 1 --decoder-layers 1 \
  --padding-weight 0.1 --length-weight 0 --pooled-weight 0.1 \
  --workers 0
```

Rank only held-out four-second trials before loading Wan:

```bash
python scripts/evaluate_eeg_wan_predictions.py \
  --trials "$TRIALS" \
  --targets "$TARGETS" \
  --checkpoint "$FOUR_S_ROOT/chentianlin/$FOLD/regression_only/best.pt" \
  --split-plan "$SPLIT_PLAN" \
  --experiment "$FOLD" --partition test \
  --duration-sec 4 \
  --output-dir "$FOUR_S_ROOT/chentianlin/$FOLD/regression_only/test_ranking"
```

For one selected target, run five matched generation controls with the same
seed and oracle target length: native text, exact target, correct EEG, EEG from
a different video, and a zero condition.

```bash
python scripts/run_eeg_wan_condition_controls.py \
  --wan-repo "$MS_MODELS_ROOT/Wan2.2" \
  --checkpoint "$FOUR_S_ROOT/chentianlin/$FOLD/regression_only/best.pt" \
  --trials "$TRIALS" \
  --targets "$TARGETS" \
  --projector "$RUN_ROOT/$FOLD/pca_decoder_validation/text_space/token_pca_projector.npz" \
  --manifest data/manifests/structured_v2_video_manifest.jsonl \
  --video-id 02-040 --session session3 \
  --shuffled-video-id 01-041 --shuffled-session session3 \
  --duration-sec 4 \
  --output-dir "$FOUR_S_ROOT/controls/02-040" \
  --enable-tf32 --skip-existing
```
