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
main actions, and relations while removing incidental visual detail. This is
especially strict for the three-entity `07` (person, dog, ball) and `08`
(person, bird, flowers) categories.

```bash
export DEEPSEEK_API_KEY='set-this-in-your-shell-only'
export DEEPSEEK_MODEL=deepseek-v4-flash

python scripts/rewrite_eeg_captions_deepseek.py \
  --input data/manifests/video_manifest.jsonl \
  --output data/manifests/structured_v2_video_manifest.jsonl
```

Inspect generated `caption`, `source_caption`, `caption_entities`, and
`caption_relations` before generating Wan targets. The script rejects a result
that omits a category's mandatory entities and writes no API key to disk.

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
