# CLIP Video EEG Baselines

This pipeline compares the existing Wan PCA condition against native fixed
CLIP conditions used by video diffusion models. AnimateDiff uses `[77,768]`
and ZeroScope uses `[77,1024]`. Neither path uses PCA or predicts token length.
Use `bfloat16` for AnimateDiff on supported NVIDIA GPUs when the FP16
MotionAdapter path produces low-contrast dark frames. Confirm this with a
matched prompt and seed before changing the experiment-wide dtype.

## 1. Download Models Directly To Local Storage

Run in Windows PowerShell. The downloader writes directly into the selected
model directory and supports resuming partial downloads.

```powershell
cd F:\2026Spring\BCMI\Multi-Subjects-Visual-Retruction\diffusion_models_evaluation

powershell -ExecutionPolicy Bypass -File scripts\download_clip_video_models.ps1 `
  -ModelsRoot .ms_video_models `
  -HfEndpoint https://hf-mirror.com
```

Expected directories:

```text
.ms_video_models/AnimateDiff/sd-v1-5
.ms_video_models/AnimateDiff/motion-adapter-v1-5-2
.ms_video_models/ZeroScope/zeroscope_v2_576w
```

Use `-SkipAnimateDiff` or `-SkipZeroScope` when only one backend is needed.
The same script can be run against the server's `T:` repository location.
The SD 1.5 files here are only the frozen backbone required by AnimateDiff;
standalone image generation is not included as an evaluation baseline.

## 2. WSL Environment

```bash
conda create -n clip-video python=3.10 -y
conda activate clip-video

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-clip-video.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Point `MS_MODELS_ROOT` at the Windows-mounted model directory rather than
copying model weights into the WSL filesystem.

## 3. Export Native Fixed CLIP Targets

```bash
cd ~/workspace/diffusion_models_evaluation
conda activate clip-video

export FOLD=video_6fold_1
export STRUCTURED_MANIFEST=data/manifests/structured_v2_video_manifest.jsonl
export CLIP_ROOT=outputs/eeg_clip_video
export ANIMATEDIFF_TARGET_ROOT="$CLIP_ROOT/animatediff/targets_pipeline_bf16"
export ANIMATEDIFF_TARGETS="$CLIP_ROOT/animatediff/condition_targets_pipeline_bf16.jsonl"

python scripts/export_clip_video_targets.py \
  --manifest "$STRUCTURED_MANIFEST" \
  --backend animatediff \
  --model-root "$MS_MODELS_ROOT/AnimateDiff/sd-v1-5" \
  --motion-adapter "$MS_MODELS_ROOT/AnimateDiff/motion-adapter-v1-5-2" \
  --dtype bfloat16 --device cuda \
  --output-dir "$ANIMATEDIFF_TARGET_ROOT" \
  --overwrite

python scripts/build_eeg_wan_targets.py \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --latent-index "$ANIMATEDIFF_TARGET_ROOT/index.jsonl" \
  --output "$ANIMATEDIFF_TARGETS" \
  --overwrite
```

The exporter loads the complete Diffusers pipeline and calls its public
`encode_prompt()` method. It does not instantiate a separate CLIP encoder and
does not perform PCA. The older `animatediff/targets` directory is retained
only for comparison and must not be used for new EEG training.

For ZeroScope, change the backend/model/output paths:

```bash
python scripts/export_clip_video_targets.py \
  --manifest "$STRUCTURED_MANIFEST" \
  --backend zeroscope \
  --model-root "$MS_MODELS_ROOT/ZeroScope/zeroscope_v2_576w" \
  --output-dir "$CLIP_ROOT/zeroscope/targets" \
  --overwrite

python scripts/build_eeg_wan_targets.py \
  --video-manifest "$STRUCTURED_MANIFEST" \
  --latent-index "$CLIP_ROOT/zeroscope/targets/index.jsonl" \
  --output "$CLIP_ROOT/zeroscope/condition_targets.jsonl" \
  --overwrite
```

## 4. Prompt-Embedding Injection Checks

First verify native text and injection of the exact same in-memory tensor in
one pipeline instance. This isolates the public `prompt_embeds` interface from
serialization, a second text encoder, PCA, and EEG.

```bash
python scripts/validate_clip_video_condition_injection.py \
  --backend animatediff \
  --model-root "$MS_MODELS_ROOT/AnimateDiff/sd-v1-5" \
  --motion-adapter "$MS_MODELS_ROOT/AnimateDiff/motion-adapter-v1-5-2" \
  --prompt "A person kicks a ball." \
  --negative-prompt "" \
  --output-dir "$CLIP_ROOT/animatediff/injection_gate_bf16" \
  --name 01-001 \
  --dtype bfloat16 \
  --height 512 --width 512 --num-frames 16 --fps 8 \
  --steps 25 --guidance-scale 7.5 --seed 0 --enable-tf32

cat "$CLIP_ROOT/animatediff/injection_gate_bf16/01-001_report.json"
```

`pixel_mae` and `pixel_rmse` should be zero or negligible. Only after this
passes should serialized targets be checked.

Before EEG training, generate native-text and exact-target videos with the
same seed. The two outputs should be close; otherwise stop and fix condition
injection first.

```bash
export VIDEO_ID=02-040
export TARGET_PATH=$(python - "$VIDEO_ID" <<'PY'
import json, sys
video_id = sys.argv[1]
for line in open("outputs/eeg_clip_video/animatediff/condition_targets_pipeline_bf16.jsonl", encoding="utf-8"):
    row = json.loads(line)
    if row["video_id"] == video_id:
        print(row["latent_path"])
        break
PY
)

python scripts/adapters/clip_video_generate.py \
  --backend animatediff \
  --model-root "$MS_MODELS_ROOT/AnimateDiff/sd-v1-5" \
  --motion-adapter "$MS_MODELS_ROOT/AnimateDiff/motion-adapter-v1-5-2" \
  --condition "$TARGET_PATH" \
  --output "outputs/eeg_clip_video/animatediff/exact_${VIDEO_ID}.mp4" \
  --dtype bfloat16 \
  --num-frames 32 --fps 8 --steps 25 --seed 0 --enable-tf32
```

## 5. Four-Second EEG Training With Early Stopping

Use one independent model per subject. The v2 encoder keeps all 800 EEG
samples, adds four one-second theta/alpha/beta/gamma feature tokens, and keeps
the three sessions of each video together for multi-positive contrastive loss.

```bash
export SPLIT_PLAN=outputs/eeg_wan_structured_v2/splits/chentianlin_video_6fold_plan.json

for SUBJECT in chentianlin duzhuoxuan; do
  python scripts/train_eeg_wan_conditioner.py \
    --trials "data/manifests/$SUBJECT/eeg_trials.csv" \
    --targets "$ANIMATEDIFF_TARGETS" \
    --split-plan "$SPLIT_PLAN" \
    --experiment "$FOLD" \
    --duration-sec 4 \
    --slots 77 --latent-dim 768 --min-tokens 77 --max-tokens 77 \
    --architecture multiscale --sample-points 800 --sampling-rate 200 \
    --target-centering \
    --group-sessions --batch-size 12 \
    --hidden-dim 256 --encoder-layers 2 --decoder-layers 2 \
    --length-weight 0 --pooled-weight 0.2 \
    --contrastive-weight 0.1 --contrastive-temperature 0.07 \
    --selection-metric pooled_cosine_loss \
    --epochs 40 --min-epochs 10 \
    --early-stop-patience 6 --early-stop-min-delta 0.001 \
    --workers 0 \
    --output-dir "$CLIP_ROOT/animatediff/centered/$SUBJECT/$FOLD"
done
```

`--target-centering` stores the mean condition of unique training videos in
the checkpoint and trains the network to predict only an EEG-specific
residual. Epoch zero is the exact train-mean baseline and is retained as
`best.pt` unless a later epoch improves the selected validation metric.
Prediction and ranking scripts apply the stored mean automatically and report
improvement over this baseline.

If centered regression does not beat epoch zero, stop video generation and run
the six-class diagnostic first. It uses the same held-out video fold and only
the 4-second categories 01--06; chance accuracy is 16.67%.

```bash
python scripts/train_eeg_category_probe.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --split-plan outputs/eeg_wan_structured_v2/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 --duration-sec 4 \
  --epochs 30 --min-epochs 8 --early-stop-patience 6 \
  --output-dir outputs/eeg_clip_video/category_probe/chentianlin/video_6fold_1
```

The resulting `report.json` contains validation/test top-1 accuracy, macro
accuracy, per-class recall, and the confusion matrix. Accuracy near chance
means the EEG pipeline must be fixed before another text-space regression run.

ZeroScope uses the same command with `--latent-dim 1024` and its own target and
output paths. Start ZeroScope only after the AnimateDiff exact-target and EEG
pilot pass.

## 6. Matched Generation Controls

```bash
python scripts/run_clip_video_condition_controls.py \
  --backend animatediff \
  --model-root "$MS_MODELS_ROOT/AnimateDiff/sd-v1-5" \
  --motion-adapter "$MS_MODELS_ROOT/AnimateDiff/motion-adapter-v1-5-2" \
  --checkpoint "$CLIP_ROOT/animatediff/chentianlin/$FOLD/best.pt" \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets "$ANIMATEDIFF_TARGETS" \
  --manifest "$STRUCTURED_MANIFEST" \
  --video-id 02-040 --session session3 \
  --shuffled-video-id 01-041 --shuffled-session session3 \
  --duration-sec 4 \
  --output-dir "$CLIP_ROOT/animatediff/controls/02-040" \
  --dtype bfloat16 \
  --num-frames 32 --fps 8 --steps 25 --seed 0 \
  --enable-tf32 --skip-existing
```

The output set contains native text, exact target, correct EEG, shuffled EEG,
and zero-condition videos under an identical generation configuration.
