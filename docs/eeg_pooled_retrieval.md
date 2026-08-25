# EEG Pooled Residual Retrieval

This probe tests whether four seconds of EEG can recover discriminative text
semantics before attempting full token-condition regression.

## Representation

For each unique training caption, the fixed video model text condition is
pooled over valid tokens. The training-caption mean is subtracted and a single
global RMS scale is applied. The transform is retained in the checkpoint.

The EEG model predicts one standardized residual vector. It does not directly
replace the full diffusion cross-attention condition. Initial generation uses
the exact condition of the nearest retrieved caption, which separates EEG
semantic retrieval from token decoding quality.

## Objective

- centered residual MSE and cosine loss;
- batch-wise symmetric multi-positive InfoNCE, or full caption-bank softmax;
- same-caption and repeated-session positives;
- variance and covariance regularization against representation collapse;
- validation MRR for checkpoint selection and early stopping.

## Training

```bash
python scripts/train_eeg_pooled_retriever.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets outputs/eeg_clip_video/animatediff/condition_targets_pipeline_bf16.jsonl \
  --split-plan outputs/eeg_wan_structured_v2/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 \
  --validation-partition validation \
  --duration-sec 4 \
  --architecture baseline \
  --hidden-dim 128 --encoder-layers 1 \
  --group-sessions --batch-size 48 \
  --contrastive-bank train \
  --mse-weight 0.01 --cosine-weight 1 --contrastive-weight 2 \
  --variance-weight 1 --covariance-weight 0.001 \
  --epochs 20 --min-epochs 5 --early-stop-patience 5 \
  --auto-resume \
  --output-dir outputs/eeg_pooled_retrieval/animatediff/chentianlin/video_6fold_1
```

## Held-out test

```bash
python scripts/train_eeg_pooled_retriever.py \
  --trials data/manifests/chentianlin/eeg_trials.csv \
  --targets outputs/eeg_clip_video/animatediff/condition_targets_pipeline_bf16.jsonl \
  --split-plan outputs/eeg_wan_structured_v2/splits/chentianlin_video_6fold_plan.json \
  --experiment video_6fold_1 \
  --validation-partition test \
  --duration-sec 4 \
  --architecture baseline \
  --hidden-dim 128 --encoder-layers 1 \
  --group-sessions --batch-size 48 \
  --contrastive-bank train \
  --mse-weight 0.01 --cosine-weight 1 --contrastive-weight 2 \
  --variance-weight 1 --covariance-weight 0.001 \
  --checkpoint outputs/eeg_pooled_retrieval/animatediff/chentianlin/video_6fold_1/best.pt \
  --output-dir outputs/eeg_pooled_retrieval/animatediff/chentianlin/video_6fold_1/test
```

The test output contains `evaluation.json` and per-trial retrieval results in
`trial_metrics.csv`. Compare Recall@1 and Recall@5 with the chance values in
the same report. The `session_averaged_*` fields first average predictions for
the same video across available sessions, then run retrieval. This diagnoses
whether repeated EEG observations improve semantic signal. Do not generate
videos if retrieval remains at chance.

`--contrastive-bank train` compares every EEG prediction with every unique
caption condition in the active split. During training this is the training
caption bank; during validation or test evaluation it is the corresponding
held-out bank. The variance term also matches the bank's observed per-feature
standard deviation instead of imposing unit variance on every CLIP dimension.
