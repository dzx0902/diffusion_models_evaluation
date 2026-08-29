#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Script Error"' ERR

# Reproduces the released Compact EEG-only test20 setup. All parameters can be
# overridden from the command line. The current terminal's Python is used.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/userhome2/zhoutianyi/Dataset/Multi-Object}"
SUBJECT="${SUBJECT:-zhoutianyi}"
EXPERIMENT="${EXPERIMENT:-compact_eeg_3session_fusion_test20}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

# 468 unique videos: 54/class train, 8/class val, 16/class test.
VAL_PER_CLASS="${VAL_PER_CLASS:-8}"
TEST_PER_CLASS="${TEST_PER_CLASS:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-800}"

TEMPORAL_FILTERS="${TEMPORAL_FILTERS:-16}"
SPATIAL_MULTIPLIER="${SPATIAL_MULTIPLIER:-2}"
FEATURE_DIM="${FEATURE_DIM:-128}"
DROPOUT="${DROPOUT:-0.35}"
PAIR_WEIGHT="${PAIR_WEIGHT:-0.5}"
FUSED_WEIGHT="${FUSED_WEIGHT:-1.0}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.05}"
NOISE_STD="${NOISE_STD:-0.02}"
TIME_MASK_SAMPLES="${TIME_MASK_SAMPLES:-20}"
AMP="${AMP:-1}"

bash "${PROJECT_ROOT}/scripts/prepare_labels.sh"

COMMAND=(
  python "${PROJECT_ROOT}/src/train.py"
  --subject "${SUBJECT}"
  --dataset-root "${DATASET_ROOT}"
  --label-package "${PROJECT_ROOT}/data/video_multilabels_2object.pt"
  --output-root "${PROJECT_ROOT}/outputs"
  --experiment "${EXPERIMENT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --num-samples "${NUM_SAMPLES}"
  --val-per-class "${VAL_PER_CLASS}"
  --test-per-class "${TEST_PER_CLASS}"
  --temporal-filters "${TEMPORAL_FILTERS}"
  --spatial-multiplier "${SPATIAL_MULTIPLIER}"
  --feature-dim "${FEATURE_DIM}"
  --dropout "${DROPOUT}"
  --pair-weight "${PAIR_WEIGHT}"
  --fused-weight "${FUSED_WEIGHT}"
  --consistency-weight "${CONSISTENCY_WEIGHT}"
  --noise-std "${NOISE_STD}"
  --time-mask-samples "${TIME_MASK_SAMPLES}"
)
if [[ "${AMP}" == "1" ]]; then
  COMMAND+=(--amp)
else
  COMMAND+=(--no-amp)
fi

echo "subject: ${SUBJECT}"
echo "split: train=324, validation=48, test=96"
echo "device: ${DEVICE}"
echo "python: $(command -v python)"
echo "outputs: ${PROJECT_ROOT}/outputs/${EXPERIMENT}/${SUBJECT}"
"${COMMAND[@]}"
