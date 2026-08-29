#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Script Error"' ERR

# Test a trained checkpoint and turn fused Top-2 EEG predictions into captions.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/userhome2/zhoutianyi/Dataset/Multi-Object}"
SUBJECT="${SUBJECT:-zhoutianyi}"
EXPERIMENT="${EXPERIMENT:-compact_eeg_3session_fusion_test20}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/outputs/${EXPERIMENT}/${SUBJECT}}"
CHECKPOINT="${CHECKPOINT:-${RESULT_DIR}/best.pt}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-Both subjects remain clearly visible throughout the shot. Natural motion, stable composition, photorealistic, high detail.}"

if [[ ! -f "${PROJECT_ROOT}/data/video_multilabels_2object.pt" ]]; then
  bash "${PROJECT_ROOT}/scripts/prepare_labels.sh"
fi

python "${PROJECT_ROOT}/src/infer.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --label-package "${PROJECT_ROOT}/data/video_multilabels_2object.pt" \
  --subject "${SUBJECT}" \
  --output-dir "${RESULT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

python "${PROJECT_ROOT}/src/generate_captions.py" \
  --predictions "${RESULT_DIR}/test_predictions.pt" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${RESULT_DIR}/captions" \
  --prompt-suffix "${PROMPT_SUFFIX}"

echo "final captions: ${RESULT_DIR}/captions/test_captions.csv"
