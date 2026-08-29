#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Script Error"' ERR

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/userhome2/zhoutianyi/Dataset/Multi-Object}"

python "${PROJECT_ROOT}/src/prepare_labels.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${PROJECT_ROOT}/data"
