#!/usr/bin/env bash
# Run on the GPU server. All four comparisons are declared before opening test results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-run}"
if [[ "$MODE" != run && "$MODE" != --dry-run ]]; then
  echo 'Usage: bash scripts/run_c2_residual_wave1.sh [--dry-run]' >&2
  exit 2
fi
run() {
  if [[ "$MODE" == --dry-run ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

PREPARED=outputs/eeg_semantic/c2_residual_v2/prepared_fold1_dim64.pt
if [[ "$MODE" == run ]]; then
  mkdir -p outputs/eeg_semantic/c2_residual_v2
fi

if [ ! -f "$PREPARED" ]; then
  run conda run --no-capture-output -n eeg-semantic python -u \
    scripts/run_c2_residual.py --stage prepare \
    --prepared "$PREPARED" --dim 64
fi

for variant in regression contrastive variance; do
  run conda run --no-capture-output -n eeg-semantic python -u \
    scripts/run_c2_residual.py --stage train \
    --prepared "$PREPARED" --variant "$variant" --device cuda --resume

  run conda run --no-capture-output -n eeg-semantic python -u \
    scripts/run_c2_residual.py --stage evaluate --partition validation \
    --prepared "$PREPARED" --variant "$variant" --device cuda
done

for variant in mean regression contrastive variance; do
  run conda run --no-capture-output -n eeg-semantic python -u \
    scripts/run_c2_residual.py --stage evaluate --partition test \
    --prepared "$PREPARED" --variant "$variant" --device cuda
done

if [[ "$MODE" == run ]]; then
  echo '[c2-residual] wave1 COMPLETE: three training runs and four test reports'
else
  echo '[c2-residual] dry-run COMPLETE; no training or evaluation executed'
fi
