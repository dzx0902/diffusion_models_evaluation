#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Environment variables such as SUBJECT, DEVICE, EPOCHS and EXPERIMENT are
# automatically inherited by both stages.
bash "${SCRIPT_DIR}/train.sh"
bash "${SCRIPT_DIR}/infer_and_caption.sh"
