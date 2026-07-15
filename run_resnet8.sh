#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_train.sh" \
    --model resnet8 \
    --output-dir "${SCRIPT_DIR}/outputs_resnet8" \
    --device-target Ascend \
    --execution-mode graph \
    --amp-level O0 \
    --batch-size 128 \
    --workers 4 \
    "$@"
