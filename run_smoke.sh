#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_train.sh" \
    --output-dir "${SCRIPT_DIR}/outputs_smoke" \
    --epochs 1 \
    --warmup-epochs 0 \
    --batch-size 32 \
    --workers 2 \
    --train-samples 256 \
    --val-samples 128 \
    --base-lr 0.001 \
    --patience 1 \
    --max-minutes 5 \
    --execution-mode graph \
    --amp-level O0 \
    "$@"
