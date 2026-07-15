#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_train.sh" \
    --output-dir "${SCRIPT_DIR}/outputs_smoke" \
    --epochs 2 \
    --warmup-epochs 1 \
    --batch-size 64 \
    --workers 4 \
    --train-samples 1024 \
    --val-samples 512 \
    --base-lr 0.0015 \
    --patience 2 \
    --max-minutes 10 \
    "$@"
