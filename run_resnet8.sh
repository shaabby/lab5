#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_resnet8"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

"${SCRIPT_DIR}/run_train.sh" \
    --model resnet8 \
    --output-dir "${OUTPUT_DIR}" \
    --device-target Ascend \
    --execution-mode graph \
    --amp-level O0 \
    --batch-size 128 \
    --workers 4

echo "WideResNet8 训练已完成，开始重新加载最佳 checkpoint 评估验证集。"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_validation.py" \
    --model resnet8 \
    --data-root "${SCRIPT_DIR}/eurosat_split" \
    --checkpoint "${OUTPUT_DIR}/best.ckpt" \
    --output-dir "${OUTPUT_DIR}" \
    --device-id "${DEVICE_ID:-0}" \
    --device-target Ascend \
    --batch-size 128 \
    --workers 4
