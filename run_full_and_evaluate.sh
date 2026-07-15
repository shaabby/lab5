#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
GUARD_PATH="${OUTPUT_DIR}/TEST_EVALUATION_STARTED.json"
RESULT_PATH="${OUTPUT_DIR}/test_result.json"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

if [[ -e "${GUARD_PATH}" || -e "${RESULT_PATH}" ]]; then
    echo "检测到测试评估已启动或已完成的记录，拒绝启动全量训练与测试。" >&2
    exit 1
fi

"${SCRIPT_DIR}/run_train.sh" \
    --device-target Ascend \
    --execution-mode graph \
    --amp-level O0 \
    --batch-size 128 \
    --workers 4

echo "全量训练已完成，开始唯一一次测试集评估。"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_test_once.py" \
    --data-root "${SCRIPT_DIR}/eurosat_split" \
    --checkpoint "${OUTPUT_DIR}/best.ckpt" \
    --output-dir "${OUTPUT_DIR}" \
    --device-id "${DEVICE_ID:-0}" \
    --batch-size 128 \
    --workers 4 \
    --confirm-final-test
