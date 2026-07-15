#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "未找到 ${PYTHON_BIN}，请先创建 .venv。" >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" \
    --data-root "${SCRIPT_DIR}/eurosat_split" \
    --output-dir "${SCRIPT_DIR}/outputs" \
    --device-id "${DEVICE_ID:-0}" \
    "$@"
