#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CONFIG="${1:-configs/exp_heavy.yaml}"
CONFIG_NAME=$(basename "${CONFIG}" .yaml)
RESULTS_DIR="results/${CONFIG_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "config  : ${CONFIG}"
echo "results : ${RESULTS_DIR}/"
echo "---"

"${PYTHON}" main_sundial_mpc.py \
    --config "${CONFIG}" \
    --save-dir "${RESULTS_DIR}"
