#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "timeout not found; install coreutils before using the formal run wrapper"
  exit 1
fi

STEP_BUDGET="${STEP_BUDGET:-80000}"
WALLTIME="${WALLTIME:-7d}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/80k_runs}"
RUN_ID="${RUN_ID:-qwen35oft_libero_80k}"
LOG_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"
cp "$0" "${LOG_DIR}/"

{
  echo "experiment=qwen35oft_libero"
  echo "step_budget=${STEP_BUDGET}"
  echo "walltime=${WALLTIME}"
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -Iseconds)"
  timeout --preserve-status "${WALLTIME}" \
    env RUN_ROOT_DIR="${RUN_ROOT_DIR}" RUN_ID="${RUN_ID}" MAX_TRAIN_STEPS="${STEP_BUDGET}" \
    bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
} 2>&1 | tee "${LOG_PATH}"
