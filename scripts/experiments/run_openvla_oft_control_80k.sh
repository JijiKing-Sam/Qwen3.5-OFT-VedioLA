#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "timeout not found; install coreutils before using the formal run wrapper"
  exit 1
fi

OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-}"
RUN_CMD="${RUN_CMD:-}"
STEP_BUDGET="${STEP_BUDGET:-80000}"
WALLTIME="${WALLTIME:-7d}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/80k_runs}"
RUN_ID="${RUN_ID:-openvla_oft_control_80k}"
LOG_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"

if [[ -z "${OPENVLA_OFT_ROOT}" ]]; then
  echo "set OPENVLA_OFT_ROOT to the external OpenVLA-OFT repo path"
  exit 1
fi

if [[ ! -d "${OPENVLA_OFT_ROOT}" ]]; then
  echo "OPENVLA_OFT_ROOT does not exist: ${OPENVLA_OFT_ROOT}"
  exit 1
fi

if [[ -z "${RUN_CMD}" ]]; then
  echo "set RUN_CMD to the OpenVLA-OFT training command"
  exit 1
fi

mkdir -p "${LOG_DIR}"
cp "$0" "${LOG_DIR}/"

{
  echo "experiment=openvla_oft_control"
  echo "repo_root=${OPENVLA_OFT_ROOT}"
  echo "step_budget=${STEP_BUDGET}"
  echo "walltime=${WALLTIME}"
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -Iseconds)"
  (
    cd "${OPENVLA_OFT_ROOT}"
    timeout --preserve-status "${WALLTIME}" env STEP_BUDGET="${STEP_BUDGET}" bash -lc "${RUN_CMD}"
  )
} 2>&1 | tee "${LOG_PATH}"
