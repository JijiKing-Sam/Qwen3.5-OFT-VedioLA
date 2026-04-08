#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "timeout not found; install coreutils before using the formal run wrapper"
  exit 1
fi

COLLAB_REPO_ROOT="${COLLAB_REPO_ROOT:-}"
RUN_CMD="${RUN_CMD:-}"
STEP_BUDGET="${STEP_BUDGET:-80000}"
WALLTIME="${WALLTIME:-7d}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/80k_runs}"
RUN_ID="${RUN_ID:-collab_control_80k}"
LOG_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"

if [[ -z "${COLLAB_REPO_ROOT}" ]]; then
  echo "set COLLAB_REPO_ROOT to the collaborator repo path"
  exit 1
fi

if [[ ! -d "${COLLAB_REPO_ROOT}" ]]; then
  echo "COLLAB_REPO_ROOT does not exist: ${COLLAB_REPO_ROOT}"
  exit 1
fi

if [[ -z "${RUN_CMD}" ]]; then
  echo "set RUN_CMD to the collaborator baseline command"
  exit 1
fi

mkdir -p "${LOG_DIR}"
cp "$0" "${LOG_DIR}/"

{
  echo "experiment=collab_control"
  echo "repo_root=${COLLAB_REPO_ROOT}"
  echo "step_budget=${STEP_BUDGET}"
  echo "walltime=${WALLTIME}"
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -Iseconds)"
  (
    cd "${COLLAB_REPO_ROOT}"
    timeout --preserve-status "${WALLTIME}" env STEP_BUDGET="${STEP_BUDGET}" bash -lc "${RUN_CMD}"
  )
} 2>&1 | tee "${LOG_PATH}"
