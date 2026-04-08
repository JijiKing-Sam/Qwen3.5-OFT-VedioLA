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
NUM_PROCESSES="${NUM_PROCESSES:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-JijiKing-Sam}"
WANDB_PROJECT="${WANDB_PROJECT:-videola}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

mkdir -p "${LOG_DIR}"
cp "$0" "${LOG_DIR}/"

if [[ -z "${NUM_PROCESSES}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NUM_PROCESSES=1
  fi
fi

REQUIRE_CUDA=1 \
MIN_GPUS="${NUM_PROCESSES}" \
MIN_VRAM_GB="${MIN_VRAM_GB:-20}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
bash scripts/bootstrap/remote_gpu_preflight.sh

{
  echo "experiment=qwen35oft_libero"
  echo "step_budget=${STEP_BUDGET}"
  echo "walltime=${WALLTIME}"
  echo "run_id=${RUN_ID}"
  echo "num_processes=${NUM_PROCESSES}"
  echo "attn_implementation=${ATTN_IMPLEMENTATION}"
  echo "started_at=$(date -Iseconds)"
  timeout --preserve-status "${WALLTIME}" \
    env \
      RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
      RUN_ID="${RUN_ID}" \
      MAX_TRAIN_STEPS="${STEP_BUDGET}" \
      NUM_PROCESSES="${NUM_PROCESSES}" \
      ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
      SAVE_INTERVAL="${SAVE_INTERVAL}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
      GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      WANDB_ENTITY="${WANDB_ENTITY}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh
} 2>&1 | tee "${LOG_PATH}"
