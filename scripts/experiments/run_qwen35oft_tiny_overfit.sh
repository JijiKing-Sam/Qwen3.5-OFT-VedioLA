#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

STEP_BUDGET="${STEP_BUDGET:-200}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/tiny_overfit}"
RUN_ID="${RUN_ID:-qwen35oft_libero_tiny_overfit}"
LOG_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/train.log"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-JijiKing-Sam}"
WANDB_PROJECT="${WANDB_PROJECT:-videola}"

mkdir -p "${LOG_DIR}"
cp "$0" "${LOG_DIR}/"

REQUIRE_CUDA=1 \
MIN_GPUS="${NUM_PROCESSES}" \
MIN_VRAM_GB="${MIN_VRAM_GB:-20}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
bash scripts/bootstrap/remote_gpu_preflight.sh

{
  echo "experiment=qwen35oft_tiny_overfit"
  echo "step_budget=${STEP_BUDGET}"
  echo "run_id=${RUN_ID}"
  echo "num_processes=${NUM_PROCESSES}"
  echo "attn_implementation=${ATTN_IMPLEMENTATION}"
  echo "started_at=$(date -Iseconds)"
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
    bash examples/LIBERO/train_files/run_qwen35oft_libero_train.sh \
      --is_debug true
} 2>&1 | tee "${LOG_PATH}"
