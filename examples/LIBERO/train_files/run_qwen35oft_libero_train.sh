#!/usr/bin/env bash
set -euo pipefail

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${REPO_ROOT}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-Qwen35OFT}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3.5-4B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CONFIG_YAML="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_spatial}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/Checkpoints}"
RUN_ID="${RUN_ID:-videola_qwen35_oft_libero_spatial}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwen35.base_vlm "${BASE_VLM}" \
  --framework.qwen35.attn_implementation "${ATTN_IMPLEMENTATION}" \
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
