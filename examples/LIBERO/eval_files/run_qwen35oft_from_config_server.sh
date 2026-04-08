#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10095}"
CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3.5-4B}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" deployment/model_server/server_policy_from_config.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework_name Qwen35OFT \
  --base_vlm "${BASE_VLM}" \
  --port "${PORT}" \
  --use_bf16 \
  --metadata_env libero
