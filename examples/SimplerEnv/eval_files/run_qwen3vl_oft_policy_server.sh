#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

: "${CKPT_PATH:?Set CKPT_PATH to the local .pt checkpoint downloaded from StarVLA/Qwen3VL-OFT-Bridge-RT-1}"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10093}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16
