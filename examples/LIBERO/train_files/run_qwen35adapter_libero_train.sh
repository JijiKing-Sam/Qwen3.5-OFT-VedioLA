#!/usr/bin/env bash
set -euo pipefail

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE="${WANDB_MODE:-offline}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${REPO_ROOT}"

CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
ENV_NAME="${ENV_NAME:-videola-qwen35-train}"
FRAMEWORK_NAME="${FRAMEWORK_NAME:-Qwen35Adapter}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3.5-4B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CONFIG_YAML="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_qwen35_adapter_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_spatial}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/Checkpoints}"
RUN_ID="${RUN_ID:-videola_qwen35_adapter_libero_spatial}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
UNFREEZE_LAST_TEXT_LAYERS="${UNFREEZE_LAST_TEXT_LAYERS:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
WANDB_ENTITY="${WANDB_ENTITY:-JijiKing-Sam}"
WANDB_PROJECT="${WANDB_PROJECT:-videola}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
IS_RESUME="${IS_RESUME:-false}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"

if [[ -z "${FREEZE_MODULES}" && -n "${UNFREEZE_LAST_TEXT_LAYERS}" ]]; then
  if ! [[ "${UNFREEZE_LAST_TEXT_LAYERS}" =~ ^[0-9]+$ ]]; then
    echo "UNFREEZE_LAST_TEXT_LAYERS must be a non-negative integer" >&2
    exit 1
  fi

  TOTAL_TEXT_LAYERS=32
  if (( UNFREEZE_LAST_TEXT_LAYERS > TOTAL_TEXT_LAYERS )); then
    UNFREEZE_LAST_TEXT_LAYERS="${TOTAL_TEXT_LAYERS}"
  fi

  FREEZE_LIST=(
    "qwen_vl_interface.model.lm_head"
    "qwen_vl_interface.model.model.visual"
    "qwen_vl_interface.model.model.language_model.embed_tokens"
  )

  FREEZE_UNTIL=$(( TOTAL_TEXT_LAYERS - UNFREEZE_LAST_TEXT_LAYERS ))
  for (( layer_idx=0; layer_idx<FREEZE_UNTIL; layer_idx++ )); do
    FREEZE_LIST+=("qwen_vl_interface.model.model.language_model.layers.${layer_idx}")
  done

  IFS=,
  FREEZE_MODULES="${FREEZE_LIST[*]}"
  unset IFS
fi

if [[ -n "${CONDA_EXE}" ]]; then
  eval "$("${CONDA_EXE}" shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda activate "${ENV_NAME}"
  fi
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NUM_PROCESSES=1
  fi
fi

if [[ -z "${NUM_PROCESSES}" || "${NUM_PROCESSES}" == "0" ]]; then
  NUM_PROCESSES=1
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"
cp "${CONFIG_YAML}" "${OUTPUT_DIR}/$(basename "${CONFIG_YAML}")"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  GIT_COMMIT="$(git rev-parse HEAD)"
elif [[ -f "${REPO_ROOT}/.codex-import-commit" ]]; then
  GIT_BRANCH="<snapshot>"
  GIT_COMMIT="$(cat "${REPO_ROOT}/.codex-import-commit")"
else
  GIT_BRANCH="<unavailable>"
  GIT_COMMIT="<unavailable>"
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  REQUIRE_CUDA=1 \
  LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}" \
  CONFIG_YAML="${CONFIG_YAML}" \
  RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
  MIN_GPUS="${NUM_PROCESSES}" \
  bash scripts/bootstrap/remote_gpu_preflight.sh
fi

{
  echo "started_at=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "repo_root=${REPO_ROOT}"
  echo "git_branch=${GIT_BRANCH}"
  echo "git_commit=${GIT_COMMIT}"
  echo "framework_name=${FRAMEWORK_NAME}"
  echo "base_vlm=${BASE_VLM}"
  echo "attn_implementation=${ATTN_IMPLEMENTATION}"
  echo "config_yaml=${CONFIG_YAML}"
  echo "libero_data_root=${LIBERO_DATA_ROOT}"
  echo "data_mix=${DATA_MIX}"
  echo "run_root_dir=${RUN_ROOT_DIR}"
  echo "run_id=${RUN_ID}"
  echo "unfreeze_last_text_layers=${UNFREEZE_LAST_TEXT_LAYERS}"
  echo "num_processes=${NUM_PROCESSES}"
  echo "max_train_steps=${MAX_TRAIN_STEPS}"
  echo "save_interval=${SAVE_INTERVAL}"
  echo "eval_interval=${EVAL_INTERVAL}"
  echo "logging_frequency=${LOGGING_FREQUENCY}"
  echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  echo "wandb_mode=${WANDB_MODE}"
  echo "wandb_entity=${WANDB_ENTITY}"
  echo "wandb_project=${WANDB_PROJECT}"
  echo "is_resume=${IS_RESUME}"
  echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
} > "${OUTPUT_DIR}/launch_env.txt"

TRAIN_ARGS=(
  --config_yaml "${CONFIG_YAML}"
  --framework.name "${FRAMEWORK_NAME}"
  --framework.qwen35.base_vlm "${BASE_VLM}"
  --framework.qwen35.attn_implementation "${ATTN_IMPLEMENTATION}"
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --trainer.freeze_modules "${FREEZE_MODULES}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)

if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

if [[ "${IS_RESUME}" == "true" ]]; then
  TRAIN_ARGS+=(--trainer.is_resume true)
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  "${TRAIN_ARGS[@]}" \
  "$@"
