#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${REPO_ROOT}"

PORT="${PORT:-10095}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
MAX_TASKS="${MAX_TASKS:-1}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-results/debug/libero_qwen35_sanity}"
ACTION_STATS_PATH="${ACTION_STATS_PATH:-results/debug/libero_identity_stats.json}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-8}"

mkdir -p .logs "${VIDEO_OUT_PATH%/*}" "${VIDEO_OUT_PATH}"

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="${REPO_ROOT}/.cache/huggingface/transformers"
export TMPDIR="${REPO_ROOT}/.cache/tmp"

source "${REPO_ROOT}/.venv-smoke/bin/activate"
python scripts/write_identity_dataset_statistics.py \
  --output_path "${ACTION_STATS_PATH}" \
  --unnorm_key franka \
  > .logs/libero_identity_stats.log 2>&1

python deployment/model_server/server_policy_from_config.py \
  --config_yaml examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml \
  --framework_name Qwen35OFT \
  --port "${PORT}" \
  --use_bf16 \
  --metadata_env libero \
  > .logs/qwen35_libero_server.log 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 24); do
  if python scripts/ping_policy_server.py --host "${SERVER_HOST}" --port "${PORT}" --num_views 2 > .logs/qwen35_libero_ping.log 2>&1; then
    break
  fi
  sleep 5
done
cat .logs/qwen35_libero_ping.log
deactivate

export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero-config"
export LIBERO_HOME="${REPO_ROOT}/third_party/LIBERO"
export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export WS_POLICY_PING_INTERVAL="${WS_POLICY_PING_INTERVAL:-0}"
export WS_POLICY_PING_TIMEOUT="${WS_POLICY_PING_TIMEOUT:-0}"

"${REPO_ROOT}/.venv-libero/bin/python" examples/LIBERO/eval_files/eval_libero.py \
  --args.host "${SERVER_HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.pretrained-path "" \
  --args.unnorm-key franka \
  --args.action-stats-path "${ACTION_STATS_PATH}" \
  --args.action-chunk-size "${ACTION_CHUNK_SIZE}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  > .logs/libero_qwen35_dry_run.log 2>&1

tail -n 120 .logs/libero_qwen35_dry_run.log
