#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_qwen35_oft_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/Checkpoints}"
ENV_NAME="${ENV_NAME:-videola-qwen35-train}"
FRAMEWORK_NAME="${FRAMEWORK_NAME:-Qwen35OFT}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3.5-4B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
MIN_GPUS="${MIN_GPUS:-1}"
MIN_VRAM_GB="${MIN_VRAM_GB:-20}"
CHECK_WANDB="${CHECK_WANDB:-0}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"

if [[ -n "${CONDA_EXE}" ]]; then
  eval "$("${CONDA_EXE}" shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda activate "${ENV_NAME}"
  fi
fi

echo "== Repo =="
echo "repo_root=${REPO_ROOT}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "git_commit=$(git rev-parse HEAD)"
elif [[ -f "${REPO_ROOT}/.codex-import-commit" ]]; then
  echo "git_branch=<snapshot>"
  echo "git_commit=$(cat "${REPO_ROOT}/.codex-import-commit")"
else
  echo "git_branch=<unavailable>"
  echo "git_commit=<unavailable>"
fi
echo

echo "== System =="
echo "hostname=$(hostname)"
echo "date=$(date -Iseconds)"
echo "python=$(command -v python)"
python --version
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    echo "nvidia-smi not found"
    exit 1
  fi
else
  echo "== GPUs =="
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo
fi

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Missing config yaml: ${CONFIG_YAML}"
  exit 1
fi

if [[ ! -d "${LIBERO_DATA_ROOT}" ]]; then
  echo "Missing LIBERO data root: ${LIBERO_DATA_ROOT}"
  exit 1
fi

mkdir -p "${RUN_ROOT_DIR}"
touch "${RUN_ROOT_DIR}/.write_test"
rm -f "${RUN_ROOT_DIR}/.write_test"

python - "${CONFIG_YAML}" "${LIBERO_DATA_ROOT}" "${RUN_ROOT_DIR}" "${FRAMEWORK_NAME}" "${BASE_VLM}" "${ATTN_IMPLEMENTATION}" "${REQUIRE_CUDA}" "${MIN_GPUS}" "${MIN_VRAM_GB}" "${CHECK_WANDB}" <<'PY'
import os
import sys

import torch
import transformers
import accelerate
import wandb
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework  # noqa: F401

(
    config_yaml,
    libero_data_root,
    run_root_dir,
    framework_name,
    base_vlm,
    attn_impl,
    require_cuda,
    min_gpus,
    min_vram_gb,
    check_wandb,
) = sys.argv[1:]

cfg = OmegaConf.load(config_yaml)
cfg.framework.name = framework_name
cfg.framework.framework_py = framework_name
if getattr(cfg.framework, "qwen35", None) is not None:
    cfg.framework.qwen35.base_vlm = base_vlm
    cfg.framework.qwen35.attn_implementation = attn_impl

assert os.path.isdir(libero_data_root), libero_data_root
assert os.path.isdir(run_root_dir), run_root_dir

print("== Python stack ==")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("wandb:", wandb.__version__)
print("framework:", cfg.framework.name)
print("base_vlm:", cfg.framework.qwen35.base_vlm)
print("attn_implementation:", cfg.framework.qwen35.attn_implementation)
print("libero_data_root:", libero_data_root)
print("run_root_dir:", run_root_dir)

cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
print("cuda_available:", cuda_available)
print("gpu_count:", device_count)

if require_cuda == "1" and not cuda_available:
    raise SystemExit("CUDA is required but torch.cuda.is_available() is false")

min_gpus_int = int(min_gpus)
if require_cuda == "1" and device_count < min_gpus_int:
    raise SystemExit(f"Need at least {min_gpus_int} GPUs, found {device_count}")

if cuda_available:
    min_vram_bytes = int(float(min_vram_gb) * (1024 ** 3))
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        total_memory = props.total_memory
        print(f"gpu[{index}]: name={props.name}, vram_gb={total_memory / (1024 ** 3):.2f}")
        if total_memory < min_vram_bytes:
            print(
                f"warning: gpu[{index}] has only {total_memory / (1024 ** 3):.2f} GB,"
                f" below requested {float(min_vram_gb):.1f} GB"
            )

if attn_impl == "flash_attention_2":
    try:
        import flash_attn  # noqa: F401

        print("flash_attn: available")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"flash_attention_2 requested but flash_attn import failed: {exc}") from exc

if check_wandb == "1":
    mode = os.environ.get("WANDB_MODE", "")
    api_key = os.environ.get("WANDB_API_KEY", "")
    if mode not in {"offline", "disabled"} and not api_key:
        raise SystemExit("WANDB_MODE is online but WANDB_API_KEY is missing")
    print("wandb_mode:", mode or "<unset>")

print("preflight_status: ok")
PY
