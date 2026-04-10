#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
ENV_NAME="${ENV_NAME:-videola-qwen35-train}"
VENV_DIR="${VENV_DIR:-}"
PYTHON_EXE="${PYTHON_EXE:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_LIBERO_EVAL="${INSTALL_LIBERO_EVAL:-1}"

activate_python_env() {
  if [[ -n "${CONDA_EXE}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"

    if [[ -d "${ENV_NAME}" ]]; then
      conda activate "${ENV_NAME}"
      echo "Using conda env: ${ENV_NAME}"
      return
    fi

    if [[ "${ENV_NAME}" == */* ]]; then
      conda create -y -p "${ENV_NAME}" python="${PYTHON_VERSION}"
      conda activate "${ENV_NAME}"
      echo "Using conda env: ${ENV_NAME}"
      return
    fi

    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
      conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    fi

    conda activate "${ENV_NAME}"
    echo "Using conda env: ${ENV_NAME}"
    return
  fi

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active venv: ${VIRTUAL_ENV}"
    return
  fi

  if [[ -n "${VENV_DIR}" ]]; then
    if [[ -z "${PYTHON_EXE}" ]]; then
      PYTHON_EXE="$(command -v python${PYTHON_VERSION} || true)"
    fi
    if [[ -z "${PYTHON_EXE}" ]]; then
      PYTHON_EXE="$(command -v python3 || true)"
    fi
    if [[ -z "${PYTHON_EXE}" ]]; then
      echo "python executable not found; set PYTHON_EXE=/path/to/python"
      exit 1
    fi

    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
      "${PYTHON_EXE}" -m venv "${VENV_DIR}"
    fi

    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    echo "Using venv: ${VENV_DIR}"
    return
  fi

  echo "No environment manager available."
  echo "Either:"
  echo "  1. set CONDA_EXE=/path/to/conda"
  echo "  2. activate a venv before running this script"
  echo "  3. set VENV_DIR=/path/to/venv and optionally PYTHON_EXE=/path/to/python"
  exit 1
}

activate_python_env

python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements/train.txt

if [[ "${INSTALL_LIBERO_EVAL}" == "1" ]]; then
  python -m pip install -r requirements/libero_eval.txt
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi

python -m pip install -e .

python - <<'PY'
import os
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("virtual_env:", os.environ.get("VIRTUAL_ENV"))
PY
