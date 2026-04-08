#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
ENV_NAME="${ENV_NAME:-videola-qwen35-train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

if [[ -z "${CONDA_EXE}" ]]; then
  echo "conda not found; set CONDA_EXE=/path/to/conda"
  exit 1
fi

eval "$("${CONDA_EXE}" shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements/train.txt

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi

python -m pip install -e .

python - <<'PY'
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
