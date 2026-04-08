#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-smoke}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.cache/pip}"
TMPDIR="${TMPDIR:-${REPO_ROOT}/.cache/tmp}"

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"
export PIP_CACHE_DIR TMPDIR

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "${PYTHON_BIN}" /tmp/get-pip.py --user
fi

if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --user virtualenv
  "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements/smoke.txt
python -m pip install -e .

python - <<'PY'
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
