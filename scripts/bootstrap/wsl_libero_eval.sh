#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export TMPDIR="${REPO_ROOT}/.cache/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/.logs" "${REPO_ROOT}/.libero-config"

if [ ! -x "${REPO_ROOT}/.venv-libero/bin/python" ]; then
  if python3 -m venv --help >/dev/null 2>&1; then
    python3 -m venv "${REPO_ROOT}/.venv-libero" || true
  fi

  if [ ! -x "${REPO_ROOT}/.venv-libero/bin/python" ]; then
    python3 -m virtualenv "${REPO_ROOT}/.venv-libero"
  fi
fi

if [ ! -x "${REPO_ROOT}/.venv-libero/bin/pip" ]; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${TMPDIR}/get-pip.py"
  "${REPO_ROOT}/.venv-libero/bin/python" "${TMPDIR}/get-pip.py"
fi

export PATH="${REPO_ROOT}/.venv-libero/bin:${PATH}"

"${REPO_ROOT}/.venv-libero/bin/pip" install --upgrade pip setuptools wheel
"${REPO_ROOT}/.venv-libero/bin/pip" install -r "${REPO_ROOT}/requirements/libero_eval.txt"
"${REPO_ROOT}/.venv-libero/bin/pip" install -e "${REPO_ROOT}/third_party/LIBERO" --no-deps

cat > "${REPO_ROOT}/.libero-config/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${REPO_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${REPO_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${REPO_ROOT}/third_party/LIBERO/datasets
assets: ${REPO_ROOT}/third_party/LIBERO/libero/libero/assets
EOF

export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero-config"
export LIBERO_HOME="${REPO_ROOT}/third_party/LIBERO"
export PYTHONPATH="${REPO_ROOT}/third_party/LIBERO:${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"${REPO_ROOT}/.venv-libero/bin/python" - <<'PY'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
print("libero_import_ok", True)
print("task_name", task.name)
print("bddl_files", get_libero_path("bddl_files"))
print("offscreen_env_cls", OffScreenRenderEnv.__name__)
PY
