#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda/edgetam}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_PREFIX}/bin/python" -m pip install -r "${ROOT}/requirements-edgetam.txt"
"${ENV_PREFIX}/bin/python" -m pip install --no-deps -e "${ROOT}"

echo "Environment ready: ${ENV_PREFIX}"
echo "Activate with:"
echo "  conda activate ${ENV_PREFIX}"
