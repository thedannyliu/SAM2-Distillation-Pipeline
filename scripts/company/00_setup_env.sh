#!/usr/bin/env bash
set -euo pipefail

SKIP_SAM2_SMOKE=0
SAM2_UPSTREAM="${SAM2_UPSTREAM:-}"
INSTALL_MODE="${INSTALL_MODE:-user}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sam2-smoke)
      SKIP_SAM2_SMOKE=1
      shift
      ;;
    --sam2-upstream)
      SAM2_UPSTREAM="$2"
      shift 2
      ;;
    --install-mode)
      INSTALL_MODE="$2"
      shift 2
      ;;
    --venv)
      echo "--venv is no longer supported. Use the company container Python directly." >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "${INSTALL_MODE}" in
  user)
    PIP_PREFIX=(--user)
    ;;
  system)
    PIP_PREFIX=()
    ;;
  *)
    echo "--install-mode must be user or system" >&2
    exit 2
    ;;
esac

python - <<'PY'
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"python_executable={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
PY

python -m pip install "${PIP_PREFIX[@]}" --upgrade pip setuptools wheel

if [[ -f requirements-stage1.txt ]]; then
  python -m pip install "${PIP_PREFIX[@]}" -r requirements-stage1.txt
fi

if [[ -f pyproject.toml || -f setup.py ]]; then
  # Official current SAM2 declares torch>=2.5.1. The company base image is
  # intentionally torch 2.4, so install editable without dependency resolution.
  python -m pip install "${PIP_PREFIX[@]}" --no-build-isolation --no-deps -e .
elif [[ -n "${SAM2_UPSTREAM}" && -d "${SAM2_UPSTREAM}" ]]; then
  python -m pip install "${PIP_PREFIX[@]}" --no-build-isolation --no-deps -e "${SAM2_UPSTREAM}"
fi

if [[ "${SKIP_SAM2_SMOKE}" -eq 0 ]]; then
  python - <<'PY'
import importlib

for module in ["torch", "timm", "huggingface_hub", "wandb", "pandas", "pyarrow", "zarr", "PIL"]:
    importlib.import_module(module)
    print(f"import_ok={module}")

try:
    importlib.import_module("sam2")
    print("import_ok=sam2")
except Exception as exc:
    raise SystemExit(
        "SAM2 import failed in this environment. Keep container torch unchanged, "
        "then either pin a compatible SAM2 commit or request a torch>=2.5.1 "
        f"company image. Original error: {exc}"
    )
PY
fi

echo "Environment ready using current container Python."
