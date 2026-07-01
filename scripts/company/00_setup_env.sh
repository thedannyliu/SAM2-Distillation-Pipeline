#!/usr/bin/env bash
set -euo pipefail

VENV="/user-volume/env/sam2_stage1_torch24"
SKIP_SAM2_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV="$2"
      shift 2
      ;;
    --skip-sam2-smoke)
      SKIP_SAM2_SMOKE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -f requirements-stage1.txt ]]; then
  python -m pip install -r requirements-stage1.txt
fi

if [[ -f pyproject.toml || -f setup.py ]]; then
  # Official current SAM2 declares torch>=2.5.1. The company base image is
  # intentionally torch 2.4, so install the editable package without dependency
  # resolution or build isolation. The compatibility smoke below is the gate.
  python -m pip install --no-build-isolation --no-deps -e .
fi

python - <<'PY'
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
PY

if [[ "${SKIP_SAM2_SMOKE}" -eq 0 ]]; then
  python - <<'PY'
import importlib

for module in ["timm", "pandas", "pyarrow", "zarr", "PIL"]:
    importlib.import_module(module)
    print(f"import_ok={module}")

try:
    importlib.import_module("sam2")
    print("import_ok=sam2")
except Exception as exc:
    raise SystemExit(
        "SAM2 import failed in this environment. Keep torch 2.4 unchanged, "
        "then either pin a compatible SAM2 commit or request a torch>=2.5.1 "
        f"company image. Original error: {exc}"
    )
PY
fi

echo "Environment ready: ${VENV}"
