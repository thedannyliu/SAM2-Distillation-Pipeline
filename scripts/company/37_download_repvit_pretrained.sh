#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/group-volume/danny-dataset/sam2_distill/checkpoints/repvit}"
FORCE=0
SKIP_TIMM_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --skip-timm-smoke)
      SKIP_TIMM_SMOKE=1
      shift
      ;;
    -h|--help)
      echo "Usage: scripts/company/37_download_repvit_pretrained.sh [--out-root PATH] [--force] [--skip-timm-smoke]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

args=(--out-root "${OUT_ROOT}")
if [[ "${FORCE}" == "1" ]]; then
  args+=(--force)
fi
if [[ "${SKIP_TIMM_SMOKE}" == "1" ]]; then
  args+=(--skip-timm-smoke)
fi

cd "${REPO_ROOT}" || exit 1
python tools/data/download_repvit_pretrained.py "${args[@]}"
status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "RepViT pretrained preparation failed with status ${status}" >&2
  exit "${status}"
fi

echo "RepViT pretrained checkpoints ready under ${OUT_ROOT}"
