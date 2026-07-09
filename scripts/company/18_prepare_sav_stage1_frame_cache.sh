#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
DEFAULT_SAV_ROOT="${DATA_ROOT}/SA-V"
if [[ -d "/mnt/data/danny-dataset/SA-V" ]]; then
  DEFAULT_SAV_ROOT="/mnt/data/danny-dataset/SA-V"
fi
SAV_ROOT="${SAV_ROOT:-${DEFAULT_SAV_ROOT}}"
TRAIN_ROOT="${TRAIN_ROOT:-${SAV_ROOT}/sav_train}"
VAL_ROOT="${VAL_ROOT:-${SAV_ROOT}/sav_val}"
TEST_ROOT="${TEST_ROOT:-${SAV_ROOT}/sav_test}"

SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
CACHE_NAME="${CACHE_NAME:-stage1_vbal16_6fps}"
OUT_ROOT="${OUT_ROOT:-${SAM2D_ROOT}/data/sav_v2/frame_cache/${CACHE_NAME}}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/${CACHE_NAME}.parquet}"

TRAIN_FRAMES_PER_VIDEO="${TRAIN_FRAMES_PER_VIDEO:-16}"
VAL_FRAMES_PER_VIDEO="${VAL_FRAMES_PER_VIDEO:-8}"
TEST_FRAMES_PER_VIDEO="${TEST_FRAMES_PER_VIDEO:-0}"
MAX_TRAIN_VIDEOS="${MAX_TRAIN_VIDEOS:-0}"
MAX_VAL_VIDEOS="${MAX_VAL_VIDEOS:-0}"
MAX_TEST_VIDEOS="${MAX_TEST_VIDEOS:-0}"
NUM_WORKERS="${NUM_WORKERS:-64}"
JPEG_QUALITY="${JPEG_QUALITY:-90}"
SEED="${SEED:-sam2_stage1_sav_vbal16_6fps_v1}"
USE_AUTO="${USE_AUTO:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/18_prepare_sav_stage1_frame_cache.sh

Storage-aware SA-V Stage 1 preparation. It extracts only selected 6fps-aligned
frames from raw MP4 videos instead of expanding full 24fps train data.

Key env vars:
  DATA_ROOT=/group-volume/danny-dataset
  SAV_ROOT=/mnt/data/danny-dataset/SA-V
  TRAIN_ROOT=$SAV_ROOT/sav_train
  VAL_ROOT=$SAV_ROOT/sav_val
  OUT_ROOT=$DATA_ROOT/sam2_distill/data/sav_v2/frame_cache/stage1_vbal16_6fps
  MANIFEST=$DATA_ROOT/sam2_distill/manifests/stage1_vbal16_6fps.parquet
  TRAIN_FRAMES_PER_VIDEO=16
  VAL_FRAMES_PER_VIDEO=8
  NUM_WORKERS=64
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

args=(
  --train-root "${TRAIN_ROOT}"
  --out-root "${OUT_ROOT}"
  --manifest "${MANIFEST}"
  --train-frames-per-video "${TRAIN_FRAMES_PER_VIDEO}"
  --val-frames-per-video "${VAL_FRAMES_PER_VIDEO}"
  --test-frames-per-video "${TEST_FRAMES_PER_VIDEO}"
  --max-train-videos "${MAX_TRAIN_VIDEOS}"
  --max-val-videos "${MAX_VAL_VIDEOS}"
  --max-test-videos "${MAX_TEST_VIDEOS}"
  --num-workers "${NUM_WORKERS}"
  --jpeg-quality "${JPEG_QUALITY}"
  --seed "${SEED}"
)

if [[ -d "${VAL_ROOT}" ]]; then
  args+=(--val-root "${VAL_ROOT}")
fi
if [[ -d "${TEST_ROOT}" && "${TEST_FRAMES_PER_VIDEO}" -gt 0 ]]; then
  args+=(--test-root "${TEST_ROOT}")
fi
if [[ "${USE_AUTO}" -eq 1 ]]; then
  args+=(--use-auto)
fi

python tools/data/prepare_sav_stage1_frame_cache.py "${args[@]}"

du -sh "${OUT_ROOT}" || true
