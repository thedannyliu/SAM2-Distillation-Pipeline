#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
OUT_DIR="${OUT_DIR:-${SAM2D_ROOT}/runs/sam2_tracking_component_profile_v1/sam21l}"
GPU="${GPU:-0}"
WARMUP_FRAMES="${WARMUP_FRAMES:-16}"
MEASURE_FRAMES="${MEASURE_FRAMES:-128}"
REPETITIONS="${REPETITIONS:-3}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"

for path in \
  "${SAM2_ROOT}" \
  "${SAM2_CHECKPOINT}" \
  "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
  "${SAV_ROOT}/sav_val/Annotations_6fps" \
  "${SAV_ROOT}/sav_val/sav_val.txt"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing input: ${path}" >&2
    return 1 2>/dev/null || exit 1
  fi
done

CUDA_VISIBLE_DEVICES="${GPU}" python \
  tools/benchmark/profile_sam2_tracking_components.py \
  --sam2-root "${SAM2_ROOT}" \
  --sam2-cfg "${SAM2_CONFIG}" \
  --checkpoint "${SAM2_CHECKPOINT}" \
  --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
  --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
  --video-list-file "${SAV_ROOT}/sav_val/sav_val.txt" \
  --out-dir "${OUT_DIR}" \
  --warmup-frames "${WARMUP_FRAMES}" \
  --measure-frames "${MEASURE_FRAMES}" \
  --repetitions "${REPETITIONS}" \
  --max-objects "${MAX_OBJECTS}" \
  --device cuda
STATUS="$?"

echo "SAM2 tracking component profile status: ${STATUS}"
echo "Summary: ${OUT_DIR}/summary.json"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
