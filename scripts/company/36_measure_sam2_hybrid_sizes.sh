#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TV21_CHECKPOINT="${TV21_CHECKPOINT:?Set TV21_CHECKPOINT to its best.pt}"
TV11_CHECKPOINT="${TV11_CHECKPOINT:?Set TV11_CHECKPOINT to its best.pt}"
TV5_CHECKPOINT="${TV5_CHECKPOINT:?Set TV5_CHECKPOINT to its best.pt}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
REPORT_DIR="${REPORT_DIR:-/user-volume/sam2_hybrid_sizes_${HOSTNAME}}"
EXPORT_DTYPES="${EXPORT_DTYPES-fp32,fp16}"

export_args=()
IFS=, read -r -a dtype_values <<< "${EXPORT_DTYPES}"
for dtype in "${dtype_values[@]}"; do
  [[ -n "${dtype}" ]] && export_args+=(--export-dtype "${dtype}")
done

python tools/experiments/measure_sam2_hybrid_sizes.py \
  --sam2-checkpoint "${SAM2_CHECKPOINT}" \
  --student "tv21m=${TV21_CHECKPOINT}" \
  --student "tv11m=${TV11_CHECKPOINT}" \
  --student "tv5m=${TV5_CHECKPOINT}" \
  --out-dir "${REPORT_DIR}" \
  "${export_args[@]}"

echo
echo "Size report: ${REPORT_DIR}/sam2_hybrid_sizes.csv"
column -s, -t "${REPORT_DIR}/sam2_hybrid_sizes.csv" || true
