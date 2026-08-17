#!/usr/bin/env bash

main() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
  cd "${repo_root}" || return 1

  local sam2d_root="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
  local sav_root="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
  local sam2_root="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
  local checkpoint="${SAM2_CHECKPOINT:-${sam2d_root}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
  local sam2_cfg="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
  local image_root="${IMAGE_ROOT:-${sav_root}/sav_val/JPEGImages_24fps}"
  local video_list="${VIDEO_LIST:-${sav_root}/sav_val/sav_val.txt}"
  local out_dir="${OUT_DIR:-${sam2d_root}/runs/sam21l_two_clock_reuse_v1/embedding_drift_20_v1}"
  local gpu="${GPU:-0}"
  local status=0 path

  for path in "${sam2_root}" "${checkpoint}" "${image_root}" "${video_list}"; do
    if [[ ! -e "${path}" ]]; then
      echo "[ERROR] Missing required path: ${path}" >&2
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "SAM2.1-L embedding drift status: ${status}"
    return "${status}"
  fi

  PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${gpu}" python \
    tools/experiments/analyze_sam21l_embedding_drift.py \
    --sam2-root "${sam2_root}" \
    --sam2-cfg "${sam2_cfg}" \
    --checkpoint "${checkpoint}" \
    --image-root "${image_root}" \
    --video-list-file "${video_list}" \
    --out-dir "${out_dir}" \
    --num-videos "${NUM_VIDEOS:-20}" \
    --seed "${SEED:-250107256}" \
    --fps "${FPS:-24}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --amp-dtype "${AMP_DTYPE:-bf16}" \
    --device cuda || status="$?"

  echo "SAM2.1-L embedding drift status: ${status}"
  echo "Summary: ${out_dir}/summary.json"
  echo "Figures: ${out_dir}/*.png"
  return "${status}"
}

main "$@"
