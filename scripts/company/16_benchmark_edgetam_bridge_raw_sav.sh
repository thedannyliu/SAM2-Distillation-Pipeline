#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
RAW_SHARD_ROOT="${RAW_SHARD_ROOT:-/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030}"
PREP_ROOT="${PREP_ROOT:-${SAM2D_ROOT}/benchmarks/raw_sav030_prepared}"
OUT_ROOT="${OUT_ROOT:-${SAM2D_ROOT}/runs/raw_sav030_stage1_video_suite_10vid_artifacts}"

EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
EDGETAM_CFG="${EDGETAM_CFG:-configs/edgetam.yaml}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/edgetam/edgetam.pt}"
SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"

TV21_MSE_COS="${TV21_MSE_COS:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_mse_cos_5ep_v1/checkpoints/best.pt}"
TINYVIT21_CKPT="${TINYVIT21_CKPT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"

MAX_VIDEOS="${MAX_VIDEOS:-10}"
MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO:-2}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-1000}"
IMAGE_ARTIFACT_VIDEOS="${IMAGE_ARTIFACT_VIDEOS:-3}"
VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS:-3}"
VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES:-0}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-4}"
DEVICE="${DEVICE:-cuda}"
SKIP_DONE="${SKIP_DONE:-1}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh prepare
  scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh image
  scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh vos
  scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh artifacts
  scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh summarize
  scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh all

Runs two EdgeTAM-side benchmark entries into OUT_ROOT:
  - official_edgetam: official open-source EdgeTAM checkpoint.
  - tv21m_mse_cos_edgetam: Stage1 TV21M MSE+cos encoder patched into
    official EdgeTAM prompt/mask/memory modules.
EOF
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

prepare() {
  RAW_SHARD_ROOT="${RAW_SHARD_ROOT}" \
  PREP_ROOT="${PREP_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  MAX_VIDEOS="${MAX_VIDEOS}" \
  MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO}" \
  scripts/company/15_benchmark_raw_sav_shard_suite.sh prepare
}

run_image_official_edgetam() {
  local prompt="$1"
  local out_dir="${OUT_ROOT}/image/official_edgetam/${prompt}"
  if [[ "${SKIP_DONE}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "skip completed image/official_edgetam/${prompt}" >&2
    return 0
  fi
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind sam2 \
    --prompt-kind "${prompt}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --checkpoint "${EDGETAM_CHECKPOINT}" \
    --config "${EDGETAM_CFG}" \
    --sam2-root "${EDGETAM_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-dir "${out_dir}" \
    --max-objects "${MAX_IMAGE_OBJECTS}" \
    --save-artifacts 0 \
    --save-video-frame-artifacts "${IMAGE_ARTIFACT_VIDEOS}" \
    --device "${DEVICE}"
}

run_image_tv21m_mse_cos_edgetam() {
  local prompt="$1"
  local out_dir="${OUT_ROOT}/image/tv21m_mse_cos_edgetam/${prompt}"
  if [[ "${SKIP_DONE}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "skip completed image/tv21m_mse_cos_edgetam/${prompt}" >&2
    return 0
  fi
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind stage1-student \
    --prompt-kind "${prompt}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --checkpoint "${TV21_MSE_COS}" \
    --config "${EDGETAM_CFG}" \
    --sam2-checkpoint "${EDGETAM_CHECKPOINT}" \
    --tinyvit-checkpoint "${TINYVIT21_CKPT}" \
    --tinyvit-model-name tiny_vit_21m_512.dist_in22k_ft_in1k \
    --sam2-root "${EDGETAM_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-dir "${out_dir}" \
    --max-objects "${MAX_IMAGE_OBJECTS}" \
    --save-artifacts 0 \
    --save-video-frame-artifacts "${IMAGE_ARTIFACT_VIDEOS}" \
    --device "${DEVICE}"
}

image() {
  require_file "EdgeTAM root" "${EDGETAM_ROOT}"
  require_file "official EdgeTAM checkpoint" "${EDGETAM_CHECKPOINT}"
  require_file "TV21M MSE+cos Stage1 checkpoint" "${TV21_MSE_COS}"
  require_file "TinyViT-21M init checkpoint" "${TINYVIT21_CKPT}"
  for prompt in box point; do
    run_image_official_edgetam "${prompt}"
    run_image_tv21m_mse_cos_edgetam "${prompt}"
  done
}

run_vos_official_edgetam() {
  local prompt="$1"
  local model_out="${OUT_ROOT}/vos/official_edgetam/${prompt}"
  local pred_root="${model_out}/pred"
  if [[ "${SKIP_DONE}" == "1" && -f "${model_out}/eval_summary.json" ]]; then
    echo "skip completed vos/official_edgetam/${prompt}" >&2
    return 0
  fi
  mkdir -p "${pred_root}"
  python tools/eval/run_sam2_vos_prompt_dataset.py \
    --model-kind sam2 \
    --prompt-kind "${prompt}" \
    --sam2-root "${EDGETAM_ROOT}" \
    --sam2-cfg "${EDGETAM_CFG}" \
    --checkpoint "${EDGETAM_CHECKPOINT}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --out-dir "${pred_root}" \
    --video-list-file "${PREP_ROOT}/sav_train_benchmark.txt" \
    --device "${DEVICE}"
  python tools/eval/run_sav_evaluator.py \
    --evaluator "${SAM2_ROOT}/sav_dataset/sav_evaluator.py" \
    --gt-root "${PREP_ROOT}/Annotations_6fps" \
    --pred-root "${pred_root}" \
    --out-json "${model_out}/eval_summary.json" \
    --num-processes "${NUM_EVAL_PROCESSES}" \
    --do-not-skip-first-and-last-frame
}

run_vos_tv21m_mse_cos_edgetam() {
  local prompt="$1"
  local model_out="${OUT_ROOT}/vos/tv21m_mse_cos_edgetam/${prompt}"
  local pred_root="${model_out}/pred"
  if [[ "${SKIP_DONE}" == "1" && -f "${model_out}/eval_summary.json" ]]; then
    echo "skip completed vos/tv21m_mse_cos_edgetam/${prompt}" >&2
    return 0
  fi
  mkdir -p "${pred_root}"
  python tools/eval/run_sam2_vos_prompt_dataset.py \
    --model-kind stage1-student \
    --prompt-kind "${prompt}" \
    --sam2-root "${EDGETAM_ROOT}" \
    --sam2-cfg "${EDGETAM_CFG}" \
    --checkpoint "${TV21_MSE_COS}" \
    --sam2-checkpoint "${EDGETAM_CHECKPOINT}" \
    --tinyvit-checkpoint "${TINYVIT21_CKPT}" \
    --tinyvit-model-name tiny_vit_21m_512.dist_in22k_ft_in1k \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --out-dir "${pred_root}" \
    --video-list-file "${PREP_ROOT}/sav_train_benchmark.txt" \
    --device "${DEVICE}"
  python tools/eval/run_sav_evaluator.py \
    --evaluator "${SAM2_ROOT}/sav_dataset/sav_evaluator.py" \
    --gt-root "${PREP_ROOT}/Annotations_6fps" \
    --pred-root "${pred_root}" \
    --out-json "${model_out}/eval_summary.json" \
    --num-processes "${NUM_EVAL_PROCESSES}" \
    --do-not-skip-first-and-last-frame
}

vos() {
  require_file "EdgeTAM root" "${EDGETAM_ROOT}"
  require_file "official EdgeTAM checkpoint" "${EDGETAM_CHECKPOINT}"
  require_file "TV21M MSE+cos Stage1 checkpoint" "${TV21_MSE_COS}"
  require_file "TinyViT-21M init checkpoint" "${TINYVIT21_CKPT}"
  require_file "SAM2 evaluator root" "${SAM2_ROOT}"
  for prompt in box point; do
    run_vos_official_edgetam "${prompt}"
    run_vos_tv21m_mse_cos_edgetam "${prompt}"
  done
}

artifacts() {
  PREP_ROOT="${PREP_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS}" \
  VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES}" \
  scripts/company/15_benchmark_raw_sav_shard_suite.sh artifacts
}

summarize() {
  OUT_ROOT="${OUT_ROOT}" scripts/company/15_benchmark_raw_sav_shard_suite.sh summarize
}

case "${1:-}" in
  prepare) prepare ;;
  image) image ;;
  vos) vos ;;
  artifacts) artifacts ;;
  summarize) summarize ;;
  all)
    prepare
    image
    vos
    artifacts
    summarize
    ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
