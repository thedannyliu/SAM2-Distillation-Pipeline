#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
RAW_SHARD_ROOT="${RAW_SHARD_ROOT:-/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030}"
PREP_ROOT="${PREP_ROOT:-${SAM2D_ROOT}/benchmarks/raw_sav030_prepared}"
OUT_ROOT="${OUT_ROOT:-${SAM2D_ROOT}/runs/raw_sav030_benchmark_suite}"

SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2L_CKPT="${SAM2L_CKPT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
SAM2L_CONFIG="${SAM2L_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2B_CKPT="${SAM2B_CKPT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_base_plus.pt}"
SAM2B_CONFIG="${SAM2B_CONFIG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"

TINYVIT21_CKPT="${TINYVIT21_CKPT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
TINYVIT11_CKPT="${TINYVIT11_CKPT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors}"
TINYVIT5_CKPT="${TINYVIT5_CKPT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors}"

TV21_MSE="${TV21_MSE:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv21m_8gpu_b4_mse_only_5ep_v1/checkpoints/best.pt}"
TV21_MSE_COS="${TV21_MSE_COS:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_mse_cos_5ep_v1/checkpoints/best.pt}"
TV21_HIGHRES="${TV21_HIGHRES:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_highres_only_5ep_v1/checkpoints/best.pt}"
TV11_MSE="${TV11_MSE:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv11m_8gpu_b8_mse_only_5ep_v1/checkpoints/best.pt}"
TV11_MSE_COS="${TV11_MSE_COS:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv11m_4gpu_b16_mse_cos_5ep_v1/checkpoints/best.pt}"
TV5_MSE="${TV5_MSE:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_only_5ep_v1/checkpoints/best.pt}"
TV5_MSE_COS="${TV5_MSE_COS:-${SAM2D_ROOT}/runs/stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_cos_5ep_v1/checkpoints/best.pt}"

MAX_VIDEOS="${MAX_VIDEOS:-2}"
MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO:-2}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-200}"
SAVE_IMAGE_ARTIFACTS="${SAVE_IMAGE_ARTIFACTS:-0}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-4}"
VOS_TRACK_LATER="${VOS_TRACK_LATER:-1}"
DEVICE="${DEVICE:-cuda}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_DONE="${SKIP_DONE:-1}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/15_benchmark_raw_sav_shard_suite.sh prepare
  scripts/company/15_benchmark_raw_sav_shard_suite.sh image
  scripts/company/15_benchmark_raw_sav_shard_suite.sh vos
  scripts/company/15_benchmark_raw_sav_shard_suite.sh summarize
  scripts/company/15_benchmark_raw_sav_shard_suite.sh all

Notes:
  - image mode runs box and point prompts for SAM2.1-L/B and Stage1 TinyViT weights.
  - VOS mode runs point/box-initialized SAM2 memory tracking for SAM2.1-L/B
    and Stage1 TinyViT encoder patched into SAM2.1-L video predictor.
EOF
}

prepare() {
  DATA_ROOT="${DATA_ROOT}" \
  SAM2D_ROOT="${SAM2D_ROOT}" \
  RAW_SHARD_ROOT="${RAW_SHARD_ROOT}" \
  PREP_ROOT="${PREP_ROOT}" \
  MAX_VIDEOS="${MAX_VIDEOS}" \
  MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO}" \
  scripts/company/14_benchmark_raw_sav_shard_sam2.sh prepare
}

run_or_skip() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    return 0
  fi
  if [[ "${SKIP_MISSING}" == "1" ]]; then
    echo "skip ${label}: missing ${path}" >&2
    return 1
  fi
  echo "missing ${label}: ${path}" >&2
  exit 1
}

run_image_sam2() {
  local name="$1"
  local ckpt="$2"
  local cfg="$3"
  local prompt="$4"
  run_or_skip "${name}" "${ckpt}" || return 0
  if [[ "${SKIP_DONE}" == "1" && -f "${OUT_ROOT}/image/${name}/${prompt}/summary.json" ]]; then
    echo "skip completed image/${name}/${prompt}" >&2
    return 0
  fi
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind sam2 \
    --prompt-kind "${prompt}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --checkpoint "${ckpt}" \
    --config "${cfg}" \
    --sam2-root "${SAM2_ROOT}" \
    --out-dir "${OUT_ROOT}/image/${name}/${prompt}" \
    --max-objects "${MAX_IMAGE_OBJECTS}" \
    --save-artifacts "${SAVE_IMAGE_ARTIFACTS}" \
    --device "${DEVICE}"
}

run_image_stage1() {
  local name="$1"
  local ckpt="$2"
  local tinyvit_ckpt="$3"
  local model_name="$4"
  local prompt="$5"
  run_or_skip "${name}" "${ckpt}" || return 0
  run_or_skip "${name} tinyvit init" "${tinyvit_ckpt}" || return 0
  if [[ "${SKIP_DONE}" == "1" && -f "${OUT_ROOT}/image/${name}/${prompt}/summary.json" ]]; then
    echo "skip completed image/${name}/${prompt}" >&2
    return 0
  fi
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind stage1-student \
    --prompt-kind "${prompt}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --checkpoint "${ckpt}" \
    --config "${SAM2L_CONFIG}" \
    --sam2-checkpoint "${SAM2L_CKPT}" \
    --tinyvit-checkpoint "${tinyvit_ckpt}" \
    --tinyvit-model-name "${model_name}" \
    --sam2-root "${SAM2_ROOT}" \
    --out-dir "${OUT_ROOT}/image/${name}/${prompt}" \
    --max-objects "${MAX_IMAGE_OBJECTS}" \
    --save-artifacts "${SAVE_IMAGE_ARTIFACTS}" \
    --device "${DEVICE}"
}

image() {
  for prompt in box point; do
    run_image_sam2 sam2p1_l "${SAM2L_CKPT}" "${SAM2L_CONFIG}" "${prompt}"
    run_image_sam2 sam2p1_bplus "${SAM2B_CKPT}" "${SAM2B_CONFIG}" "${prompt}"
    run_image_stage1 tv21m_mse "${TV21_MSE}" "${TINYVIT21_CKPT}" tiny_vit_21m_512.dist_in22k_ft_in1k "${prompt}"
    run_image_stage1 tv21m_mse_cos "${TV21_MSE_COS}" "${TINYVIT21_CKPT}" tiny_vit_21m_512.dist_in22k_ft_in1k "${prompt}"
    run_image_stage1 tv21m_highres "${TV21_HIGHRES}" "${TINYVIT21_CKPT}" tiny_vit_21m_512.dist_in22k_ft_in1k "${prompt}"
    run_image_stage1 tv11m_mse "${TV11_MSE}" "${TINYVIT11_CKPT}" tiny_vit_11m_224.dist_in22k_ft_in1k "${prompt}"
    run_image_stage1 tv11m_mse_cos "${TV11_MSE_COS}" "${TINYVIT11_CKPT}" tiny_vit_11m_224.dist_in22k_ft_in1k "${prompt}"
    run_image_stage1 tv5m_mse "${TV5_MSE}" "${TINYVIT5_CKPT}" tiny_vit_5m_224.dist_in22k_ft_in1k "${prompt}"
    run_image_stage1 tv5m_mse_cos "${TV5_MSE_COS}" "${TINYVIT5_CKPT}" tiny_vit_5m_224.dist_in22k_ft_in1k "${prompt}"
  done
}

run_vos_sam2() {
  local name="$1"
  local ckpt="$2"
  local cfg="$3"
  local prompt="$4"
  run_or_skip "${name}" "${ckpt}" || return 0
  local model_out="${OUT_ROOT}/vos/${name}/${prompt}"
  local pred_root="${model_out}/pred"
  if [[ "${SKIP_DONE}" == "1" && -f "${model_out}/eval_summary.json" ]]; then
    echo "skip completed vos/${name}/${prompt}" >&2
    return 0
  fi
  mkdir -p "${pred_root}"
  python tools/eval/run_sam2_vos_prompt_dataset.py \
    --model-kind sam2 \
    --prompt-kind "${prompt}" \
    --sam2-root "${SAM2_ROOT}" \
    --sam2-cfg "${cfg}" \
    --checkpoint "${ckpt}" \
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

run_vos_stage1() {
  local name="$1"
  local ckpt="$2"
  local tinyvit_ckpt="$3"
  local model_name="$4"
  local prompt="$5"
  run_or_skip "${name}" "${ckpt}" || return 0
  run_or_skip "${name} tinyvit init" "${tinyvit_ckpt}" || return 0
  run_or_skip "sam2p1_l" "${SAM2L_CKPT}" || return 0
  local model_out="${OUT_ROOT}/vos/${name}/${prompt}"
  local pred_root="${model_out}/pred"
  if [[ "${SKIP_DONE}" == "1" && -f "${model_out}/eval_summary.json" ]]; then
    echo "skip completed vos/${name}/${prompt}" >&2
    return 0
  fi
  mkdir -p "${pred_root}"
  python tools/eval/run_sam2_vos_prompt_dataset.py \
    --model-kind stage1-student \
    --prompt-kind "${prompt}" \
    --sam2-root "${SAM2_ROOT}" \
    --sam2-cfg "${SAM2L_CONFIG}" \
    --checkpoint "${ckpt}" \
    --sam2-checkpoint "${SAM2L_CKPT}" \
    --tinyvit-checkpoint "${tinyvit_ckpt}" \
    --tinyvit-model-name "${model_name}" \
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
  for prompt in box point; do
    run_vos_sam2 sam2p1_l "${SAM2L_CKPT}" "${SAM2L_CONFIG}" "${prompt}"
    run_vos_sam2 sam2p1_bplus "${SAM2B_CKPT}" "${SAM2B_CONFIG}" "${prompt}"
    run_vos_stage1 tv21m_mse "${TV21_MSE}" "${TINYVIT21_CKPT}" tiny_vit_21m_512.dist_in22k_ft_in1k "${prompt}"
    run_vos_stage1 tv21m_mse_cos "${TV21_MSE_COS}" "${TINYVIT21_CKPT}" tiny_vit_21m_512.dist_in22k_ft_in1k "${prompt}"
    run_vos_stage1 tv21m_highres "${TV21_HIGHRES}" "${TINYVIT21_CKPT}" tiny_vit_21m_512.dist_in22k_ft_in1k "${prompt}"
    run_vos_stage1 tv11m_mse "${TV11_MSE}" "${TINYVIT11_CKPT}" tiny_vit_11m_224.dist_in22k_ft_in1k "${prompt}"
    run_vos_stage1 tv11m_mse_cos "${TV11_MSE_COS}" "${TINYVIT11_CKPT}" tiny_vit_11m_224.dist_in22k_ft_in1k "${prompt}"
    run_vos_stage1 tv5m_mse "${TV5_MSE}" "${TINYVIT5_CKPT}" tiny_vit_5m_224.dist_in22k_ft_in1k "${prompt}"
    run_vos_stage1 tv5m_mse_cos "${TV5_MSE_COS}" "${TINYVIT5_CKPT}" tiny_vit_5m_224.dist_in22k_ft_in1k "${prompt}"
  done
}

summarize() {
  python tools/benchmark/summarize_sav_benchmark_suite.py \
    --root "${OUT_ROOT}" \
    --out-json "${OUT_ROOT}/benchmark_summary.json" \
    --out-csv "${OUT_ROOT}/benchmark_summary.csv"
}

case "${1:-}" in
  prepare) prepare ;;
  image) image ;;
  vos) vos ;;
  summarize) summarize ;;
  all)
    prepare
    image
    vos
    summarize
    ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
