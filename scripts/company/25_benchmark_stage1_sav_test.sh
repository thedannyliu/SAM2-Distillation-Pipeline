#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT to the Stage 1 experiment name}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed Stage 1 run directory}"
MODEL_FAMILY="${MODEL_FAMILY:-sam2}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${RUN_DIR}/checkpoints/best.pt}"
SAV_SPLIT="${SAV_SPLIT:-sav_test}"
SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
SAV_SPLIT_ROOT="${SAV_SPLIT_ROOT:-${SAV_ROOT}/${SAV_SPLIT}}"
BENCH_ROOT="${BENCH_ROOT:-${RUN_DIR}/${SAV_SPLIT}_box_benchmark}"
AGGREGATE_CSV="${AGGREGATE_CSV:-$(dirname "${RUN_DIR}")/${SAV_SPLIT}_metrics.csv}"

IMAGE_ROOT="${IMAGE_ROOT:-${SAV_SPLIT_ROOT}/JPEGImages_24fps}"
ANN_ROOT="${ANN_ROOT:-${SAV_SPLIT_ROOT}/Annotations_6fps}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${SAV_SPLIT_ROOT}/${SAV_SPLIT}.txt}"

SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/group-volume/danny-dataset/sam2_distill/checkpoints}"
SAM2L_CONFIG="${SAM2L_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2L_CHECKPOINT="${SAM2L_CHECKPOINT:-${CHECKPOINT_ROOT}/sam2.1/sam2.1_hiera_large.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SAM3_ROOT="${SAM3_ROOT:-/user-volume/repo/facebookresearch-sam3}"
SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-/group-volume/danny-dataset/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt}"

MAX_VIDEOS="${MAX_VIDEOS:-${MAX_TEST_VIDEOS:-0}}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-0}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-16}"
CLEAN_PREDICTIONS="${CLEAN_PREDICTIONS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DEVICE="${DEVICE:-cuda}"

required_paths=("${STAGE1_CHECKPOINT}" "${IMAGE_ROOT}" "${ANN_ROOT}")
if [[ "${MODEL_FAMILY}" == "sam2" ]]; then
  required_paths+=("${SAM2L_CHECKPOINT}" "${TINYVIT_CHECKPOINT}" "${SAM2_ROOT}")
elif [[ "${MODEL_FAMILY}" == "sam31" ]]; then
  required_paths+=("${SAM31_CHECKPOINT}" "${SAM3_ROOT}" "${SAM2_ROOT}/sav_dataset/sav_evaluator.py")
else
  echo "MODEL_FAMILY must be sam2 or sam31; got ${MODEL_FAMILY}" >&2
  exit 2
fi
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing SAV test benchmark input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${BENCH_ROOT}"

image_out="${BENCH_ROOT}/image/${EXPERIMENT}/box"
if [[ "${SKIP_DONE}" != "1" || ! -f "${image_out}/summary.json" ]]; then
  image_args=(
    --prompt-kind box
    --image-root "${IMAGE_ROOT}"
    --ann-root "${ANN_ROOT}"
    --checkpoint "${STAGE1_CHECKPOINT}"
    --out-dir "${image_out}"
    --max-videos "${MAX_VIDEOS}"
    --max-objects "${MAX_IMAGE_OBJECTS}"
    --save-artifacts 0
    --device "${DEVICE}"
  )
  if [[ "${MODEL_FAMILY}" == "sam2" ]]; then
    image_args+=(
      --model-kind stage1-student
      --config "${SAM2L_CONFIG}"
      --sam2-checkpoint "${SAM2L_CHECKPOINT}"
      --tinyvit-checkpoint "${TINYVIT_CHECKPOINT}"
      --sam2-root "${SAM2_ROOT}"
    )
  else
    image_args+=(
      --model-kind sam31-stage1-student
      --config sam3.1_multiplex
      --sam3-root "${SAM3_ROOT}"
      --sam31-checkpoint "${SAM31_CHECKPOINT}"
    )
  fi
  python tools/benchmark/benchmark_sav_prompt_masks.py "${image_args[@]}"
else
  echo "skip completed SAV test image benchmark: ${EXPERIMENT}" >&2
fi

vos_out="${BENCH_ROOT}/vos/${EXPERIMENT}/box"
pred_root="${vos_out}/pred"
if [[ "${SKIP_DONE}" != "1" || ! -f "${vos_out}/eval_summary.json" ]]; then
  rm -rf "${pred_root}"
  mkdir -p "${pred_root}"
  if [[ "${MODEL_FAMILY}" == "sam2" ]]; then
    vos_program=tools/eval/run_sam2_vos_prompt_dataset.py
    vos_args=(
      --model-kind stage1-student
      --prompt-kind box
      --sam2-root "${SAM2_ROOT}"
      --sam2-cfg "${SAM2L_CONFIG}"
      --checkpoint "${STAGE1_CHECKPOINT}"
      --sam2-checkpoint "${SAM2L_CHECKPOINT}"
      --tinyvit-checkpoint "${TINYVIT_CHECKPOINT}"
      --image-root "${IMAGE_ROOT}"
      --ann-root "${ANN_ROOT}"
      --out-dir "${pred_root}"
      --device "${DEVICE}"
    )
  else
    vos_program=tools/eval/run_sam31_vos_prompt_dataset.py
    vos_args=(
      --sam3-root "${SAM3_ROOT}"
      --sam31-checkpoint "${SAM31_CHECKPOINT}"
      --checkpoint "${STAGE1_CHECKPOINT}"
      --image-root "${IMAGE_ROOT}"
      --ann-root "${ANN_ROOT}"
      --out-dir "${pred_root}"
      --device "${DEVICE}"
    )
  fi
  if [[ -f "${VIDEO_LIST_FILE}" ]]; then
    vos_args+=(--video-list-file "${VIDEO_LIST_FILE}")
  fi
  if [[ "${MAX_VIDEOS}" -gt 0 ]]; then
    vos_args+=(--max-videos "${MAX_VIDEOS}")
  fi
  python "${vos_program}" "${vos_args[@]}"
  cp "${pred_root}/summary.json" "${vos_out}/run_summary.json"
  python tools/eval/run_sav_evaluator.py \
    --evaluator "${SAM2_ROOT}/sav_dataset/sav_evaluator.py" \
    --gt-root "${ANN_ROOT}" \
    --pred-root "${pred_root}" \
    --out-json "${vos_out}/eval_summary.json" \
    --num-processes "${NUM_EVAL_PROCESSES}"
  if [[ "${CLEAN_PREDICTIONS}" == "1" ]]; then
    rm -rf "${pred_root}"
  fi
else
  echo "skip completed SAV test VOS benchmark: ${EXPERIMENT}" >&2
fi

python tools/benchmark/summarize_sav_benchmark_suite.py \
  --root "${BENCH_ROOT}" \
  --out-json "${BENCH_ROOT}/metrics.json" \
  --out-csv "${BENCH_ROOT}/metrics.csv" \
  --aggregate-csv "${AGGREGATE_CSV}"

echo "${SAV_SPLIT} metrics: ${BENCH_ROOT}/metrics.csv"
echo "Aggregate metrics: ${AGGREGATE_CSV}"
