#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:?Set EXPERIMENT to the Stage 1 experiment name}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed Stage 1 run directory}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${RUN_DIR}/checkpoints/best.pt}"
BENCH_ROOT="${BENCH_ROOT:-${RUN_DIR}/sav_test_box_benchmark}"
AGGREGATE_CSV="${AGGREGATE_CSV:-$(dirname "${RUN_DIR}")/sav_test_metrics.csv}"

SAV_TEST_ROOT="${SAV_TEST_ROOT:-/mnt/data/danny-dataset/SA-V/sav_test}"
IMAGE_ROOT="${IMAGE_ROOT:-${SAV_TEST_ROOT}/JPEGImages_24fps}"
ANN_ROOT="${ANN_ROOT:-${SAV_TEST_ROOT}/Annotations_6fps}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${SAV_TEST_ROOT}/sav_test.txt}"

SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/group-volume/danny-dataset/sam2_distill/checkpoints}"
SAM2L_CONFIG="${SAM2L_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2L_CHECKPOINT="${SAM2L_CHECKPOINT:-${CHECKPOINT_ROOT}/sam2.1/sam2.1_hiera_large.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"

MAX_TEST_VIDEOS="${MAX_TEST_VIDEOS:-0}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-0}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-16}"
CLEAN_PREDICTIONS="${CLEAN_PREDICTIONS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DEVICE="${DEVICE:-cuda}"

for path in "${STAGE1_CHECKPOINT}" "${IMAGE_ROOT}" "${ANN_ROOT}" "${SAM2L_CHECKPOINT}" "${TINYVIT_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing SAV test benchmark input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${BENCH_ROOT}"

image_out="${BENCH_ROOT}/image/${EXPERIMENT}/box"
if [[ "${SKIP_DONE}" != "1" || ! -f "${image_out}/summary.json" ]]; then
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind stage1-student \
    --prompt-kind box \
    --image-root "${IMAGE_ROOT}" \
    --ann-root "${ANN_ROOT}" \
    --checkpoint "${STAGE1_CHECKPOINT}" \
    --config "${SAM2L_CONFIG}" \
    --sam2-checkpoint "${SAM2L_CHECKPOINT}" \
    --tinyvit-checkpoint "${TINYVIT_CHECKPOINT}" \
    --sam2-root "${SAM2_ROOT}" \
    --out-dir "${image_out}" \
    --max-videos "${MAX_TEST_VIDEOS}" \
    --max-objects "${MAX_IMAGE_OBJECTS}" \
    --save-artifacts 0 \
    --device "${DEVICE}"
else
  echo "skip completed SAV test image benchmark: ${EXPERIMENT}" >&2
fi

vos_out="${BENCH_ROOT}/vos/${EXPERIMENT}/box"
pred_root="${vos_out}/pred"
if [[ "${SKIP_DONE}" != "1" || ! -f "${vos_out}/eval_summary.json" ]]; then
  rm -rf "${pred_root}"
  mkdir -p "${pred_root}"
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
  if [[ -f "${VIDEO_LIST_FILE}" ]]; then
    vos_args+=(--video-list-file "${VIDEO_LIST_FILE}")
  fi
  if [[ "${MAX_TEST_VIDEOS}" -gt 0 ]]; then
    vos_args+=(--max-videos "${MAX_TEST_VIDEOS}")
  fi
  python tools/eval/run_sam2_vos_prompt_dataset.py "${vos_args[@]}"
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

echo "SAV test metrics: ${BENCH_ROOT}/metrics.csv"
echo "Aggregate metrics: ${AGGREGATE_CSV}"
