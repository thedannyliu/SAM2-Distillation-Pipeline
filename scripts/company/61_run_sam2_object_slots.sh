#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
VARIANT="${2:-}"
VARIANTS=(
  MX1_slot4_decoder_kd_3ep
  MX2_slot8_decoder_kd_3ep
  MX3_slot4_sharedkv_kd_3ep
  MX4_slot8_sharedkv_kd_3ep
  MX5_slot8_decoder_t8_logits2_5ep
  MX6_slot8_sharedkv_t8_mem1_5ep
  MX7_slot8_sharedkv_t8_mem4_5ep
  MX8_slot8_sharedkv_t8_mem1_logits4_5ep
)

is_variant() {
  local candidate="$1" item
  for item in "${VARIANTS[@]}"; do
    [[ "${candidate}" == "${item}" ]] && return 0
  done
  return 1
}

if [[ -z "${SAM2D_ROOT:-}" ]]; then
  for candidate in \
    /danny-dataset/sam2_distill \
    /group-volume/danny-dataset/sam2_distill \
    /mnt/data/danny-dataset/sam2_distill; do
    if [[ -f "${candidate}/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt" ]]; then
      SAM2D_ROOT="${candidate}"
      break
    fi
  done
fi
SAM2D_ROOT="${SAM2D_ROOT:-/danny-dataset/sam2_distill}"

if [[ -z "${SAV_ROOT:-}" ]]; then
  for candidate in \
    /danny-dataset/SA-V \
    /group-volume/danny-dataset/SA-V \
    /mnt/data/danny-dataset/SA-V; do
    if [[ -f "${candidate}/sav_val/sav_val.txt" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/danny-dataset/SA-V}"

GPUS="${GPUS:-0,1,2,3}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "[ERROR] Object-slot training requires four GPUs: ${GPUS}" >&2
  return 2 2>/dev/null || exit 2
fi

SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  case "${VARIANT}" in
    MX5_*|MX6_*|MX7_*|MX8_*)
      RUN_ROOT="${SAM2D_ROOT}/runs/sam2_object_slots_v2"
      ;;
    *)
      RUN_ROOT="${SAM2D_ROOT}/runs/sam2_object_slots_v1"
      ;;
  esac
fi
SUMMARY_CSV="${SUMMARY_CSV:-${RUN_ROOT}/summary.csv}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-object-slots-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
LATENCY_GPU="${LATENCY_GPU:-${GPU_ARRAY[0]}}"
LATENCY_REPETITIONS="${LATENCY_REPETITIONS:-3}"
LATENCY_MAX_VIDEOS="${LATENCY_MAX_VIDEOS:-16}"
LATENCY_VERIFY_FRAMES="${LATENCY_VERIFY_FRAMES:-128}"
LATENCY_COHORT_DIR="${RUN_ROOT}/cohorts/val_dense8"
LATENCY_COHORT="${LATENCY_COHORT_DIR}/cohort.txt"
if [[ -z "${REFERENCE_LATENCY_DIR:-}" ]]; then
  for candidate in \
    "${SAM2D_ROOT}" \
    /danny-dataset/sam2_distill \
    /group-volume/danny-dataset/sam2_distill \
    /mnt/data/danny-dataset/sam2_distill; do
    candidate="${candidate}/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4"
    if [[ -f "${candidate}/aggregate.csv" ]]; then
      REFERENCE_LATENCY_DIR="${candidate}"
      break
    fi
  done
fi
REFERENCE_LATENCY_DIR="${REFERENCE_LATENCY_DIR:-${SAM2D_ROOT}/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_ROOT}/comparison}"
REFERENCE_VAL_JF="${REFERENCE_VAL_JF:-72.4}"
REFERENCE_TEST_JF="${REFERENCE_TEST_JF:-74.7}"
MIN_QUALITY_RETENTION="${MIN_QUALITY_RETENTION:-0.95}"
MIN_LEARNED_MASK_IOU="${MIN_LEARNED_MASK_IOU:-0.95}"
MIN_N1_FPS_RETENTION="${MIN_N1_FPS_RETENTION:-0.95}"

slot_count() {
  case "$1" in
    MX1_*|MX3_*) echo 4 ;;
    MX2_*|MX4_*|MX5_*|MX6_*|MX7_*|MX8_*) echo 8 ;;
    *) return 2 ;;
  esac
}

describe() {
  echo "SAM2 learned object slots v1"
  echo "Question: can learned fixed slots preserve >=95% quality while reducing N=8 latency?"
  echo "MX1/MX2: learned slot decoder with standard per-object memory K/V"
  echo "MX3/MX4: learned slot decoder with bucket-shared memory K/V"
  echo "Capacities: 4 and 8 objects"
  echo "Each node: 3-epoch dense8 train -> full val -> full test -> isolated latency"
  echo "Training GPUs: ${GPUS}; latency GPU: ${LATENCY_GPU}"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}; mode ${WANDB_MODE}"
  echo "Reference latency: ${REFERENCE_LATENCY_DIR}"
  echo "Quality gate: val/test J&F retention >= ${MIN_QUALITY_RETENTION}"
}

ensure_latency_cohort() {
  mkdir -p "${LATENCY_COHORT_DIR}"
  exec 8>"${LATENCY_COHORT_DIR}/.lock" || return 1
  flock 8 || return 1
  if [[ ! -s "${LATENCY_COHORT}" ]]; then
    python tools/data/audit_vos_object_density.py \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${SAV_ROOT}/sav_val/sav_val.txt" \
      --out-dir "${LATENCY_COHORT_DIR}" \
      --min-shared-objects 8 \
      --max-cohort-videos "${LATENCY_MAX_VIDEOS}" \
      --seed 310107256 || return 1
  fi
  flock -u 8
}

run_latency() {
  local slots run_dir out_dir
  slots="$(slot_count "${VARIANT}")" || return 1
  run_dir="${RUN_ROOT}/${VARIANT}/main"
  out_dir="${run_dir}/multiobject_latency/point_n1-2-4-8"
  ensure_latency_cohort || return 1
  if [[ "${SKIP_DONE:-1}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skip completed latency: ${VARIANT}"
    cat "${out_dir}/aggregate.csv"
    return 0
  fi
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${LATENCY_GPU}" \
  PYTHONPATH="${REPO_ROOT}:${EDGETAM_ROOT}:${SAM2_TRAINING_ROOT}:${PYTHONPATH:-}" \
    python tools/benchmark/benchmark_sam2_multiobject_scaling.py \
      --model-kind edgetam-trainer \
      --prompt-kind point \
      --sam2-root "${EDGETAM_ROOT}" \
      --sam2-cfg "${run_dir}/resolved_config.yaml" \
      --checkpoint "${run_dir}/checkpoints/last.pt" \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${LATENCY_COHORT}" \
      --out-dir "${out_dir}" \
      --object-counts 1,2,4,8 \
      --max-videos "${LATENCY_MAX_VIDEOS}" \
      --repetitions "${LATENCY_REPETITIONS}" \
      --warmup-videos 1 \
      --execution-mode bucket \
      --bucket-size "${slots}" \
      --bucket-min-objects 4 \
      --verify-bucket-frames "${LATENCY_VERIFY_FRAMES}" \
      --seed 310107256 \
      --device cuda \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${VARIANT}_latency" \
      --wandb-mode "${WANDB_MODE}"
}

run_variant() {
  local slots
  slots="$(slot_count "${VARIANT}")" || return 1
  export SAM2D_ROOT SAV_ROOT GPUS SAM2_TRAINING_ROOT EDGETAM_ROOT
  export EDGETAM_MEMORY_ROOT="${RUN_ROOT}"
  export EDGETAM_MEMORY_SUMMARY_CSV="${SUMMARY_CSV}"
  export WANDB_PROJECT WANDB_MODE
  export VOS_EXECUTION_MODE=bucket
  export VOS_BUCKET_SIZE="${slots}"
  export VOS_BUCKET_MIN_OBJECTS=4
  export EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE:-1}"
  scripts/company/49_run_edgetam_memory_ablation.sh run "${VARIANT}" || return 1
  run_latency || return 1
  scripts/company/49_run_edgetam_memory_ablation.sh summarize || return 1
  touch "${RUN_ROOT}/${VARIANT}/.pipeline_complete"
  echo "Completed: ${VARIANT}"
  echo "Summary: ${SUMMARY_CSV}"
}

summarize_results() {
  export EDGETAM_MEMORY_ROOT="${RUN_ROOT}"
  export EDGETAM_MEMORY_SUMMARY_CSV="${SUMMARY_CSV}"
  scripts/company/49_run_edgetam_memory_ablation.sh summarize || return 1
  if [[ ! -f "${REFERENCE_LATENCY_DIR}/aggregate.csv" ]]; then
    echo "[WARN] Reference latency is missing: ${REFERENCE_LATENCY_DIR}/aggregate.csv"
    echo "Set REFERENCE_LATENCY_DIR to the selected persistent-bucket result."
    return 0
  fi
  python tools/benchmark/summarize_sam2_object_slots.py \
    --run-root "${RUN_ROOT}" \
    --reference-latency-dir "${REFERENCE_LATENCY_DIR}" \
    --out-dir "${RESULTS_DIR}" \
    --reference-val-jf "${REFERENCE_VAL_JF}" \
    --reference-test-jf "${REFERENCE_TEST_JF}" \
    --min-quality-retention "${MIN_QUALITY_RETENTION}" \
    --min-learned-mask-iou "${MIN_LEARNED_MASK_IOU}" \
    --min-n1-fps-retention "${MIN_N1_FPS_RETENTION}"
}

describe
case "${ACTION}" in
  describe)
    ;;
  run)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Unknown object-slot variant: ${VARIANT}" >&2
      return 2 2>/dev/null || exit 2
    fi
    run_variant
    ;;
  summarize)
    summarize_results
    ;;
  *)
    echo "Usage: $0 {describe|run VARIANT|summarize}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

STATUS="$?"
echo "SAM2 object-slot status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY_CSV}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
