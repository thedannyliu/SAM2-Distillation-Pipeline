#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
TARGET="${2:-}"
VARIANTS=(
  MX13_slot8_r2_mean_screen3ep
  MX14_slot8_r4_mean_screen3ep
  MX15_slot8_r8_mean_screen3ep
  MX16_slot8_r16_mean_screen3ep
  MX17_slot8_r8_mean_ptr4_screen3ep
  MX18_slot8_r8_mean_ptr8_screen3ep
  MX19_slot8_r8_latest_ptr8_screen3ep
  MX20_slot8_r8_recency050_ptr8_screen3ep
  MX21_slot8_r8_recency025_ptr8_screen3ep
  MX22_slot8_r8_recency075_ptr8_screen3ep
  MX23_slot4_r8_mean_ptr8_screen3ep
  MX24_slot6_r8_mean_ptr8_screen3ep
  MX25_slot8_r8_mean_ptr8_objkd025_screen3ep
  MX26_slot8_r8_mean_ptr8_objkd100_screen3ep
  MX27_slot8_min2_r8_mean_ptr8_screen3ep
  MX28_slot8_min3_r8_mean_ptr8_screen3ep
)
QUEUES=(
  "MX13_slot8_r2_mean_screen3ep MX14_slot8_r4_mean_screen3ep"
  "MX15_slot8_r8_mean_screen3ep MX16_slot8_r16_mean_screen3ep"
  "MX17_slot8_r8_mean_ptr4_screen3ep MX18_slot8_r8_mean_ptr8_screen3ep"
  "MX19_slot8_r8_latest_ptr8_screen3ep MX20_slot8_r8_recency050_ptr8_screen3ep"
  "MX21_slot8_r8_recency025_ptr8_screen3ep MX22_slot8_r8_recency075_ptr8_screen3ep"
  "MX23_slot4_r8_mean_ptr8_screen3ep MX24_slot6_r8_mean_ptr8_screen3ep"
  "MX25_slot8_r8_mean_ptr8_objkd025_screen3ep MX26_slot8_r8_mean_ptr8_objkd100_screen3ep"
  "MX27_slot8_min2_r8_mean_ptr8_screen3ep MX28_slot8_min3_r8_mean_ptr8_screen3ep"
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
    if [[ -f "${candidate}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/checkpoints/last.pt" ]]; then
      SAM2D_ROOT="${candidate}"
      break
    fi
  done
fi
SAM2D_ROOT="${SAM2D_ROOT:-/danny-dataset/sam2_distill}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_multiplex_overnight_v4}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-multiplex-overnight-v4}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_ROOT}/comparison}"
MX5_RUN="${SAM2D_ROOT}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main"
REFERENCE_LATENCY_DIR="${REFERENCE_LATENCY_DIR:-${SAM2D_ROOT}/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4}"
VARIANT_CSV="$(IFS=,; echo "${VARIANTS[*]}")"

describe() {
  echo "SAM2 multiplex overnight screen v4"
  echo "Budget: eight independent nodes, four H100s per node, two sequential experiments per node"
  echo "Common: MX5 initializer, shared K/V, T8, 3 epochs, fixed 32-video quality gate, one-repeat N=1/2/4/8 latency"
  echo "Promotion: mini-val J&F retention >=95%, mask IoU >=0.95, N=8 faster than runtime bucket"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}"
  local index
  for index in "${!QUEUES[@]}"; do
    echo "Node $((index + 1)): ${QUEUES[index]}"
  done
}

screen_variant() {
  local variant="$1"
  SAM2D_ROOT="${SAM2D_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  TASK_EPOCHS_OVERRIDE=3 \
  EDGETAM_EVAL_MODE=gate \
  EDGETAM_GATE_ENFORCE=0 \
  EDGETAM_GATE_MAX_VIDEOS=32 \
  EDGETAM_GATE_MIN_JF=0 \
  EDGETAM_GATE_MIN_JF_RETENTION=0.95 \
  EDGETAM_GATE_MAX_JF_DROP=100 \
  EDGETAM_GATE_MAX_IMAGE_DROP=0.01 \
  EDGETAM_GATE_REFERENCE_CHECKPOINT="${MX5_RUN}/checkpoints/last.pt" \
  EDGETAM_GATE_REFERENCE_CONFIG="${MX5_RUN}/resolved_config.yaml" \
  EDGETAM_GATE_REFERENCE_TAG=mx5 \
  LATENCY_REPETITIONS=1 \
  LATENCY_MAX_VIDEOS=1 \
  LATENCY_VERIFY_FRAMES=32 \
  PIPELINE_COMPLETE_MARKER=.screen_complete \
    scripts/company/61_run_sam2_object_slots.sh run "${variant}"
}

promote_variant() {
  local variant="$1"
  SAM2D_ROOT="${SAM2D_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  TASK_EPOCHS_OVERRIDE=5 \
  EDGETAM_EVAL_MODE=full \
  LATENCY_REPETITIONS=3 \
  LATENCY_MAX_VIDEOS=16 \
  LATENCY_VERIFY_FRAMES=128 \
  PIPELINE_COMPLETE_MARKER=.pipeline_complete \
  SKIP_DONE=0 \
    scripts/company/61_run_sam2_object_slots.sh run "${variant}"
}

run_queue() {
  local node="$1" variant status
  if [[ ! "${node}" =~ ^[1-8]$ ]]; then
    echo "[ERROR] Queue number must be 1 through 8" >&2
    return 2
  fi
  for variant in ${QUEUES[$((node - 1))]}; do
    echo "===== Node ${node}: ${variant} ====="
    screen_variant "${variant}"
    status="$?"
    echo "${variant} status: ${status}"
    if [[ "${status}" -ne 0 ]]; then
      return "${status}"
    fi
  done
}

summarize() {
  python tools/benchmark/summarize_sam2_multiplex_screen.py \
    --run-root "${RUN_ROOT}" \
    --reference-latency-dir "${REFERENCE_LATENCY_DIR}" \
    --out-dir "${RESULTS_DIR}" \
    --variants "${VARIANT_CSV}" \
    --gate-videos 32 \
    --min-quality-retention 0.95 \
    --min-learned-mask-iou 0.95 \
    --max-promotions 4
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  run)
    if ! is_variant "${TARGET}"; then
      echo "[ERROR] Unknown v4 variant: ${TARGET}" >&2
      STATUS=2
    else
      screen_variant "${TARGET}" || STATUS="$?"
    fi
    ;;
  queue)
    run_queue "${TARGET}" || STATUS="$?"
    ;;
  promote)
    if ! is_variant "${TARGET}"; then
      echo "[ERROR] Unknown v4 variant: ${TARGET}" >&2
      STATUS=2
    else
      promote_variant "${TARGET}" || STATUS="$?"
    fi
    ;;
  summarize)
    summarize || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|run VARIANT|queue NODE|promote VARIANT|summarize}" >&2
    STATUS=2
    ;;
esac

echo "SAM2 multiplex overnight v4 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "Results: ${RESULTS_DIR}/screen_results.md"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
