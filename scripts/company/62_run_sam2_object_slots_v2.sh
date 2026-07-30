#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
VARIANT="${2:-}"
VARIANTS=(
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
    if [[ -f "${candidate}/runs/sam2_object_slots_v1/MX2_slot8_decoder_kd_3ep/main/checkpoints/last.pt" ]]; then
      SAM2D_ROOT="${candidate}"
      break
    fi
  done
fi
SAM2D_ROOT="${SAM2D_ROOT:-/danny-dataset/sam2_distill}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_object_slots_v2}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-object-slots-v2}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_ROOT}/comparison}"
REFERENCE_LATENCY_DIR="${REFERENCE_LATENCY_DIR:-${SAM2D_ROOT}/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4}"
VARIANT_CSV="$(IFS=,; echo "${VARIANTS[*]}")"

describe() {
  echo "SAM2 learned object slots v2"
  echo "Budget: 5 epochs at T8 plus full val/test/latency; target 16-20 hours on 4xH100"
  echo "Parent: completed MX2 slot8 decoder checkpoint"
  echo "Selection: val/test quality retention >=95%, N1 FPS retention >=95%, N8 faster than runtime bucket"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}"
  printf '  %s\n' "${VARIANTS[@]}"
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  run)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Unknown v2 variant: ${VARIANT}" >&2
      STATUS=2
    else
      RUN_ROOT="${RUN_ROOT}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      scripts/company/61_run_sam2_object_slots.sh run "${VARIANT}"
      STATUS="$?"
    fi
    ;;
  summarize)
    EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
    EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh summarize || STATUS="$?"
    if [[ "${STATUS}" -eq 0 ]]; then
      python tools/benchmark/summarize_sam2_object_slots.py \
        --run-root "${RUN_ROOT}" \
        --reference-latency-dir "${REFERENCE_LATENCY_DIR}" \
        --out-dir "${RESULTS_DIR}" \
        --variants "${VARIANT_CSV}" \
        --reference-val-jf 72.4 \
        --reference-test-jf 74.7 \
        --min-quality-retention 0.95 \
        --min-learned-mask-iou 0.95 \
        --min-n1-fps-retention 0.95 || STATUS="$?"
    fi
    ;;
  *)
    echo "Usage: $0 {describe|run VARIANT|summarize}" >&2
    STATUS=2
    ;;
esac

echo "SAM2 object-slot v2 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${RUN_ROOT}/summary.csv"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
