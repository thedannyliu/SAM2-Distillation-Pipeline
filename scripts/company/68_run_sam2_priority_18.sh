#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
NODE="${2:-}"
RUNNER="scripts/company/67_run_sam2_full_data_50.sh"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_full_data_50_v1}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_ROOT}/priority18_comparison}"
REFERENCE_LATENCY_DIR="${REFERENCE_LATENCY_DIR:-${SAM2D_ROOT}/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4}"

queue_for_node() {
  case "$1" in
    1) echo "FD46_mem4_t8_decmem_8ep FD47_mem2_t8_decmem_8ep FD48_mem2_t8_joint_logits2_8ep" ;;
    2) echo "FD04_tv21_t8_joint_logits1_8ep FD49_edgetam2_temporal_logits2_8ep FD50_edgetam2_joint_img_logits2_8ep" ;;
    3) echo "FD01_tv21_t4_decmem_5ep FD03_tv21_t8_logits1_8ep FD05_tv21_t12_joint_mem025_logits2_8ep" ;;
    4) echo "FD11_sharedkv_all_t8_r16_8ep FD12_sharedkv_dense4_t8_r16_8ep FD13_sharedkv_dense8_t8_r16_8ep" ;;
    5) echo "FD18_sharedkv_r8_ptr8_8ep FD19_sharedkv_r16_ptr8_8ep FD20_sharedkv_r32_ptr8_8ep" ;;
    6) echo "FD21_sharedkv_r16_mem025_logits2_8ep FD25_sharedkv_r16_mem100_logits4_8ep FD30_sharedkv_r16_ptr8_recency050_obj025_8ep" ;;
    *) return 2 ;;
  esac
}

all_variants() {
  local node
  for node in 1 2 3 4 5 6; do
    queue_for_node "${node}"
  done
}

describe() {
  echo "SAM2 priority-18 full-data suite"
  echo "Six independent nodes; four H100s per node; three sequential long runs"
  echo "Nodes 1-3: memory and EdgeTAM-style temporal compression"
  echo "Nodes 4-6: multiplex speed/quality Pareto"
  echo "Existing full-data run root: ${RUN_ROOT}"
  local node
  for node in 1 2 3 4 5 6; do
    echo "Node ${node}: $(queue_for_node "${node}")"
  done
}

run_node() {
  local node="$1" variant status queue
  local -a failures=()
  queue="$(queue_for_node "${node}")" || {
    echo "[ERROR] Node must be 1 through 6" >&2
    return 2
  }
  "${RUNNER}" audit || return $?
  for variant in ${queue}; do
    echo "===== Priority node ${node}: ${variant} ====="
    "${RUNNER}" run "${variant}"
    status="$?"
    echo "${variant} status: ${status}"
    if [[ "${status}" -ne 0 ]]; then
      failures+=("${variant}:${status}")
      echo "[CONTINUE] Failure recorded; continuing to the next experiment." >&2
    fi
  done
  if [[ "${#failures[@]}" -gt 0 ]]; then
    echo "===== Priority node ${node} failures =====" >&2
    printf '%s\n' "${failures[@]}" >&2
    return 1
  fi
  echo "Priority node ${node}: all three experiments completed."
}

summarize() {
  local variants_csv
  variants_csv="$(all_variants | tr ' ' '\n' | sed '/^$/d' | paste -sd, -)"
  python tools/benchmark/summarize_sam2_multiplex_screen.py \
    --run-root "${RUN_ROOT}" \
    --reference-latency-dir "${REFERENCE_LATENCY_DIR}" \
    --out-dir "${RESULTS_DIR}" \
    --variants "${variants_csv}" \
    --gate-videos "${GATE_VIDEOS:-64}" \
    --min-quality-retention 0.95 \
    --min-learned-mask-iou 0.95 \
    --max-promotions 10
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  audit)
    "${RUNNER}" audit || STATUS="$?"
    ;;
  node)
    run_node "${NODE}" || STATUS="$?"
    ;;
  summarize)
    summarize || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|audit|node 1|node 2|node 3|node 4|node 5|node 6|summarize}" >&2
    STATUS=2
    ;;
esac

echo "SAM2 priority-18 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "Results: ${RESULTS_DIR}/screen_results.md"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
