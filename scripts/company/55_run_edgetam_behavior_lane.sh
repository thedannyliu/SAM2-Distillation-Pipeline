#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
case "${LANE}" in
  staged|joint|scratch) ;;
  *)
    echo "Usage: $0 {staged|joint|scratch}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
BEHAVIOR_ROOT="${EDGETAM_BEHAVIOR_ROOT:-${SAM2D_ROOT}/runs/edgetam_tinyvit21_behavior_v4}"
LOG_ROOT="${LOG_ROOT:-/user-volume/edgetam_behavior_logs/${LANE}}"
WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "${LOG_ROOT}" "${BEHAVIOR_ROOT}"

FAILED=()

run_job() {
  local name="$1"
  shift
  local log
  log="${LOG_ROOT}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "================================================================"
  echo "Starting: ${name}"
  echo "Log: ${log}"
  echo "================================================================"
  "$@" 2>&1 | tee -a "${log}"
  local status="${PIPESTATUS[0]}"
  echo "${name} status: ${status}"
  if [[ "${status}" -ne 0 ]]; then
    FAILED+=("${name}:${status}")
  fi
  return "${status}"
}

if [[ "${LANE}" == "staged" ]]; then
  run_job E1_a02_official_nonimage \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_BEHAVIOR_ROOT="${BEHAVIOR_ROOT}" \
      scripts/company/54_prepare_eval_edgetam_e1.sh all
  run_job D1_staged_image_align_1ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run D1_staged_image_align_1ep
  run_job D2_staged_temporal_2ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run D2_staged_temporal_2ep
  run_job D3_staged_t8_refine_1ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run D3_staged_t8_refine_1ep
elif [[ "${LANE}" == "joint" ]]; then
  run_job E1_prepare \
    env \
      WANDB_MODE=disabled \
      EDGETAM_BEHAVIOR_ROOT="${BEHAVIOR_ROOT}" \
      scripts/company/54_prepare_eval_edgetam_e1.sh prepare
  run_job J1_joint_behavior_2ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run J1_joint_behavior_2ep
  run_job J2_joint_temporal_refine_1ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run J2_joint_temporal_refine_1ep
  run_job J3_joint_t8_refine_1ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run J3_joint_t8_refine_1ep
else
  run_job S0_scratch_temporal_task_2ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run S0_scratch_temporal_task_2ep
  run_job S1_scratch_behavior_2ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run S1_scratch_behavior_2ep
  run_job S2_scratch_t8_refine_1ep \
    env \
      WANDB_MODE="${WANDB_MODE}" \
      GPUS="${GPUS:-0,1,2,3}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-0,1,2,3}" \
      EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh \
        run S2_scratch_t8_refine_1ep
fi

EDGETAM_MEMORY_ROOT="${BEHAVIOR_ROOT}" \
EDGETAM_MEMORY_SUMMARY_CSV="${BEHAVIOR_ROOT}/summary.csv" \
  scripts/company/49_run_edgetam_memory_ablation.sh summarize
SUMMARY_STATUS="$?"
if [[ "${SUMMARY_STATUS}" -ne 0 ]]; then
  FAILED+=("summary:${SUMMARY_STATUS}")
fi

echo
echo "EdgeTAM behavior lane: ${LANE}"
echo "Run root: ${BEHAVIOR_ROOT}"
echo "Summary: ${BEHAVIOR_ROOT}/summary.csv"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed jobs: ${FAILED[*]}"
  return 1 2>/dev/null || exit 1
fi
echo "Lane status: 0"
return 0 2>/dev/null || exit 0
