#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-${DATA_ROOT}/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_group_runtime.parquet}"
GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
LOG_ROOT="${LOG_ROOT:-/user-volume/remaining_experiment_logs/${LANE}}"
RECOVERY_DRY_RUN="${RECOVERY_DRY_RUN:-0}"
FAILED=()

export DATA_ROOT SAM2D_ROOT SAV_ROOT MANIFEST GPUS FULL_EVAL_GPUS
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
export FULL_EVAL=1
export SKIP_DONE=1
export MASK_ABLATION_SKIP_DONE=1
export MASK_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2"
export MASK_ABLATION_V1_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1"
export MASK_ABLATION_V1_COMPAT_ROOT="${MASK_ABLATION_V1_ROOT}"
export MASK_ABLATION_WANDB_PROJECT="sam2-mask-finetune-ablation-v1"
export MASK_ABLATION_BASE_CHECKPOINT="${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v2/stage1_encoder_task_2ep_v2/checkpoints/checkpoint.pt"

mkdir -p "${LOG_ROOT}"

run_logged() {
  local label="$1"
  shift
  local log
  log="${LOG_ROOT}/${label}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "================================================================"
  echo "Starting: ${label}"
  echo "Log: ${log}"
  echo "================================================================"
  local status=0
  if [[ "${RECOVERY_DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: $*" | tee "${log}"
  else
    "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
  fi
  echo "${label} status: ${status}"
  if [[ "${status}" -ne 0 ]]; then
    FAILED+=("${label}:${status}")
  fi
}

run_stage1_lane() {
  RUN_ROOT="${SAM2D_ROOT}/runs" \
    scripts/company/34_run_stage1_recovery_lane.sh "$1"
}

run_repvit() {
  RUN_ROOT="${SAM2D_ROOT}/runs/repvit_stage1_v1" \
  CHECKPOINT_ROOT="${SAM2D_ROOT}/checkpoints" \
  WANDB_PROJECT=sam2-distill-repvit-stage1-v1 \
  WANDB_RUN_ID='' \
    scripts/company/38_run_repvit_sam21l_stage1.sh all
}

run_mask_v1() {
  MASK_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1" \
    scripts/company/43_run_sam2_mask_finetune_ablation.sh "$1"
}

run_mask_v2() {
  MASK_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2" \
  WANDB_PROJECT=sam2-mask-finetune-ablation-v2 \
  WANDB_RUN_ID='' \
    scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh "$@"
}

run_report() {
  RUNS_ROOTS="${SAM2D_ROOT}/runs" \
  REPORT_DIR="${LOG_ROOT}/final_report" \
    scripts/company/45_report_all_experiments.sh
}

case "${LANE}" in
  node1)
    run_logged stage1_lane4 run_stage1_lane lane4
    for variant in decoder_lr2e7 decoder_lr5e7; do
      run_logged "mask_v1_${variant}" run_mask_v1 "${variant}"
    done
    for variant in A00_e2e_t4_box1_control A03_decmem_t4 A04_memory_t4; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  node2)
    run_logged stage1_lane5 run_stage1_lane lane5
    run_logged repvit_all run_repvit
    for variant in decoder_lr2e6 encdec_low_frozenbn; do
      run_logged "mask_v1_${variant}" run_mask_v1 "${variant}"
    done
    run_logged mask_v2_A10 run_mask_v2 run A10_e2e_t4_box0_imgkd
    ;;
  node3)
    for lane in lane1 lane2 lane3; do
      run_logged "stage1_${lane}" run_stage1_lane "${lane}"
    done
    for variant in encdec_low_trainbn decoder_lr5e7_boxonly; do
      run_logged "mask_v1_${variant}" run_mask_v1 "${variant}"
    done
    run_logged mask_v2_prepare_hardness run_mask_v2 prepare-hardness
    for variant in \
      A01_e2e_t4_box0 \
      A02_e2e_t4_official_prompt \
      A05_e2e_t8 \
      A06_e2e_t8_s4_t16_hard \
      A07_e2e_t4_warmup5 \
      A08_e2e_t4_gb8 \
      A09_e2e_t4_hard50x2 \
      A11_e2e_t4_box0_imgmemkd; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  *)
    echo "Usage: $0 {node1|node2|node3}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

run_logged all_experiment_report run_report

if [[ "${#FAILED[@]}" -eq 0 ]]; then
  STATUS=0
  echo "All jobs in ${LANE} completed successfully."
else
  STATUS=1
  echo "Failed jobs in ${LANE}: ${FAILED[*]}"
fi
echo "Lane logs: ${LOG_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
