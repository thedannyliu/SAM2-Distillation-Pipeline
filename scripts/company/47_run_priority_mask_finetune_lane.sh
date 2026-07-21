#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
case "${LANE}" in
  priority1|priority2|priority3) ;;
  *)
    echo "Usage: $0 {priority1|priority2|priority3}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-${DATA_ROOT}/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_group_runtime.parquet}"
GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
LOG_ROOT="${LOG_ROOT:-/user-volume/priority_mask_finetune_logs/${LANE}}"
PRIORITY_DRY_RUN="${PRIORITY_DRY_RUN:-0}"
ABLATION_ROOT="${MASK_ABLATION_ROOT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2}"
FAILED=()

if [[ "${WANDB_MODE:-online}" != "online" ]]; then
  echo "WANDB_MODE=online is required; got ${WANDB_MODE}" >&2
  return 2 2>/dev/null || exit 2
fi

export DATA_ROOT SAM2D_ROOT SAV_ROOT MANIFEST GPUS FULL_EVAL_GPUS
export WANDB_MODE=online
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
export MASK_ABLATION_ROOT="${ABLATION_ROOT}"
export MASK_ABLATION_V1_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1"
export MASK_ABLATION_V1_COMPAT_ROOT="${MASK_ABLATION_V1_ROOT}"
export MASK_ABLATION_SKIP_DONE=1

mkdir -p "${LOG_ROOT}"

gpu_count="$(python - "${GPUS}" <<'PY'
import sys
print(len([value for value in sys.argv[1].split(",") if value.strip()]))
PY
)"
if [[ "${gpu_count}" -ne 4 ]]; then
  echo "${LANE} requires exactly 4 GPUs; got GPUS=${GPUS}" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ "${PRIORITY_DRY_RUN}" != "1" ]]; then
  python - <<'PY' || return 1 2>/dev/null || exit 1
import wandb

viewer = wandb.Api(timeout=30).viewer
identity = viewer.get("username") or viewer.get("entity") or "authenticated user"
print(f"W&B online preflight: PASS | {identity}", flush=True)
PY
fi

run_logged() {
  local label="$1"
  shift
  local log status=0
  log="${LOG_ROOT}/${label}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "================================================================"
  echo "Starting: ${label}"
  echo "Log: ${log}"
  echo "================================================================"
  if [[ "${PRIORITY_DRY_RUN}" == "1" ]]; then
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

verify_task_stage() {
  local stage_dir="$1" unexpected
  [[ -s "${stage_dir}/checkpoints/checkpoint.pt" && \
     -s "${stage_dir}/checkpoints/stage.pt" ]] || {
    echo "missing task last/export checkpoint pair: ${stage_dir}" >&2
    return 1
  }
  unexpected="$(find "${stage_dir}/checkpoints" -maxdepth 1 -type f -name '*.pt' \
    ! -name checkpoint.pt ! -name stage.pt -print)"
  if [[ -n "${unexpected}" ]]; then
    echo "unexpected task checkpoints (only checkpoint.pt/stage.pt are allowed):" >&2
    echo "${unexpected}" >&2
    return 1
  fi
  echo "checkpoint retention: PASS | checkpoint.pt=last, stage.pt=best/export | ${stage_dir}"
}

verify_wandb_training() {
  local stage_dir="$1"
  python tools/train/verify_wandb_history.py \
    --run-file "${stage_dir}/wandb/wandb_run.json" \
    --metric train/loss_total
}

run_variant() {
  local variant="$1" stage_dir
  WANDB_PROJECT=sam2-mask-finetune-ablation-v2 \
  WANDB_RUN_ID='' \
    scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh \
      run "${variant}" || return 1
  stage_dir="${ABLATION_ROOT}/${variant}/main"
  verify_task_stage "${stage_dir}" || return 1
  verify_wandb_training "${stage_dir}" || return 1
  if [[ "${variant}" == "A06_e2e_t8_s4_t16_hard" ]]; then
    stage_dir="${ABLATION_ROOT}/${variant}/refine_t16"
    verify_task_stage "${stage_dir}" || return 1
    verify_wandb_training "${stage_dir}" || return 1
  fi
}

prepare_hardness() {
  scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh prepare-hardness
}

write_summaries() {
  scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh summarize || return 1
  RUNS_ROOTS="${SAM2D_ROOT}/runs" \
  REPORT_DIR="${LOG_ROOT}/final_report" \
    scripts/company/45_report_all_experiments.sh
}

case "${LANE}" in
  priority1)
    for variant in \
      A01_e2e_t4_box0 \
      A00_e2e_t4_box1_control \
      A02_e2e_t4_official_prompt; do
      run_logged "mask_v2_${variant}" run_variant "${variant}"
    done
    ;;
  priority2)
    for variant in \
      A10_e2e_t4_box0_imgkd \
      A03_decmem_t4 \
      A04_memory_t4; do
      run_logged "mask_v2_${variant}" run_variant "${variant}"
    done
    ;;
  priority3)
    run_logged mask_v2_A11_e2e_t4_box0_imgmemkd \
      run_variant A11_e2e_t4_box0_imgmemkd
    run_logged mask_v2_prepare_hardness prepare_hardness
    for variant in A05_e2e_t8 A06_e2e_t8_s4_t16_hard; do
      run_logged "mask_v2_${variant}" run_variant "${variant}"
    done
    ;;
esac

run_logged priority_mask_summaries write_summaries

if [[ "${#FAILED[@]}" -eq 0 ]]; then
  STATUS=0
  echo "All priority jobs in ${LANE} completed successfully."
else
  STATUS=1
  echo "Failed priority jobs in ${LANE}: ${FAILED[*]}"
fi
echo "Lane logs: ${LOG_ROOT}"
echo "Mask summary: ${ABLATION_ROOT}/summary.csv"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
