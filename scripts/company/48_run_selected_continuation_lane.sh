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
LOG_ROOT="${LOG_ROOT:-/user-volume/selected_continuation_logs/${LANE}}"
CONTINUATION_DRY_RUN="${CONTINUATION_DRY_RUN:-0}"
FAILED=()

export DATA_ROOT SAM2D_ROOT SAV_ROOT MANIFEST GPUS FULL_EVAL_GPUS
export WANDB_MODE=online
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
export FULL_EVAL=1
export SKIP_DONE=1
export MASK_ABLATION_SKIP_DONE=1
export MASK_ABLATION_V1_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1"
export MASK_ABLATION_V1_COMPAT_ROOT="${MASK_ABLATION_V1_ROOT}"
export MASK_ABLATION_WANDB_PROJECT=sam2-mask-finetune-ablation-v1
export MASK_ABLATION_BASE_CHECKPOINT="${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v2/stage1_encoder_task_2ep_v2/checkpoints/checkpoint.pt"

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

if [[ "${CONTINUATION_DRY_RUN}" != "1" ]]; then
  python - <<'PY' || return 1 2>/dev/null || exit 1
import wandb

viewer = wandb.Api(timeout=30).viewer
if isinstance(viewer, dict):
    identity = viewer.get("username") or viewer.get("email") or viewer.get("entity")
else:
    identity = next(
        (getattr(viewer, name, None) for name in ("username", "email", "entity")
         if getattr(viewer, name, None)),
        None,
    )
identity = identity or str(viewer) or "authenticated user"
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
  if [[ "${CONTINUATION_DRY_RUN}" == "1" ]]; then
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

run_stage1_sam2() {
  RUN_ROOT="${SAM2D_ROOT}/runs" \
    scripts/company/34_run_stage1_recovery_lane.sh sam2 "$1" "$2"
}

run_stage1_sam31() {
  RUN_ROOT="${SAM2D_ROOT}/runs" \
    scripts/company/34_run_stage1_recovery_lane.sh sam31 "$1"
}

run_repvit_m09() {
  RUN_ROOT="${SAM2D_ROOT}/runs/repvit_stage1_v1" \
  CHECKPOINT_ROOT="${SAM2D_ROOT}/checkpoints" \
  WANDB_PROJECT=sam2-distill-repvit-stage1-v1 \
    env -u WANDB_RUN_ID scripts/company/38_run_repvit_sam21l_stage1.sh m09
}

run_mask_v1() {
  local variant="$1"
  MASK_ABLATION_ROOT="${MASK_ABLATION_V1_ROOT}" \
    scripts/company/43_run_sam2_mask_finetune_ablation.sh "${variant}"
}

finalize_mask_v2() {
  local variant="$1"
  MASK_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2" \
  WANDB_PROJECT=sam2-mask-finetune-ablation-v2 \
    env -u WANDB_RUN_ID \
      scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh finalize "${variant}"
}

run_report() {
  RUNS_ROOTS="${SAM2D_ROOT}/runs" \
  REPORT_DIR="${LOG_ROOT}/final_report" \
    scripts/company/45_report_all_experiments.sh
}

case "${LANE}" in
  selected1)
    for variant in \
      A00_e2e_t4_box1_control \
      A01_e2e_t4_box0 \
      A02_e2e_t4_official_prompt \
      A03_decmem_t4; do
      run_logged "finalize_${variant}" finalize_mask_v2 "${variant}"
    done
    run_logged mask_v1_encdec_low_frozenbn run_mask_v1 encdec_low_frozenbn
    run_logged sam2_tv21_bplus \
      run_stage1_sam2 tv21_proj_sam21bplus_msehr 252265
    ;;
  selected2)
    for variant in \
      A04_memory_t4 \
      A05_e2e_t8 \
      A06_e2e_t8_s4_t16_hard \
      A07_e2e_t4_warmup5; do
      run_logged "finalize_${variant}" finalize_mask_v2 "${variant}"
    done
    run_logged sam31_n1_cos000 \
      run_stage1_sam31 n1_cos000_adapter_ft_w2k
    run_logged sam31_n1_cos100 \
      run_stage1_sam31 n1_cos100_adapter_ft_w2k
    run_logged sam31_n1_cos025 \
      run_stage1_sam31 n1_cos025_adapter_ft_w2k
    ;;
  selected3)
    for variant in \
      A08_e2e_t4_gb8 \
      A09_e2e_t4_hard50x2 \
      A10_e2e_t4_box0_imgkd \
      A11_e2e_t4_box0_imgmemkd; do
      run_logged "finalize_${variant}" finalize_mask_v2 "${variant}"
    done
    run_logged sam31_n2_projection \
      run_stage1_sam31 n2_projection_cos025_ft_w2k
    run_logged repvit_m09 run_repvit_m09
    run_logged sam31_n2_adapter_frozen \
      run_stage1_sam31 n2_adapter_cos025_frozen
    ;;
  *)
    echo "Usage: $0 {selected1|selected2|selected3}" >&2
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
