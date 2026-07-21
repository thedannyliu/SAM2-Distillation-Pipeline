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
if [[ "${WANDB_MODE:-online}" != "online" ]]; then
  echo "WANDB_MODE=online is required; got ${WANDB_MODE}" >&2
  return 2 2>/dev/null || exit 2
fi
export WANDB_MODE=online
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

gpu_count="$(python - "${GPUS}" <<'PY'
import sys
print(len([value for value in sys.argv[1].split(",") if value.strip()]))
PY
)"
if [[ "${gpu_count}" -ne 4 ]]; then
  echo "${LANE} requires exactly 4 GPUs; got GPUS=${GPUS}" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ "${RECOVERY_DRY_RUN}" != "1" ]]; then
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

run_stage1_sam2() {
  RUN_ROOT="${SAM2D_ROOT}/runs" \
    scripts/company/34_run_stage1_recovery_lane.sh sam2 "$1" "$2"
}

run_stage1_sam31() {
  RUN_ROOT="${SAM2D_ROOT}/runs" \
    scripts/company/34_run_stage1_recovery_lane.sh sam31 "$1"
}

run_repvit() {
  RUN_ROOT="${SAM2D_ROOT}/runs/repvit_stage1_v1" \
  CHECKPOINT_ROOT="${SAM2D_ROOT}/checkpoints" \
  WANDB_PROJECT=sam2-distill-repvit-stage1-v1 \
  WANDB_RUN_ID='' \
    scripts/company/38_run_repvit_sam21l_stage1.sh all
}

run_mask_v1() {
  local variant="$1" stage_dir
  MASK_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1" \
    scripts/company/43_run_sam2_mask_finetune_ablation.sh "${variant}" || return 1
  stage_dir="${MASK_ABLATION_V1_ROOT}/${variant}/mask_${variant}"
  verify_task_stage "${stage_dir}" || return 1
  python tools/train/verify_wandb_history.py \
    --run-file "${stage_dir}/wandb/wandb_run.json" \
    --metric train/loss_total || return 1
  python tools/train/summarize_mask_finetune_ablations.py scan \
    --root "${MASK_ABLATION_ROOT}" \
    --legacy-root "${MASK_ABLATION_V1_ROOT}" \
    --central-csv "${MASK_ABLATION_ROOT}/summary.csv"
}

run_mask_v2() {
  local action="$1" variant="${2:-}" stage_dir
  MASK_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2" \
  WANDB_PROJECT=sam2-mask-finetune-ablation-v2 \
  WANDB_RUN_ID='' \
    scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh "$@" || return 1
  [[ "${action}" == "run" ]] || return
  stage_dir="${MASK_ABLATION_ROOT}/${variant}/main"
  verify_task_stage "${stage_dir}" || return 1
  python tools/train/verify_wandb_history.py \
    --run-file "${stage_dir}/wandb/wandb_run.json" \
    --metric train/loss_total || return 1
  if [[ "${variant}" == "A06_e2e_t8_s4_t16_hard" ]]; then
    stage_dir="${MASK_ABLATION_ROOT}/${variant}/refine_t16"
    verify_task_stage "${stage_dir}" || return 1
    python tools/train/verify_wandb_history.py \
      --run-file "${stage_dir}/wandb/wandb_run.json" \
      --metric train/loss_total || return 1
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

run_report() {
  RUNS_ROOTS="${SAM2D_ROOT}/runs" \
  REPORT_DIR="${LOG_ROOT}/final_report" \
    scripts/company/45_report_all_experiments.sh
}

case "${LANE}" in
  node1)
    run_logged sam31_n1_cos100 run_stage1_sam31 n1_cos100_adapter_ft_w2k
    run_logged sam31_n1_cos025 run_stage1_sam31 n1_cos025_adapter_ft_w2k
    for variant in decoder_lr2e7 decoder_lr5e7; do
      run_logged "mask_v1_${variant}" run_mask_v1 "${variant}"
    done
    for variant in A00_e2e_t4_box1_control A03_decmem_t4; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  node2)
    run_logged sam31_n2_adapter_frozen run_stage1_sam31 n2_adapter_cos025_frozen
    run_logged sam31_n2_adapter_ft_w0 run_stage1_sam31 n2_adapter_cos025_ft_w0
    for variant in decoder_lr2e6 encdec_low_frozenbn; do
      run_logged "mask_v1_${variant}" run_mask_v1 "${variant}"
    done
    for variant in A04_memory_t4 A10_e2e_t4_box0_imgkd; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  node3)
    run_logged sam31_n3_cos150 run_stage1_sam31 n3_cos150_adapter_ft_w2k
    run_logged sam31_n3_relation010 run_stage1_sam31 n3_relation010_adapter_ft_w2k
    for variant in encdec_low_trainbn decoder_lr5e7_boxonly; do
      run_logged "mask_v1_${variant}" run_mask_v1 "${variant}"
    done
    for variant in A01_e2e_t4_box0 A02_e2e_t4_official_prompt; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  node4)
    run_logged sam31_n3_cos025_relation010 \
      run_stage1_sam31 n3_cos025_relation010_adapter_ft_w2k
    run_logged sam2_tv21_adapter_cos025 \
      run_stage1_sam2 tv21_adapter_sam21l_msehr_cos025 252265
    for variant in A07_e2e_t4_warmup5 A08_e2e_t4_gb8 A11_e2e_t4_box0_imgmemkd; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  node5)
    run_logged sam2_tv21_bplus \
      run_stage1_sam2 tv21_proj_sam21bplus_msehr 252265
    run_logged repvit_all run_repvit
    ;;
  node6)
    run_logged sam31_n1_cos000 run_stage1_sam31 n1_cos000_adapter_ft_w2k
    run_logged sam31_n2_projection run_stage1_sam31 n2_projection_cos025_ft_w2k
    run_logged mask_v2_prepare_hardness run_mask_v2 prepare-hardness
    for variant in A05_e2e_t8 A06_e2e_t8_s4_t16_hard A09_e2e_t4_hard50x2; do
      run_logged "mask_v2_${variant}" run_mask_v2 run "${variant}"
    done
    ;;
  *)
    echo "Usage: $0 {node1|node2|node3|node4|node5|node6}" >&2
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
