#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-run}"
case "${ACTION}" in
  describe|run) ;;
  *)
    echo "Usage: $0 {describe|run}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "[ERROR] TinyViT-5M pseudo-label lane requires four GPUs: ${GPUS}" >&2
  return 2 2>/dev/null || exit 2
fi

SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
if [[ -z "${SAV_ROOT:-}" ]]; then
  for candidate in \
    /group-volume/danny-dataset/SA-V \
    /mnt/data/danny-dataset/SA-V \
    /danny-dataset/SA-V; do
    if [[ -f "${candidate}/sav_val/sav_val.txt" && \
          -f "${candidate}/sav_test/sav_test.txt" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_group_runtime.parquet}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
TEACHER_CONFIG="${TEACHER_CONFIG:-${SAM2_TRAINING_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${SAM2_CHECKPOINT}}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_extra_adapter_cos/tv5_adapter_sam21l_msehr/checkpoints/best.pt}"
BASE_TASK_CHECKPOINT="${BASE_TASK_CHECKPOINT:-${SAM2D_ROOT}/runs/weekend_72h_v1/tinyvit/tv5_W1_decmem_t4_3ep/checkpoints/last.pt}"
ELIGIBLE_T8="${ELIGIBLE_T8:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/hardness_base_t4_box/eligible_t8.txt}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/tinyvit5_pseudolabel_v1}"
LOG_ROOT="${LOG_ROOT:-/user-volume/tinyvit5_pseudolabel_logs}"
WANDB_PROJECT="${WANDB_PROJECT:-tinyvit5-pseudolabel-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"
PSEUDO_REFINE_MIN_VAL_JF="${PSEUDO_REFINE_MIN_VAL_JF:-66.5}"

MODEL_NAME="tiny_vit_5m_224.dist_in22k_ft_in1k"
ADAPTER_MODE="residual_dwconv"
FAILED=()
COMPLETED_CANDIDATES=()

CONTROL_NAME="tv5_PL0_gt_t4_3ep"
PSEUDO025_NAME="tv5_PL1_sam21l_soft025_t4_3ep"
PSEUDO050_NAME="tv5_PL2_sam21l_soft050_t4_3ep"
REFINE_NAME="tv5_PL3_selected_t8_2ep"

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

wandb_preflight() {
  [[ "${WANDB_MODE}" != "online" ]] && return 0
  python - <<'PY'
import wandb

viewer = wandb.Api(timeout=30).viewer
identity = (
    getattr(viewer, "username", None)
    or getattr(viewer, "email", None)
    or str(viewer)
)
if not identity:
    raise SystemExit("W&B viewer identity is empty")
print(f"W&B online preflight: PASS | {identity}", flush=True)
PY
}

describe_lane() {
  echo "TinyViT-5M SAM2.1-L pseudo-label continuation"
  echo "Hardware: one independent 4xH100 node"
  echo "Base: ${BASE_TASK_CHECKPOINT}"
  echo "Teacher: ${TEACHER_CHECKPOINT}"
  echo "Tracking: W&B ${WANDB_PROJECT}; mode ${WANDB_MODE}"
  echo "Selection: full SA-V val J&F; test is descriptive only"
  echo "Retention: only last.pt and best.pt are physical task checkpoints"
  echo "Budget: 13 T4-equivalent SA-V epochs plus four full val/test evaluations"
  echo "  ${CONTROL_NAME}: matched GT-only E2E T4, 3 epochs"
  echo "  ${PSEUDO025_NAME}: GT + SAM2.1-L soft masks (weight 0.25), T4, 3 epochs"
  echo "  ${PSEUDO050_NAME}: GT + SAM2.1-L soft masks (weight 0.50), T4, 3 epochs"
  echo "  ${REFINE_NAME}: val-selected pseudo branch, T8, 2 epochs"
  echo "  Refine gate: val J&F >= ${PSEUDO_REFINE_MIN_VAL_JF}"
}

normalize_checkpoints() {
  local checkpoint_dir="$1"
  if [[ -f "${checkpoint_dir}/checkpoint.pt" && \
        ! -L "${checkpoint_dir}/checkpoint.pt" ]]; then
    mv -f "${checkpoint_dir}/checkpoint.pt" "${checkpoint_dir}/last.pt" || return 1
  fi
  if [[ -f "${checkpoint_dir}/stage.pt" && \
        ! -L "${checkpoint_dir}/stage.pt" ]]; then
    mv -f "${checkpoint_dir}/stage.pt" "${checkpoint_dir}/best.pt" || return 1
  fi
  require_path "${checkpoint_dir}/last.pt" || return 1
  require_path "${checkpoint_dir}/best.pt" || return 1
  ln -sfn last.pt "${checkpoint_dir}/checkpoint.pt"
  ln -sfn best.pt "${checkpoint_dir}/stage.pt"
  find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.pt' \
    ! -name last.pt ! -name best.pt -delete
}

record_failure() {
  FAILED+=("$1:$2")
  echo "$1 status: $2"
}

run_stage() {
  local name="$1"
  local previous_checkpoint="$2"
  local epochs="$3"
  local frames="$4"
  local encoder_lr="$5"
  local encoder_lr_end="$6"
  local head_lr="$7"
  local head_lr_end="$8"
  local pseudo_weight="$9"
  local video_ids_file="${10}"
  local stage_dir="${RUN_ROOT}/${name}"
  local checkpoint_dir="${stage_dir}/checkpoints"
  local log
  local teacher_config=""
  local teacher_checkpoint=""
  log="${LOG_ROOT}/${name}_$(date +%Y%m%d_%H%M%S).log"

  if [[ "${pseudo_weight}" != "0" ]]; then
    teacher_config="${TEACHER_CONFIG}"
    teacher_checkpoint="${TEACHER_CHECKPOINT}"
  fi

  echo
  echo "================================================================"
  echo "Starting: ${name}"
  echo "T${frames}; epochs: ${epochs}; pseudo-mask weight: ${pseudo_weight}"
  echo "Previous checkpoint: ${previous_checkpoint}"
  echo "Log: ${log}"
  echo "================================================================"

  if [[ -f "${stage_dir}/.pipeline_complete" && \
        -f "${stage_dir}/sav_val_box_benchmark/metrics.csv" && \
        -f "${stage_dir}/sav_test_box_benchmark/metrics.csv" ]]; then
    echo "skip completed formal stage: ${name}"
    COMPLETED_CANDIDATES+=("${name}=${stage_dir}")
    return 0
  fi

  env \
    GPUS="${GPUS}" \
    FULL_EVAL_GPUS="${FULL_EVAL_GPUS}" \
    SAM2D_ROOT="${SAM2D_ROOT}" \
    SAV_ROOT="${SAV_ROOT}" \
    MANIFEST="${MANIFEST}" \
    SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT}" \
    SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
    CONFIG="${CONFIG}" \
    RUN_ROOT="${RUN_ROOT}" \
    STUDENT_FAMILY=tinyvit \
    TINYVIT_MODEL_NAME="${MODEL_NAME}" \
    TINYVIT_ADAPTER_MODE="${ADAPTER_MODE}" \
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS}" \
    PRINT_EVERY="${PRINT_EVERY}" \
    LOG_EVERY="${LOG_EVERY}" \
    STAGE1_NAME="${name}" \
    STAGE1_MODE=image_encoder_mask_decoder_memory \
    STAGE1_EPOCHS="${epochs}" \
    STAGE1_FRAMES="${frames}" \
    STAGE1_ENCODER_LR="${encoder_lr}" \
    STAGE1_ENCODER_LR_END="${encoder_lr_end}" \
    STAGE1_HEAD_LR="${head_lr}" \
    STAGE1_HEAD_LR_END="${head_lr_end}" \
    STAGE1_PREVIOUS_CHECKPOINT="${previous_checkpoint}" \
    TASK_EXPERIMENT_SUITE=tinyvit5_pseudolabel_v1 \
    TASK_MASK_ABLATION_V2=1 \
    TASK_SEED=250107256 \
    TASK_TRAIN_BATCH_SIZE=1 \
    TASK_MAX_NUM_OBJECTS=2 \
    TASK_FREEZE_BATCHNORM=true \
    TASK_LR_WARMUP_FRACTION=0.05 \
    TASK_LR_WARMUP_START_FACTOR=0.1 \
    TASK_PROB_USE_POINT=0.5 \
    TASK_PROB_USE_BOX=0.5 \
    TASK_PROB_SAMPLE_GT=0.1 \
    TASK_NUM_FRAMES_TO_CORRECT=2 \
    TASK_RANDOM_CORRECTION_FRAMES=true \
    TASK_NUM_INIT_COND_FRAMES=1 \
    TASK_RANDOM_INIT_COND_FRAMES=false \
    TASK_NUM_CORRECTION_POINTS=7 \
    TASK_LAMBDA_TASK=1.0 \
    TASK_LAMBDA_IMG=0 \
    TASK_LAMBDA_MEM=0 \
    TASK_LAMBDA_MASK_LOGITS="${pseudo_weight}" \
    TASK_LAMBDA_OBJ_PTR=0 \
    TASK_TEACHER_MODEL_CONFIG="${teacher_config}" \
    TASK_TEACHER_CHECKPOINT="${teacher_checkpoint}" \
    TASK_VIDEO_IDS_FILE="${video_ids_file}" \
    TASK_EXPORT_STAGE_CHECKPOINT=1 \
    TASK_EVAL_SPLITS=sav_val,sav_test \
    SKIP_DONE=1 \
      scripts/company/39_run_sam2_task_finetune_3stage.sh stage1 \
        2>&1 | tee -a "${log}"
  local stage_status="${PIPESTATUS[0]}"
  if [[ "${stage_status}" -ne 0 ]]; then
    return "${stage_status}"
  fi

  normalize_checkpoints "${checkpoint_dir}" || return 1
  touch "${stage_dir}/.pipeline_complete"
  COMPLETED_CANDIDATES+=("${name}=${stage_dir}")
}

select_candidates() {
  local output_prefix="$1"
  shift
  local -a args=()
  local spec
  for spec in "$@"; do
    args+=(--candidate "${spec}")
  done
  python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${RUN_ROOT}/${output_prefix}_selection.json" \
    --out-csv "${RUN_ROOT}/${output_prefix}_summary.csv" \
    --print-selected
}

candidate_dir() {
  local selected="$1"
  shift
  local spec
  for spec in "$@"; do
    if [[ "${spec%%=*}" == "${selected}" ]]; then
      echo "${spec#*=}"
      return 0
    fi
  done
  return 1
}

write_summary() {
  local -a args=()
  local spec
  for spec in "${COMPLETED_CANDIDATES[@]}"; do
    args+=(--candidate "${spec}")
  done
  if [[ "${#args[@]}" -eq 0 ]]; then
    echo "[ERROR] No completed candidate available for summary" >&2
    return 1
  fi
  python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${RUN_ROOT}/selection.json" \
    --out-csv "${RUN_ROOT}/summary.csv"
}

if [[ "${ACTION}" == "describe" ]]; then
  describe_lane
  return 0 2>/dev/null || exit 0
fi

for path in \
  "${MANIFEST}" \
  "${SAV_ROOT}/sav_val/sav_val.txt" \
  "${SAV_ROOT}/sav_test/sav_test.txt" \
  "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
  "${SAM2_CHECKPOINT}" \
  "${TEACHER_CONFIG}" \
  "${TEACHER_CHECKPOINT}" \
  "${TINYVIT_CHECKPOINT}" \
  "${SOURCE_STAGE1_CHECKPOINT}" \
  "${BASE_TASK_CHECKPOINT}" \
  "${ELIGIBLE_T8}" \
  "${CONFIG}"; do
  require_path "${path}" || return 1 2>/dev/null || exit 1
done
wandb_preflight || return 1 2>/dev/null || exit 1
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
describe_lane

run_stage \
  "${CONTROL_NAME}" "${BASE_TASK_CHECKPOINT}" \
  3 4 1.0e-7 1.0e-8 3.0e-7 3.0e-8 0 "" || \
    record_failure "${CONTROL_NAME}" "$?"

PSEUDO_CANDIDATES=()
if run_stage \
  "${PSEUDO025_NAME}" "${BASE_TASK_CHECKPOINT}" \
  3 4 1.0e-7 1.0e-8 3.0e-7 3.0e-8 0.25 ""; then
  PSEUDO_CANDIDATES+=("${PSEUDO025_NAME}=${RUN_ROOT}/${PSEUDO025_NAME}")
else
  record_failure "${PSEUDO025_NAME}" "$?"
fi
if run_stage \
  "${PSEUDO050_NAME}" "${BASE_TASK_CHECKPOINT}" \
  3 4 1.0e-7 1.0e-8 3.0e-7 3.0e-8 0.50 ""; then
  PSEUDO_CANDIDATES+=("${PSEUDO050_NAME}=${RUN_ROOT}/${PSEUDO050_NAME}")
else
  record_failure "${PSEUDO050_NAME}" "$?"
fi

if [[ "${#PSEUDO_CANDIDATES[@]}" -gt 0 ]]; then
  SELECTED="$(select_candidates pseudo_fork "${PSEUDO_CANDIDATES[@]}")"
  SELECT_STATUS="$?"
  if [[ "${SELECT_STATUS}" -eq 0 ]]; then
    SELECTED_DIR="$(candidate_dir "${SELECTED}" "${PSEUDO_CANDIDATES[@]}")"
    REFINE_PSEUDO_WEIGHT=0.25
    if [[ "${SELECTED}" == "${PSEUDO050_NAME}" ]]; then
      REFINE_PSEUDO_WEIGHT=0.50
    fi
    if python tools/experiments/check_task_metric_gate.py \
      --metrics "${SELECTED_DIR}/sav_val_box_benchmark/metrics.csv" \
      --min-jf "${PSEUDO_REFINE_MIN_VAL_JF}"; then
      run_stage \
        "${REFINE_NAME}" "${SELECTED_DIR}/checkpoints/last.pt" \
        2 8 5.0e-8 5.0e-9 1.5e-7 1.5e-8 \
        "${REFINE_PSEUDO_WEIGHT}" "${ELIGIBLE_T8}" || \
          record_failure "${REFINE_NAME}" "$?"
    else
      echo "[SKIP] ${REFINE_NAME}: selected pseudo branch failed refine gate."
    fi
  else
    record_failure pseudo_selection "${SELECT_STATUS}"
  fi
else
  record_failure pseudo_selection 1
fi

write_summary || record_failure summary "$?"

echo
echo "TinyViT-5M pseudo-label lane"
echo "Run root: ${RUN_ROOT}"
echo "Log root: ${LOG_ROOT}"
if [[ -f "${RUN_ROOT}/summary.csv" ]]; then
  echo "Summary: ${RUN_ROOT}/summary.csv"
  cat "${RUN_ROOT}/summary.csv"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed jobs: ${FAILED[*]}"
  return 1 2>/dev/null || exit 1
fi
echo "Lane status: 0"
return 0 2>/dev/null || exit 0
