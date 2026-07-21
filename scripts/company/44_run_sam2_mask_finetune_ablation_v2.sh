#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-list}"
VARIANT="${2:-}"
VARIANTS=(
  A00_e2e_t4_box1_control
  A01_e2e_t4_box0
  A02_e2e_t4_official_prompt
  A03_decmem_t4
  A04_memory_t4
  A05_e2e_t8
  A06_e2e_t8_s4_t16_hard
  A07_e2e_t4_warmup5
  A08_e2e_t4_gb8
  A09_e2e_t4_hard50x2
  A10_e2e_t4_box0_imgkd
  A11_e2e_t4_box0_imgmemkd
)

if [[ "${ACTION}" == "list" ]]; then
  printf '%s\n' "${VARIANTS[@]}"
  return 0 2>/dev/null || exit 0
fi

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"

SAM2D_ROOT="${SAM2D_ROOT:-/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_MODEL_CONFIG="${SAM2_MODEL_CONFIG:-${SAM2_TRAINING_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v2/stage1_encoder_task_2ep_v2/checkpoints/checkpoint.pt}"
BASE_STAGE_CHECKPOINT="${BASE_STAGE_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v2/stage1_encoder_task_2ep_v2/checkpoints/stage.pt}"
ABLATION_ROOT="${MASK_ABLATION_ROOT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2}"
LEGACY_ROOT="${MASK_ABLATION_V1_ROOT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1}"
LEGACY_COMPAT_ROOT="${MASK_ABLATION_V1_COMPAT_ROOT:-/group-volume/danny-dataset/sam2_distill/runs/sam2_mask_finetune_ablation_v1}"
HARDNESS_ROOT="${MASK_HARDNESS_ROOT:-${ABLATION_ROOT}/hardness_base_t4_box}"
CENTRAL_CSV="${MASK_ABLATION_SUMMARY_CSV:-${ABLATION_ROOT}/summary.csv}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-mask-finetune-ablation-v2}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"
SKIP_DONE="${MASK_ABLATION_SKIP_DONE:-1}"
export GPUS MANIFEST BASE_CHECKPOINT BASE_STAGE_CHECKPOINT SAM2_TRAINING_ROOT

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  fi
}

is_variant() {
  local candidate="$1"
  local item
  for item in "${VARIANTS[@]}"; do
    if [[ "${candidate}" == "${item}" ]]; then
      return 0
    fi
  done
  return 1
}

checkpoint_reached_epoch() {
  python - "$1" "$2" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
target = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
print(f"checkpoint epoch: {checkpoint.get('epoch', -1)}; target: {target}")
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) >= target else 1)
PY
}

audit_inputs() {
  python tools/train/audit_sam2_task_inputs.py \
    --manifest "${MANIFEST}" \
    --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
    --sav-root "${SAV_ROOT}" \
    --sample-videos "${AUDIT_SAMPLE_VIDEOS:-500}" \
    --compact
}

ensure_base_stage_checkpoint() {
  if [[ -f "${BASE_STAGE_CHECKPOINT}" ]]; then
    return 0
  fi
  require_path "${BASE_CHECKPOINT}" || return 1
  echo "Exporting base task checkpoint for hardness mining: ${BASE_STAGE_CHECKPOINT}"
  python tools/train/export_sam2_task_checkpoint.py \
    --trainer-checkpoint "${BASE_CHECKPOINT}" \
    --output "${BASE_STAGE_CHECKPOINT}" \
    --stage-name shared_stage1_encoder_task_2ep_v2 \
    --trainable-mode image_encoder_only \
    --source-stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}"
}

prepare_hardness() {
  local force_args=()
  if [[ "${HARDNESS_FORCE:-0}" == "1" ]]; then
    force_args+=(--force)
  fi
  ensure_base_stage_checkpoint || return 1
  echo "===== Mining fixed T4 base-error hardness on GPUs ${GPUS} ====="
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  PYTHONPATH="${REPO_ROOT}:${SAM2_TRAINING_ROOT}:${PYTHONPATH:-}" \
  torchrun --standalone --nproc-per-node "${NPROC}" \
    tools/train/mine_sav_mask_hardness.py \
    --manifest "${MANIFEST}" \
    --sav-root "${SAV_ROOT}" \
    --out-dir "${HARDNESS_ROOT}" \
    --sam2-root "${SAM2_TRAINING_ROOT}" \
    --sam2-cfg "${SAM2_CONFIG}" \
    --stage-checkpoint "${BASE_STAGE_CHECKPOINT}" \
    --sam2-checkpoint "${SAM2_CHECKPOINT}" \
    --student-checkpoint "${TINYVIT_CHECKPOINT}" \
    --seed 250107256 \
    --max-objects 2 \
    --max-videos "${HARDNESS_MAX_VIDEOS:-0}" \
    "${force_args[@]}"
}

require_hardness() {
  for path in \
    "${HARDNESS_ROOT}/summary.json" \
    "${HARDNESS_ROOT}/eligible_t8.txt" \
    "${HARDNESS_ROOT}/hard50_x2.txt" \
    "${HARDNESS_ROOT}/hard_t16_budget.txt"; do
    require_path "${path}" || return 1
  done
}

reset_variant_environment() {
  export TASK_MASK_ABLATION_V2=1
  export TASK_SEED=250107256
  export TASK_EPOCHS=1
  export TASK_MAX_VIDEOS=0
  export TASK_NUM_FRAMES=4
  export TASK_TRAIN_BATCH_SIZE=1
  export TASK_MAX_NUM_OBJECTS=2
  export TASK_TRAINABLE_MODE=image_encoder_mask_decoder_memory
  export TASK_FREEZE_BATCHNORM=true
  export TASK_ENCODER_LR=3.0e-7
  export TASK_ENCODER_LR_END=3.0e-8
  export TASK_HEAD_LR=1.0e-6
  export TASK_HEAD_LR_END=1.0e-7
  export TASK_LR_WARMUP_FRACTION=0
  export TASK_LR_WARMUP_START_FACTOR=0.1
  export TASK_PROB_USE_POINT=1.0
  export TASK_PROB_USE_BOX=1.0
  export TASK_PROB_SAMPLE_GT=0.0
  export TASK_NUM_FRAMES_TO_CORRECT=1
  export TASK_RANDOM_CORRECTION_FRAMES=false
  export TASK_NUM_INIT_COND_FRAMES=1
  export TASK_RANDOM_INIT_COND_FRAMES=false
  export TASK_NUM_CORRECTION_POINTS=1
  export TASK_LAMBDA_IMG=0
  export TASK_LAMBDA_MEM=0
  export TASK_VIDEO_IDS_FILE=""
  export TASK_TEACHER_MODEL_CONFIG=""
  export TASK_TEACHER_CHECKPOINT=""
}

configure_variant() {
  reset_variant_environment
  case "$1" in
    A00_e2e_t4_box1_control)
      ;;
    A01_e2e_t4_box0)
      export TASK_NUM_CORRECTION_POINTS=0
      ;;
    A02_e2e_t4_official_prompt)
      export TASK_PROB_USE_POINT=0.5
      export TASK_PROB_USE_BOX=0.5
      export TASK_PROB_SAMPLE_GT=0.1
      export TASK_NUM_FRAMES_TO_CORRECT=2
      export TASK_RANDOM_CORRECTION_FRAMES=true
      export TASK_NUM_CORRECTION_POINTS=7
      ;;
    A03_decmem_t4)
      export TASK_TRAINABLE_MODE=mask_decoder_memory
      ;;
    A04_memory_t4)
      export TASK_TRAINABLE_MODE=memory_only
      ;;
    A05_e2e_t8)
      export TASK_NUM_FRAMES=8
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      ;;
    A06_e2e_t8_s4_t16_hard)
      export TASK_NUM_FRAMES=8
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      ;;
    A07_e2e_t4_warmup5)
      export TASK_LR_WARMUP_FRACTION=0.05
      ;;
    A08_e2e_t4_gb8)
      export TASK_TRAIN_BATCH_SIZE=2
      ;;
    A09_e2e_t4_hard50x2)
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/hard50_x2.txt"
      ;;
    A10_e2e_t4_box0_imgkd)
      export TASK_NUM_CORRECTION_POINTS=0
      export TASK_LAMBDA_IMG=0.5
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    A11_e2e_t4_box0_imgmemkd)
      export TASK_NUM_CORRECTION_POINTS=0
      export TASK_LAMBDA_IMG=0.5
      export TASK_LAMBDA_MEM=0.25
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    *)
      echo "[ERROR] Unknown v2 mask ablation: $1" >&2
      return 2
      ;;
  esac
}

describe_variant() {
  configure_variant "$1" || return 1
  echo "Variant: $1"
  echo "Frames: ${TASK_NUM_FRAMES}; batch/GPU: ${TASK_TRAIN_BATCH_SIZE}; GPUs: ${NPROC}"
  echo "Mode: ${TASK_TRAINABLE_MODE}; BN frozen: ${TASK_FREEZE_BATCHNORM}"
  echo "Encoder LR: ${TASK_ENCODER_LR} -> ${TASK_ENCODER_LR_END}"
  echo "Other LR: ${TASK_HEAD_LR} -> ${TASK_HEAD_LR_END}; warmup: ${TASK_LR_WARMUP_FRACTION}"
  echo "Prompt P(point): ${TASK_PROB_USE_POINT}; P(box|point): ${TASK_PROB_USE_BOX}"
  echo "Correction frames/points: ${TASK_NUM_FRAMES_TO_CORRECT}/${TASK_NUM_CORRECTION_POINTS}"
  echo "KD image/memory: ${TASK_LAMBDA_IMG}/${TASK_LAMBDA_MEM}"
  echo "Video list: ${TASK_VIDEO_IDS_FILE:-full train split}"
}

export_stage_checkpoint() {
  local stage_dir="$1"
  local stage_name="$2"
  python tools/train/export_sam2_task_checkpoint.py \
    --trainer-checkpoint "${stage_dir}/checkpoints/checkpoint.pt" \
    --output "${stage_dir}/checkpoints/stage.pt" \
    --stage-name "${stage_name}" \
    --trainable-mode "${TASK_TRAINABLE_MODE}" \
    --source-stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}"
}

train_stage() {
  local stage_dir="$1"
  local stage_name="$2"
  local previous_checkpoint="$3"
  local trainer_checkpoint="${stage_dir}/checkpoints/checkpoint.pt"

  if [[ "${TASK_LAMBDA_IMG}" != "0" || "${TASK_LAMBDA_MEM}" != "0" ]]; then
    require_path "${TASK_TEACHER_MODEL_CONFIG}" || return 1
    require_path "${TASK_TEACHER_CHECKPOINT}" || return 1
  fi
  if [[ -n "${TASK_VIDEO_IDS_FILE}" ]]; then
    require_path "${TASK_VIDEO_IDS_FILE}" || return 1
  fi

  mkdir -p "${stage_dir}/wandb"
  if checkpoint_reached_epoch "${trainer_checkpoint}" "${TASK_EPOCHS}"; then
    echo "skip completed training: ${stage_name}"
  else
    echo "===== Training ${stage_name} on GPUs ${GPUS} ====="
    touch "${stage_dir}/.full_eval_required"
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    PYTHONPATH="${REPO_ROOT}:${SAM2_TRAINING_ROOT}:${PYTHONPATH:-}" \
    TASK_RUN_DIR="${stage_dir}" \
    TASK_STAGE_NAME="${stage_name}" \
    TASK_MANIFEST="${MANIFEST}" \
    SAV_ROOT="${SAV_ROOT}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS}" \
    TASK_PRINT_EVERY="${PRINT_EVERY}" \
    TASK_LOG_EVERY="${LOG_EVERY}" \
    TASK_MAX_VIDEOS="${TASK_MAX_VIDEOS:-0}" \
    SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
    PREVIOUS_TASK_CHECKPOINT="${previous_checkpoint}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_LOSS_EMA_BETA=0.98 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    torchrun --standalone --nproc-per-node "${NPROC}" \
      tools/train/run_sam2_task_training.py \
      --config "${CONFIG}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${stage_name}" \
      --wandb-dir "${stage_dir}/wandb" || return 1
  fi
  checkpoint_reached_epoch "${trainer_checkpoint}" "${TASK_EPOCHS}" || return 1
  export_stage_checkpoint "${stage_dir}" "${stage_name}"
}

evaluate_split() {
  local stage_dir="$1"
  local stage_name="$2"
  local split="$3"
  local eval_skip_done="$4"
  MODEL_FAMILY=sam2 \
  STUDENT_FAMILY=tinyvit \
  STUDENT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
  STUDENT_MODEL_NAME=tiny_vit_21m_512.dist_in22k_ft_in1k \
  STAGE1_CHECKPOINT="${stage_dir}/checkpoints/stage.pt" \
  EXPERIMENT="${stage_name}" \
  RUN_DIR="${stage_dir}" \
  SAV_ROOT="${SAV_ROOT}" \
  SAV_SPLIT="${split}" \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  SKIP_DONE="${eval_skip_done}" \
  CLEAN_PREDICTIONS=1 \
  scripts/company/25_benchmark_stage1_sav_test.sh
}

evaluate_stage() {
  local stage_dir="$1"
  local stage_name="$2"
  local eval_skip_done="${SKIP_DONE}"
  if [[ -f "${stage_dir}/.full_eval_required" ]]; then
    eval_skip_done=0
  fi
  echo "===== Full SA-V val: ${stage_name} ====="
  evaluate_split "${stage_dir}" "${stage_name}" sav_val "${eval_skip_done}" || return 1
  echo "===== Full SA-V test: ${stage_name} ====="
  evaluate_split "${stage_dir}" "${stage_name}" sav_test "${eval_skip_done}" || return 1
  if [[ "${WANDB_MODE}" == "online" ]]; then
    python tools/train/log_task_eval_to_wandb.py \
      --run-file "${stage_dir}/wandb/wandb_run.json" \
      --metrics "sav_val=${stage_dir}/sav_val_box_benchmark/metrics.csv" \
      --metrics "sav_test=${stage_dir}/sav_test_box_benchmark/metrics.csv" || return 1
  fi
  rm -f "${stage_dir}/.full_eval_required"
}

record_summary() {
  local variant_dir="$1"
  local stage_dir="$2"
  python tools/train/summarize_mask_finetune_ablations.py record \
    --variant-dir "${variant_dir}" \
    --stage-dir "${stage_dir}" \
    --central-csv "${CENTRAL_CSV}"
}

run_variant() {
  local name="$1"
  local variant_dir="${ABLATION_ROOT}/${name}"
  local main_dir="${variant_dir}/main"
  local final_dir="${main_dir}"
  local main_name="${name}_main"
  if [[ "${name}" == "A05_e2e_t8" || \
        "${name}" == "A06_e2e_t8_s4_t16_hard" || \
        "${name}" == "A09_e2e_t4_hard50x2" ]]; then
    require_hardness || return 1
  fi
  configure_variant "${name}" || return 1
  describe_variant "${name}" || return 1
  record_summary "${variant_dir}" "${main_dir}" || return 1
  if ! train_stage "${main_dir}" "${main_name}" "${BASE_CHECKPOINT}"; then
    record_summary "${variant_dir}" "${main_dir}"
    return 1
  fi

  if [[ "${name}" == "A06_e2e_t8_s4_t16_hard" ]]; then
    local refine_dir="${variant_dir}/refine_t16"
    export TASK_NUM_FRAMES=16
    export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/hard_t16_budget.txt"
    export TASK_TRAINABLE_MODE=mask_decoder_memory
    export TASK_ENCODER_LR=1.5e-7
    export TASK_ENCODER_LR_END=1.5e-8
    export TASK_HEAD_LR=5.0e-7
    export TASK_HEAD_LR_END=5.0e-8
    record_summary "${variant_dir}" "${refine_dir}" || return 1
    if ! train_stage \
        "${refine_dir}" \
        "${name}_refine_t16" \
        "${main_dir}/checkpoints/checkpoint.pt"; then
      record_summary "${variant_dir}" "${refine_dir}"
      return 1
    fi
    final_dir="${refine_dir}"
  fi

  if ! evaluate_stage "${final_dir}" "${name}"; then
    record_summary "${variant_dir}" "${final_dir}"
    return 1
  fi
  record_summary "${variant_dir}" "${final_dir}"
}

summarize_all() {
  python tools/train/summarize_mask_finetune_ablations.py scan \
    --root "${ABLATION_ROOT}" \
    --legacy-root "${LEGACY_ROOT}" \
    --legacy-root "${LEGACY_COMPAT_ROOT}" \
    --central-csv "${CENTRAL_CSV}"
}

validate_common_paths() {
  for path in \
    "${MANIFEST}" \
    "${SAV_ROOT}/JPEGImages" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt" \
    "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
    "${SAM2_CHECKPOINT}" \
    "${TINYVIT_CHECKPOINT}" \
    "${SOURCE_STAGE1_CHECKPOINT}" \
    "${BASE_CHECKPOINT}" \
    "${CONFIG}"; do
    require_path "${path}" || return 1
  done
}

STATUS=0
case "${ACTION}" in
  describe)
    is_variant "${VARIANT}" && describe_variant "${VARIANT}"
    STATUS="$?"
    ;;
  prepare-hardness)
    validate_common_paths && audit_inputs && prepare_hardness
    STATUS="$?"
    ;;
  run)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Set a valid variant after 'run'; use '$0 list'." >&2
      STATUS=2
    else
      validate_common_paths && audit_inputs && run_variant "${VARIANT}"
      STATUS="$?"
    fi
    ;;
  eval)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Set a valid variant after 'eval'." >&2
      STATUS=2
    else
      configure_variant "${VARIANT}" || STATUS="$?"
      final_dir="${ABLATION_ROOT}/${VARIANT}/main"
      if [[ "${VARIANT}" == "A06_e2e_t8_s4_t16_hard" ]]; then
        final_dir="${ABLATION_ROOT}/${VARIANT}/refine_t16"
        export TASK_NUM_FRAMES=16
        export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/hard_t16_budget.txt"
        export TASK_TRAINABLE_MODE=mask_decoder_memory
        export TASK_ENCODER_LR=1.5e-7
        export TASK_ENCODER_LR_END=1.5e-8
        export TASK_HEAD_LR=5.0e-7
        export TASK_HEAD_LR_END=5.0e-8
      fi
      if [[ "${STATUS}" -eq 0 ]]; then
        evaluate_stage "${final_dir}" "${VARIANT}" && \
          record_summary "${ABLATION_ROOT}/${VARIANT}" "${final_dir}"
        STATUS="$?"
      fi
    fi
    ;;
  summarize)
    summarize_all
    STATUS="$?"
    ;;
  smoke)
    validate_common_paths && audit_inputs || STATUS="$?"
    if [[ "${STATUS}" -eq 0 ]]; then
      configure_variant A00_e2e_t4_box1_control || STATUS="$?"
      export TASK_MAX_VIDEOS="${SMOKE_MAX_VIDEOS:-8}"
      smoke_dir="${ABLATION_ROOT}/smoke_A00_${HOSTNAME:-node}"
      train_stage "${smoke_dir}" smoke_A00_e2e_t4_box1_control "${BASE_CHECKPOINT}" || STATUS="$?"
    fi
    ;;
  all)
    validate_common_paths && audit_inputs && prepare_hardness || STATUS="$?"
    if [[ "${STATUS}" -eq 0 ]]; then
      for item in "${VARIANTS[@]}"; do
        run_variant "${item}" || {
          STATUS="$?"
          break
        }
      done
    fi
    ;;
  *)
    echo "Usage: $0 {list|describe VARIANT|smoke|prepare-hardness|run VARIANT|eval VARIANT|summarize|all}" >&2
    STATUS=2
    ;;
esac

echo "SAM2 mask fine-tuning ablation v2 status: ${STATUS}"
echo "Run root: ${ABLATION_ROOT}"
echo "Central summary: ${CENTRAL_CSV}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
