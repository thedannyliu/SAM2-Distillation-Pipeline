#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-all}"
GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v1}"
SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-task-finetune-tv21-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
SKIP_DONE="${SKIP_DONE:-1}"

STAGE1_NAME="stage1_encoder_task_2ep"
STAGE2_NAME="stage2_encoder_decoder_task_2ep"
STAGE3_NAME="stage3_encoder_decoder_memory_task_1ep"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  fi
}

audit_inputs() {
  echo "===== Auditing task fine-tuning inputs ====="
  python tools/train/audit_sam2_task_inputs.py \
    --manifest "${MANIFEST}" \
    --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
    --sav-root "${SAV_ROOT}" \
    --sample-videos "${AUDIT_SAMPLE_VIDEOS:-500}" || return 1
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
epoch = int(checkpoint.get("epoch", -1))
print(f"checkpoint epoch: {epoch}; target: {target}")
raise SystemExit(0 if epoch >= target else 1)
PY
}

export_stage_checkpoint() {
  local stage_name="$1"
  local mode="$2"
  local stage_dir="${RUN_ROOT}/${stage_name}"
  python tools/train/export_sam2_task_checkpoint.py \
    --trainer-checkpoint "${stage_dir}/checkpoints/checkpoint.pt" \
    --output "${stage_dir}/checkpoints/stage.pt" \
    --stage-name "${stage_name}" \
    --trainable-mode "${mode}" \
    --source-stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" || return 1
}

train_stage() {
  local stage_name="$1"
  local mode="$2"
  local epochs="$3"
  local frames="$4"
  local encoder_lr="$5"
  local encoder_lr_end="$6"
  local head_lr="$7"
  local head_lr_end="$8"
  local previous_checkpoint="$9"
  local max_videos="${10:-0}"
  local stage_dir="${RUN_ROOT}/${stage_name}"
  local trainer_checkpoint="${stage_dir}/checkpoints/checkpoint.pt"

  mkdir -p "${stage_dir}/wandb"
  if checkpoint_reached_epoch "${trainer_checkpoint}" "${epochs}"; then
    echo "skip training-complete task stage: ${stage_name}"
  else
    echo "===== Training ${stage_name} on GPUs ${GPUS} ====="
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    PYTHONPATH="${REPO_ROOT}:${SAM2_TRAINING_ROOT}:${PYTHONPATH:-}" \
    SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT}" \
    TASK_RUN_DIR="${stage_dir}" \
    TASK_STAGE_NAME="${stage_name}" \
    TASK_TRAINABLE_MODE="${mode}" \
    TASK_MANIFEST="${MANIFEST}" \
    SAV_ROOT="${SAV_ROOT}" \
    TASK_EPOCHS="${epochs}" \
    TASK_NUM_FRAMES="${frames}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS}" \
    TASK_MAX_VIDEOS="${max_videos}" \
    TASK_ENCODER_LR="${encoder_lr}" \
    TASK_ENCODER_LR_END="${encoder_lr_end}" \
    TASK_HEAD_LR="${head_lr}" \
    TASK_HEAD_LR_END="${head_lr_end}" \
    SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
    PREVIOUS_TASK_CHECKPOINT="${previous_checkpoint}" \
    WANDB_MODE="${WANDB_MODE}" \
    torchrun --standalone --nproc-per-node "${NPROC}" \
      tools/train/run_sam2_task_training.py \
      --config "${CONFIG}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${stage_name}" \
      --wandb-dir "${stage_dir}/wandb" || return 1
  fi

  checkpoint_reached_epoch "${trainer_checkpoint}" "${epochs}" || {
    echo "[ERROR] ${stage_name} did not reach epoch ${epochs}" >&2
    return 1
  }
  export_stage_checkpoint "${stage_name}" "${mode}" || return 1
}

evaluate_stage_split() {
  local stage_name="$1"
  local split="$2"
  local stage_dir="${RUN_ROOT}/${stage_name}"
  echo "===== Full ${split} evaluation: ${stage_name} on GPUs ${FULL_EVAL_GPUS} ====="
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
  SKIP_DONE="${SKIP_DONE}" \
  scripts/company/25_benchmark_stage1_sav_test.sh || return 1
}

evaluate_stage() {
  evaluate_stage_split "$1" sav_val || return 1
  evaluate_stage_split "$1" sav_test || return 1
}

run_stage1() {
  train_stage "${STAGE1_NAME}" image_encoder_only 2 2 \
    1.0e-6 1.0e-7 1.0e-6 1.0e-7 "" 0 || return 1
  evaluate_stage "${STAGE1_NAME}" || return 1
}

run_stage2() {
  local previous="${RUN_ROOT}/${STAGE1_NAME}/checkpoints/checkpoint.pt"
  require_path "${previous}" || return 1
  train_stage "${STAGE2_NAME}" image_encoder_mask_decoder 2 2 \
    5.0e-7 5.0e-8 2.0e-6 2.0e-7 "${previous}" 0 || return 1
  evaluate_stage "${STAGE2_NAME}" || return 1
}

run_stage3() {
  local previous="${RUN_ROOT}/${STAGE2_NAME}/checkpoints/checkpoint.pt"
  require_path "${previous}" || return 1
  train_stage "${STAGE3_NAME}" image_encoder_mask_decoder_memory 1 4 \
    3.0e-7 3.0e-8 1.0e-6 1.0e-7 "${previous}" 0 || return 1
  evaluate_stage "${STAGE3_NAME}" || return 1
}

smoke_pipeline() {
  local saved_run_root="${RUN_ROOT}"
  local saved_wandb_mode="${WANDB_MODE}"
  local revision
  revision="$(git rev-parse --short HEAD)"
  RUN_ROOT="${SMOKE_RUN_ROOT:-/user-volume/sam2_task_finetune_smoke_${HOSTNAME}_${revision}}"
  WANDB_MODE=disabled
  echo "===== Four-GPU task-loss smoke test (8 videos) ====="
  train_stage smoke_encoder_task image_encoder_only 1 2 \
    1.0e-6 1.0e-7 1.0e-6 1.0e-7 "" 8 || {
      RUN_ROOT="${saved_run_root}"
      WANDB_MODE="${saved_wandb_mode}"
      return 1
    }
  RUN_ROOT="${saved_run_root}"
  WANDB_MODE="${saved_wandb_mode}"
}

for path in "${MANIFEST}" "${SAV_ROOT}/JPEGImages" \
  "${SAV_ROOT}/sav_val/sav_val.txt" "${SAV_ROOT}/sav_test/sav_test.txt" \
  "${SAM2_TRAINING_ROOT}/training/model/sam2.py" "${SAM2_CHECKPOINT}" \
  "${TINYVIT_CHECKPOINT}" "${SOURCE_STAGE1_CHECKPOINT}" "${CONFIG}"; do
  require_path "${path}" || return 1 2>/dev/null || exit 1
done

mkdir -p "${RUN_ROOT}"

case "${ACTION}" in
  audit)
    audit_inputs
    ;;
  smoke)
    audit_inputs && smoke_pipeline
    ;;
  stage1)
    audit_inputs && run_stage1
    ;;
  stage2)
    audit_inputs && run_stage2
    ;;
  stage3)
    audit_inputs && run_stage3
    ;;
  eval)
    evaluate_stage "${EVAL_STAGE:?Set EVAL_STAGE for eval action}"
    ;;
  all)
    audit_inputs && smoke_pipeline && run_stage1 && run_stage2 && run_stage3
    ;;
  *)
    echo "Usage: $0 {audit|smoke|stage1|stage2|stage3|eval|all}" >&2
    false
    ;;
esac

STATUS="$?"
echo "SAM2 task fine-tuning pipeline status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
