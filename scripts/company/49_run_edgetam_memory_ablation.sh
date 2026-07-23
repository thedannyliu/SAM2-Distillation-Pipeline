#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-list}"
VARIANT="${2:-}"
VARIANTS=(
  M0_sam2_mem4
  M1_sam2_mem2
  M2a_edgetam_hybrid2_official
  M2b_edgetam_hybrid2_current
  R0_edgetam_e2e_t4_task
  R1_edgetam_e2e_t4_imgkd
  R2_edgetam_e2e_t4_imgmemkd
  R3_edgetam_e2e_t8_imgmemkd
)

if [[ "${ACTION}" == "list" ]]; then
  printf '%s\n' "${VARIANTS[@]}"
  return 0 2>/dev/null || exit 0
fi

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"

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
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
EDGETAM_REQUIRED_COMMIT="${EDGETAM_REQUIRED_COMMIT:-7711e012a30a2402c4eaab637bdb00a521302c91}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/edgetam/edgetam.pt}"
SAM2_MODEL_CONFIG="${SAM2_MODEL_CONFIG:-${SAM2_TRAINING_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/A02_e2e_t4_official_prompt/main/checkpoints/checkpoint.pt}"
HARDNESS_ROOT="${MASK_HARDNESS_ROOT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/hardness_base_t4_box}"
ABLATION_ROOT="${EDGETAM_MEMORY_ROOT:-${SAM2D_ROOT}/runs/edgetam_memory_ablation_v1}"
CENTRAL_CSV="${EDGETAM_MEMORY_SUMMARY_CSV:-${ABLATION_ROOT}/summary.csv}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-edgetam-memory-ablation-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"
SKIP_DONE="${EDGETAM_MEMORY_SKIP_DONE:-1}"

is_variant() {
  local candidate="$1" item
  for item in "${VARIANTS[@]}"; do
    [[ "${candidate}" == "${item}" ]] && return 0
  done
  return 1
}

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

configure_variant() {
  export TASK_EXPERIMENT_SUITE=edgetam_memory_v1
  export TASK_MANIFEST="${MANIFEST}"
  export BASE_CHECKPOINT="${BASE_CHECKPOINT}"
  export BASE_STAGE_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}"
  export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
  export SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}"
  export GPUS="${GPUS}"
  export WORLD_SIZE="${NPROC}"
  export TASK_MASK_ABLATION_V2=1
  export TASK_EDGETAM_MEMORY_ABLATION=1
  export TASK_SEED=250107256
  export TASK_EPOCHS=1
  export TASK_MAX_VIDEOS=0
  export TASK_NUM_FRAMES=4
  export TASK_TRAIN_BATCH_SIZE=2
  export TASK_MAX_NUM_OBJECTS=2
  export TASK_FREEZE_BATCHNORM=true
  export TASK_ENCODER_LR=0
  export TASK_ENCODER_LR_END=0
  export TASK_HEAD_LR=1.0e-6
  export TASK_HEAD_LR_END=1.0e-7
  export TASK_MEMORY_LR=3.0e-6
  export TASK_MEMORY_LR_END=3.0e-7
  export TASK_MEMORY_AUX_LR=1.0e-6
  export TASK_MEMORY_AUX_LR_END=1.0e-7
  export TASK_PERCEIVER_LR=1.0e-5
  export TASK_PERCEIVER_LR_END=1.0e-6
  export TASK_LR_WARMUP_FRACTION=0.05
  export TASK_LR_WARMUP_START_FACTOR=0.1
  export TASK_PROB_USE_POINT=1.0
  export TASK_PROB_USE_BOX=1.0
  export TASK_PROB_SAMPLE_GT=0.0
  export TASK_NUM_FRAMES_TO_CORRECT=1
  export TASK_RANDOM_CORRECTION_FRAMES=false
  export TASK_NUM_INIT_COND_FRAMES=1
  export TASK_RANDOM_INIT_COND_FRAMES=false
  export TASK_NUM_CORRECTION_POINTS=0
  export TASK_LAMBDA_IMG=0
  export TASK_LAMBDA_MEM=0
  export TASK_VIDEO_IDS_FILE=""
  export TASK_LOSS_OUTLIER_THRESHOLD=20
  export TASK_NUM_GLOBAL_LATENTS=0
  export TASK_NUM_2D_LATENTS=0
  export TASK_TEACHER_MODEL_CONFIG=""
  export TASK_TEACHER_CHECKPOINT=""

  case "$1" in
    R0_*|R1_*|R2_*|R3_*)
      export TASK_TRAIN_BATCH_SIZE=1
      export TASK_MAX_NUM_OBJECTS=3
      export TASK_TRAINABLE_MODE=image_encoder_mask_decoder_memory
      export TASK_ENCODER_LR=3.0e-7
      export TASK_ENCODER_LR_END=3.0e-8
      export TASK_LR_WARMUP_FRACTION=0.1
      export TASK_PROB_USE_POINT=0.5
      export TASK_PROB_USE_BOX=0.5
      export TASK_PROB_SAMPLE_GT=0.1
      export TASK_NUM_FRAMES_TO_CORRECT=2
      export TASK_RANDOM_CORRECTION_FRAMES=true
      export TASK_NUM_CORRECTION_POINTS=7
      export TASK_MEMORY_TOPOLOGY=edgetam_hybrid2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=official_pair
      export TASK_NUM_GLOBAL_LATENTS=256
      export TASK_NUM_2D_LATENTS=256
      ;;
  esac

  case "$1" in
    M0_sam2_mem4)
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_MEMORY_INITIALIZER=current
      export TASK_TRAINABLE_MODE=memory_only
      ;;
    M1_sam2_mem2)
      export TASK_MEMORY_TOPOLOGY=standard2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=current
      export TASK_TRAINABLE_MODE=memory_only
      ;;
    M2a_edgetam_hybrid2_official)
      export TASK_MEMORY_TOPOLOGY=edgetam_hybrid2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=official_pair
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_GLOBAL_LATENTS=256
      export TASK_NUM_2D_LATENTS=256
      ;;
    M2b_edgetam_hybrid2_current)
      export TASK_MEMORY_TOPOLOGY=edgetam_hybrid2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=current_pair
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_GLOBAL_LATENTS=256
      export TASK_NUM_2D_LATENTS=256
      ;;
    R0_edgetam_e2e_t4_task)
      ;;
    R1_edgetam_e2e_t4_imgkd)
      export TASK_LAMBDA_IMG=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    R2_edgetam_e2e_t4_imgmemkd)
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MEM=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    R3_edgetam_e2e_t8_imgmemkd)
      export TASK_NUM_FRAMES=8
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MEM=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    *)
      echo "[ERROR] Unknown EdgeTAM memory variant: $1" >&2
      return 2
      ;;
  esac
}

checkpoint_reached_epoch() {
  python - "$1" "${TASK_EPOCHS}" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("epoch", -1))
target = int(sys.argv[2])
print(f"checkpoint epoch: {epoch}; target: {target}")
raise SystemExit(0 if epoch >= target else 1)
PY
}

last_checkpoint() {
  local run_dir="$1"
  if [[ -f "${run_dir}/checkpoints/last.pt" ]]; then
    echo "${run_dir}/checkpoints/last.pt"
  else
    echo "${run_dir}/checkpoints/checkpoint.pt"
  fi
}

normalize_checkpoints() {
  local checkpoint_dir="$1"
  mkdir -p "${checkpoint_dir}"
  if [[ -f "${checkpoint_dir}/checkpoint.pt" && ! -L "${checkpoint_dir}/checkpoint.pt" ]]; then
    mv -f "${checkpoint_dir}/checkpoint.pt" "${checkpoint_dir}/last.pt"
  fi
  [[ -f "${checkpoint_dir}/last.pt" ]] || return 1
  ln -sfn last.pt "${checkpoint_dir}/checkpoint.pt"
}

mark_best_checkpoint() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/last.pt" ]] || return 1
  ln -sfn last.pt "${checkpoint_dir}/best.pt"
}

record_summary() {
  local variant_dir="$1" run_dir="$2"
  python tools/train/summarize_mask_finetune_ablations.py record \
    --variant-dir "${variant_dir}" \
    --stage-dir "${run_dir}" \
    --central-csv "${CENTRAL_CSV}"
}

audit_inputs() {
  python tools/train/audit_sam2_task_inputs.py \
    --manifest "${MANIFEST}" \
    --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
    --sav-root "${SAV_ROOT}" \
    --sample-videos "${AUDIT_SAMPLE_VIDEOS:-500}" \
    --compact
}

validate_common_paths() {
  local path
  for path in \
    "${MANIFEST}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt" \
    "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
    "${EDGETAM_ROOT}/sam2/modeling/perceiver.py" \
    "${TINYVIT_CHECKPOINT}" \
    "${SOURCE_STAGE1_CHECKPOINT}" \
    "${BASE_CHECKPOINT}" \
    "${CONFIG}"; do
    require_path "${path}" || return 1
  done
  git -C "${EDGETAM_ROOT}" merge-base --is-ancestor \
    "${EDGETAM_REQUIRED_COMMIT}" HEAD || {
      echo "[ERROR] EdgeTAM checkout must contain ${EDGETAM_REQUIRED_COMMIT}" >&2
      return 1
    }
  if [[ "${TASK_MEMORY_TOPOLOGY}" == "edgetam_hybrid2" ]]; then
    require_path "${EDGETAM_CHECKPOINT}" || return 1
  fi
  if [[ "${TASK_LAMBDA_IMG}" != "0" || "${TASK_LAMBDA_MEM}" != "0" ]]; then
    require_path "${TASK_TEACHER_MODEL_CONFIG}" || return 1
    require_path "${TASK_TEACHER_CHECKPOINT}" || return 1
  fi
  if [[ -n "${TASK_VIDEO_IDS_FILE}" ]]; then
    require_path "${TASK_VIDEO_IDS_FILE}" || return 1
  fi
}

ensure_edgetam_checkpoint() {
  [[ "${TASK_MEMORY_TOPOLOGY}" == "edgetam_hybrid2" ]] || return 0
  mkdir -p "$(dirname "${EDGETAM_CHECKPOINT}")"
  exec 8>"${EDGETAM_CHECKPOINT}.download.lock" || return 1
  flock 8 || return 1
  if [[ ! -f "${EDGETAM_CHECKPOINT}" ]]; then
    OUT="${EDGETAM_CHECKPOINT}" EDGETAM_ROOT="${EDGETAM_ROOT}" \
      scripts/company/17_download_edgetam_checkpoint.sh || return 1
  fi
  flock -u 8
}

train_variant() {
  local name="$1" variant_dir="${ABLATION_ROOT}/$1" run_dir="${ABLATION_ROOT}/$1/main"
  local checkpoint
  mkdir -p "${run_dir}/wandb" "${run_dir}/checkpoints"
  checkpoint="$(last_checkpoint "${run_dir}")"
  if checkpoint_reached_epoch "${checkpoint}"; then
    echo "skip completed training: ${name}"
  else
    touch "${run_dir}/.full_eval_required"
    echo "===== Training ${name} on GPUs ${GPUS} ====="
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    PYTHONPATH="${REPO_ROOT}:${EDGETAM_ROOT}:${SAM2_TRAINING_ROOT}:${PYTHONPATH:-}" \
    SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT}" \
    EDGETAM_ROOT="${EDGETAM_ROOT}" \
    EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT}" \
    TASK_RUN_DIR="${run_dir}" \
    TASK_STAGE_NAME="${name}" \
    TASK_MANIFEST="${MANIFEST}" \
    SAV_ROOT="${SAV_ROOT}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS}" \
    TASK_PRINT_EVERY="${PRINT_EVERY}" \
    TASK_LOG_EVERY="${LOG_EVERY}" \
    PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}" \
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_LOSS_EMA_BETA=0.98 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
      torchrun --standalone --nproc-per-node "${NPROC}" \
        tools/train/run_sam2_task_training.py \
        --config "${CONFIG}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-name "${name}" \
        --wandb-dir "${run_dir}/wandb" || return 1
  fi
  normalize_checkpoints "${run_dir}/checkpoints" || return 1
  checkpoint_reached_epoch "${run_dir}/checkpoints/last.pt" || return 1
  record_summary "${variant_dir}" "${run_dir}"
}

evaluate_split() {
  local name="$1" run_dir="$2" split="$3" eval_skip_done="$4"
  MODEL_FAMILY=edgetam \
  STAGE1_CHECKPOINT="${run_dir}/checkpoints/last.pt" \
  EDGETAM_ROOT="${EDGETAM_ROOT}" \
  EDGETAM_CONFIG="${run_dir}/resolved_config.yaml" \
  SAM2_ROOT="${SAM2_TRAINING_ROOT}" \
  EXPERIMENT="${name}" \
  RUN_DIR="${run_dir}" \
  SAV_ROOT="${SAV_ROOT}" \
  SAV_SPLIT="${split}" \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  SKIP_DONE="${eval_skip_done}" \
  CLEAN_PREDICTIONS=1 \
    scripts/company/25_benchmark_stage1_sav_test.sh
}

evaluate_variant() {
  local name="$1" variant_dir="${ABLATION_ROOT}/$1" run_dir="${ABLATION_ROOT}/$1/main"
  local eval_skip_done="${SKIP_DONE}"
  [[ -f "${run_dir}/.full_eval_required" ]] && eval_skip_done=0
  echo "===== Full SA-V val: ${name} ====="
  evaluate_split "${name}" "${run_dir}" sav_val "${eval_skip_done}" || return 1
  mark_best_checkpoint "${run_dir}/checkpoints" || return 1
  echo "===== Full SA-V test: ${name} ====="
  evaluate_split "${name}" "${run_dir}" sav_test "${eval_skip_done}" || return 1
  if [[ "${WANDB_MODE}" == "online" ]]; then
    env -u WANDB_RUN_ID python tools/train/log_task_eval_to_wandb.py \
      --run-file "${run_dir}/wandb/wandb_run.json" \
      --metrics "sav_val=${run_dir}/sav_val_box_benchmark/metrics.csv" \
      --metrics "sav_test=${run_dir}/sav_test_box_benchmark/metrics.csv" || return 1
  fi
  rm -f "${run_dir}/.full_eval_required"
  record_summary "${variant_dir}" "${run_dir}"
}

acquire_lock() {
  local lock_file="${ABLATION_ROOT}/${VARIANT}/.pipeline.lock"
  mkdir -p "$(dirname "${lock_file}")"
  command -v flock >/dev/null 2>&1 || return 1
  exec 9>"${lock_file}" || return 1
  if ! flock -n 9; then
    echo "Another node owns ${lock_file}; waiting to avoid collision."
    flock 9 || return 1
  fi
  echo "Pipeline lock acquired: ${lock_file}"
}

summarize_all() {
  python tools/train/summarize_mask_finetune_ablations.py scan \
    --root "${ABLATION_ROOT}" \
    --central-csv "${CENTRAL_CSV}"
}

STATUS=0
case "${ACTION}" in
  describe)
    is_variant "${VARIANT}" && configure_variant "${VARIANT}"
    STATUS="$?"
    if [[ "${STATUS}" -eq 0 ]]; then
      echo "Variant: ${VARIANT}"
      echo "Topology/layers: ${TASK_MEMORY_TOPOLOGY}/${TASK_MEMORY_LAYERS}"
      echo "Initializer: ${TASK_MEMORY_INITIALIZER}"
      echo "Trainable mode: ${TASK_TRAINABLE_MODE}"
      echo "T/global batch: ${TASK_NUM_FRAMES}/$((TASK_TRAIN_BATCH_SIZE * NPROC))"
      echo "Prompt point/box/GT: ${TASK_PROB_USE_POINT}/${TASK_PROB_USE_BOX}/${TASK_PROB_SAMPLE_GT}"
      echo "Correction frames/points: ${TASK_NUM_FRAMES_TO_CORRECT}/${TASK_NUM_CORRECTION_POINTS}"
      echo "KD image/memory: ${TASK_LAMBDA_IMG}/${TASK_LAMBDA_MEM}"
    fi
    ;;
  run)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Set a valid variant after run" >&2
      STATUS=2
    else
      configure_variant "${VARIANT}" || STATUS="$?"
      if [[ "${STATUS}" -eq 0 ]]; then
        acquire_lock && ensure_edgetam_checkpoint && validate_common_paths && audit_inputs && \
          record_summary "${ABLATION_ROOT}/${VARIANT}" "${ABLATION_ROOT}/${VARIANT}/main" && \
          train_variant "${VARIANT}" && evaluate_variant "${VARIANT}"
        STATUS="$?"
      fi
    fi
    ;;
  summarize)
    summarize_all
    STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {list|describe VARIANT|run VARIANT|summarize}" >&2
    STATUS=2
    ;;
esac

echo "EdgeTAM memory ablation status: ${STATUS}"
echo "Run root: ${ABLATION_ROOT}"
echo "Central summary: ${CENTRAL_CSV}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
