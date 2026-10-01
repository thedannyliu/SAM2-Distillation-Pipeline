#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
STAGE="${2:-}"
BATCHES="${3:-}"

GPUS="${GPUS:-0,1,2,3}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"

SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
NEW_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
STAGED_ROOT="${STAGED_ROOT:-/group-volume/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${STAGED_ROOT}/manifests/sav_train_6fps_full.parquet}"
PROBE_TAG="${PROBE_TAG:-v1}"
PROBE_ROOT="${PROBE_ROOT:-${NEW_ROOT}/runs/edgetam_tv21_sam21l_v1/batch_probe/${PROBE_TAG}}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"

SAM2_MODEL_CONFIG="${SAM2_MODEL_CONFIG:-${SAM2_TRAINING_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${NEW_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${NEW_ROOT}/checkpoints/edgetam/edgetam.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${STAGED_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
BEST_TV21_CHECKPOINT="${BEST_TV21_CHECKPOINT:-${STAGED_ROOT}/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${STAGED_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"

TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
MAX_RESERVED_GIB="${MAX_RESERVED_GIB:-72}"

usage() {
  echo "Usage: $0 {describe|audit|probe STAGE [BATCHES]|all|summarize}"
  echo "Stages: image, t4, t8, t16"
  echo "Batch list example: 4,2,1"
}

describe() {
  echo "TV21 EdgeTAM + online SAM2.1-L batch-capacity probe"
  echo "GPUs: ${GPUS} (${NPROC} ranks)"
  echo "Manifest: ${MANIFEST}"
  echo "SA-V root: ${SAV_ROOT}"
  echo "TV21 checkpoint: ${BEST_TV21_CHECKPOINT}"
  echo "Teacher: ${SAM2_CHECKPOINT}"
  echo "EdgeTAM initializer: ${EDGETAM_CHECKPOINT}"
  echo "Output: ${PROBE_ROOT}"
  echo "HBM acceptance ceiling: ${MAX_RESERVED_GIB} GiB per GPU"
  echo "Default candidates: image=8,4,2; t4=4,2,1; t8=2,1; t16=2,1"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing required file: $1" >&2
    return 1
  fi
}

audit() {
  local status=0
  require_file "${MANIFEST}" || status=1
  require_file "${SAV_ROOT}/sav_val/sav_val.txt" || status=1
  require_file "${SAV_ROOT}/sav_test/sav_test.txt" || status=1
  require_file "${SAM2_TRAINING_ROOT}/training/model/sam2.py" || status=1
  require_file "${SAM2_MODEL_CONFIG}" || status=1
  require_file "${EDGETAM_ROOT}/sam2/modeling/perceiver.py" || status=1
  require_file "${EDGETAM_CHECKPOINT}" || status=1
  require_file "${SAM2_CHECKPOINT}" || status=1
  require_file "${TINYVIT_CHECKPOINT}" || status=1
  require_file "${BEST_TV21_CHECKPOINT}" || status=1
  require_file "${SOURCE_STAGE1_CHECKPOINT}" || status=1
  require_file "${CONFIG}" || status=1
  if [[ "${NPROC}" -ne 4 ]]; then
    echo "[ERROR] This probe requires exactly four GPU IDs; got ${GPUS}" >&2
    status=1
  fi
  if [[ "${status}" -eq 0 ]]; then
    python tools/train/audit_sam2_task_inputs.py \
      --manifest "${MANIFEST}" \
      --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
      --sav-root "${SAV_ROOT}" \
      --sample-videos 64 \
      --compact || status="$?"
  fi
  echo "Batch probe audit status: ${status}"
  return "${status}"
}

configure_stage() {
  export TASK_EPOCHS=1
  export TASK_MAX_NUM_OBJECTS=3
  export TASK_PROB_USE_POINT=0.5
  export TASK_PROB_USE_BOX=0.5
  export TASK_PROB_SAMPLE_GT=0.1
  export TASK_RANDOM_CORRECTION_FRAMES=true
  export TASK_NUM_CORRECTION_POINTS=7
  export TASK_LOSS_MASK_WEIGHT=20
  export TASK_LOSS_DICE_WEIGHT=1
  export TASK_WEIGHT_DECAY=0.1
  export TASK_LR_WARMUP_FRACTION=0.1
  export TASK_LR_WARMUP_START_FACTOR=0.1
  export TASK_ENCODER_LR=1.0e-6
  export TASK_ENCODER_LR_END=1.0e-7
  export TASK_HEAD_LR=5.0e-6
  export TASK_HEAD_LR_END=5.0e-7
  export TASK_MEMORY_LR=5.0e-6
  export TASK_MEMORY_LR_END=5.0e-7
  export TASK_MEMORY_AUX_LR=5.0e-6
  export TASK_MEMORY_AUX_LR_END=5.0e-7
  export TASK_PERCEIVER_LR=5.0e-6
  export TASK_PERCEIVER_LR_END=5.0e-7
  export TASK_LAMBDA_TASK=1
  export TASK_LAMBDA_IMG=0
  export TASK_LAMBDA_MEM=0
  export TASK_LAMBDA_MASK_LOGITS=0
  export TASK_LAMBDA_OBJ_PTR=0
  export TASK_TEACHER_MODEL_CONFIG=""
  export TASK_TEACHER_CHECKPOINT=""

  case "$1" in
    image)
      export TASK_NUM_FRAMES=1
      export TASK_MAX_VIDEOS="${IMAGE_PROBE_VIDEOS:-512}"
      export TASK_MAX_NUM_OBJECTS=8
      export TASK_TRAINABLE_MODE=image_encoder_mask_decoder
      export TASK_NUM_FRAMES_TO_CORRECT=1
      export TASK_LAMBDA_IMG=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    t4)
      export TASK_NUM_FRAMES=4
      export TASK_MAX_VIDEOS="${T4_PROBE_VIDEOS:-128}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES_TO_CORRECT=2
      export TASK_LAMBDA_MEM=1
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    t8)
      export TASK_NUM_FRAMES=8
      export TASK_MAX_VIDEOS="${T8_PROBE_VIDEOS:-64}"
      export TASK_TRAINABLE_MODE=image_encoder_mask_decoder_memory
      export TASK_NUM_FRAMES_TO_CORRECT=2
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MEM=1
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_CHECKPOINT}"
      ;;
    t16)
      export TASK_NUM_FRAMES=16
      export TASK_MAX_VIDEOS="${T16_PROBE_VIDEOS:-48}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES_TO_CORRECT=2
      ;;
    *)
      echo "[ERROR] Unknown stage: $1" >&2
      return 2
      ;;
  esac
}

default_batches() {
  case "$1" in
    image) echo "8,4,2" ;;
    t4) echo "4,2,1" ;;
    t8|t16) echo "2,1" ;;
    *) return 2 ;;
  esac
}

run_candidate() {
  local stage="$1" batch="$2"
  local name="${stage}_batch${batch}"
  local run_dir="${PROBE_ROOT}/${name}"
  local completed_ranks=0
  local status=0

  if [[ -d "${run_dir}" ]]; then
    completed_ranks="$(find "${run_dir}" -maxdepth 1 -type f -name 'capacity_rank*.json' | wc -l)"
  fi
  if [[ -f "${run_dir}/exit_code.txt" ]] && \
     [[ "$(<"${run_dir}/exit_code.txt")" == "0" ]] && \
     [[ "${completed_ranks}" -eq "${NPROC}" ]]; then
    echo "Skip completed probe: ${name}"
    return 0
  fi

  configure_stage "${stage}" || return $?
  mkdir -p "${run_dir}/wandb" "${run_dir}/checkpoints"
  echo "===== Probe ${name}: T=${TASK_NUM_FRAMES}, per-GPU batch=${batch}, global batch=$((batch * NPROC)) ====="

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
  TASK_PRINT_EVERY=1 \
  TASK_LOG_EVERY=1 \
  TASK_TRAIN_BATCH_SIZE="${batch}" \
  TASK_MASK_ABLATION_V2=1 \
  TASK_EDGETAM_MEMORY_ABLATION=1 \
  TASK_MEMORY_TOPOLOGY=edgetam_hybrid2 \
  TASK_MEMORY_LAYERS=2 \
  TASK_MEMORY_INITIALIZER=official_temporal \
  TASK_MEMORY_LAYOUT=official \
  TASK_NUM_GLOBAL_LATENTS=256 \
  TASK_NUM_2D_LATENTS=256 \
  TASK_FREEZE_BATCHNORM=true \
  TASK_CAPACITY_PROBE=1 \
  TASK_CAPACITY_WARMUP_STEPS=2 \
  TASK_LOSS_OUTLIER_THRESHOLD=0 \
  PREVIOUS_TASK_CHECKPOINT="${BEST_TV21_CHECKPOINT}" \
  SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
  TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
  SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
  WANDB_MODE=disabled \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    torchrun --standalone --nproc-per-node "${NPROC}" \
      tools/train/run_sam2_task_training.py \
      --config "${CONFIG}" \
      --wandb-project edgetam-tv21-batch-probe \
      --wandb-name "${name}" \
      --wandb-dir "${run_dir}/wandb" || status="$?"

  printf '%s\n' "${status}" > "${run_dir}/exit_code.txt"
  echo "Probe ${name} status: ${status}"
  return 0
}

probe_stage() {
  local stage="$1" batches="$2" batch
  if [[ -z "${stage}" ]]; then
    echo "[ERROR] probe requires a stage" >&2
    usage
    return 2
  fi
  batches="${batches:-$(default_batches "${stage}")}" || return $?
  IFS=, read -r -a batch_array <<< "${batches}"
  for batch in "${batch_array[@]}"; do
    if [[ ! "${batch}" =~ ^[1-9][0-9]*$ ]]; then
      echo "[ERROR] Invalid batch size: ${batch}" >&2
      return 2
    fi
    run_candidate "${stage}" "${batch}" || return $?
  done
}

summarize() {
  python tools/train/summarize_edgetam_batch_probe.py \
    --root "${PROBE_ROOT}" \
    --max-reserved-gib "${MAX_RESERVED_GIB}"
}

run_all() {
  audit || return $?
  probe_stage image "" || return $?
  probe_stage t4 "" || return $?
  probe_stage t8 "" || return $?
  probe_stage t16 "" || return $?
  summarize
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  audit)
    audit || STATUS="$?"
    ;;
  probe)
    probe_stage "${STAGE}" "${BATCHES}" || STATUS="$?"
    summarize || true
    ;;
  all)
    run_all || STATUS="$?"
    ;;
  summarize)
    summarize || STATUS="$?"
    ;;
  *)
    usage >&2
    STATUS=2
    ;;
esac

echo "TV21 EdgeTAM batch probe status: ${STATUS}"
echo "Probe root: ${PROBE_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
