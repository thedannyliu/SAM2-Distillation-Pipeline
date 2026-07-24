#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
case "${LANE}" in
  tinyvit|repvit) ;;
  *)
    echo "Usage: $0 {tinyvit|repvit}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
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
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"

if [[ "${LANE}" == "tinyvit" ]]; then
  SUITE="tinyvit_capacity_freeze_v2"
  RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/${SUITE}}"
  WANDB_PROJECT="${WANDB_PROJECT:-tinyvit-capacity-freeze-v2}"
else
  SUITE="repvit_task_finetune_v2"
  RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/${SUITE}}"
  WANDB_PROJECT="${WANDB_PROJECT:-repvit-task-finetune-v2}"
fi
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/selection.json"
FAILED=()
COMPLETED_CANDIDATES=()

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

first_checkpoint() {
  local path
  for path in "$@"; do
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  echo "$1"
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

run_stage() {
  local name="$1"
  local student_family="$2"
  local model_name="$3"
  local adapter_mode="$4"
  local student_checkpoint="$5"
  local source_stage1_checkpoint="$6"
  local previous_task_checkpoint="$7"
  local trainable_mode="$8"
  local epochs="$9"
  local frames="${10}"
  local encoder_lr="${11}"
  local encoder_lr_end="${12}"
  local head_lr="${13}"
  local head_lr_end="${14}"
  local freeze_batchnorm="${15}"
  local video_ids_file="${16:-}"
  local stage_dir="${RUN_ROOT}/${name}"
  local checkpoint_dir="${stage_dir}/checkpoints"

  echo
  echo "================================================================"
  echo "Starting: ${name}"
  echo "Family/model: ${student_family} / ${model_name}"
  echo "Mode: ${trainable_mode}; epochs: ${epochs}; frames: ${frames}"
  echo "Freeze BN: ${freeze_batchnorm}; W&B: ${WANDB_PROJECT}"
  echo "================================================================"

  if [[ -f "${stage_dir}/.pipeline_complete" && \
        -f "${stage_dir}/sav_val_box_benchmark/metrics.csv" && \
        -f "${stage_dir}/sav_test_box_benchmark/metrics.csv" ]]; then
    echo "skip completed formal stage: ${name}"
    COMPLETED_CANDIDATES+=("${name}=${stage_dir}")
    return 0
  fi

  for path in \
    "${student_checkpoint}" \
    "${source_stage1_checkpoint}"; do
    require_path "${path}" || return 1
  done
  if [[ -n "${previous_task_checkpoint}" ]]; then
    require_path "${previous_task_checkpoint}" || return 1
  fi
  if [[ -n "${video_ids_file}" ]]; then
    require_path "${video_ids_file}" || return 1
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
    STUDENT_FAMILY="${student_family}" \
    TINYVIT_MODEL_NAME="${model_name}" \
    TINYVIT_ADAPTER_MODE="${adapter_mode}" \
    TINYVIT_CHECKPOINT="${student_checkpoint}" \
    SOURCE_STAGE1_CHECKPOINT="${source_stage1_checkpoint}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS}" \
    PRINT_EVERY="${PRINT_EVERY}" \
    LOG_EVERY="${LOG_EVERY}" \
    STAGE1_NAME="${name}" \
    STAGE1_MODE="${trainable_mode}" \
    STAGE1_EPOCHS="${epochs}" \
    STAGE1_FRAMES="${frames}" \
    STAGE1_ENCODER_LR="${encoder_lr}" \
    STAGE1_ENCODER_LR_END="${encoder_lr_end}" \
    STAGE1_HEAD_LR="${head_lr}" \
    STAGE1_HEAD_LR_END="${head_lr_end}" \
    STAGE1_PREVIOUS_CHECKPOINT="${previous_task_checkpoint}" \
    TASK_EXPERIMENT_SUITE="${SUITE}" \
    TASK_MASK_ABLATION_V2=1 \
    TASK_TRAIN_BATCH_SIZE=1 \
    TASK_MAX_NUM_OBJECTS=2 \
    TASK_FREEZE_BATCHNORM="${freeze_batchnorm}" \
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
    TASK_VIDEO_IDS_FILE="${video_ids_file}" \
    TASK_EXPORT_STAGE_CHECKPOINT=1 \
    TASK_EVAL_SPLITS=sav_val,sav_test \
    SKIP_DONE=1 \
      scripts/company/39_run_sam2_task_finetune_3stage.sh stage1 || return 1

  normalize_checkpoints "${checkpoint_dir}" || return 1
  touch "${stage_dir}/.pipeline_complete"
  COMPLETED_CANDIDATES+=("${name}=${stage_dir}")
}

record_failure() {
  local name="$1" status="$2"
  FAILED+=("${name}:${status}")
  echo "${name} status: ${status}"
}

run_tinyvit_size() {
  local size="$1"
  local model_name adapter_mode student_checkpoint source_checkpoint base_task
  local frozen_name joint_name encoder_lr encoder_lr_end
  case "${size}" in
    tv5)
      model_name="tiny_vit_5m_224.dist_in22k_ft_in1k"
      adapter_mode="residual_dwconv"
      student_checkpoint="${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors"
      source_checkpoint="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_extra_adapter_cos/tv5_adapter_sam21l_msehr/checkpoints/best.pt"
      base_task=""
      encoder_lr="2.0e-7"
      encoder_lr_end="2.0e-8"
      ;;
    tv11)
      model_name="tiny_vit_11m_224.dist_in22k_ft_in1k"
      adapter_mode="projection"
      student_checkpoint="${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors"
      source_checkpoint="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_size_scaling/tv11_proj_sam21l_msehr_cos025/checkpoints/best.pt"
      base_task=""
      encoder_lr="1.5e-7"
      encoder_lr_end="1.5e-8"
      ;;
    tv21)
      model_name="tiny_vit_21m_512.dist_in22k_ft_in1k"
      adapter_mode="projection"
      student_checkpoint="${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
      source_checkpoint="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt"
      base_task="$(first_checkpoint \
        "${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/A02_e2e_t4_official_prompt/main/checkpoints/last.pt" \
        "${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/A02_e2e_t4_official_prompt/main/checkpoints/checkpoint.pt")"
      encoder_lr="1.0e-7"
      encoder_lr_end="1.0e-8"
      ;;
  esac
  frozen_name="${size}_F1_decmem_frozen_2ep"
  joint_name="${size}_F2_joint_low_1ep"

  run_stage \
    "${frozen_name}" tinyvit "${model_name}" "${adapter_mode}" \
    "${student_checkpoint}" "${source_checkpoint}" "${base_task}" \
    mask_decoder_memory 2 4 0 0 1.0e-6 1.0e-7 true "" || {
      record_failure "${frozen_name}" "$?"
      return 1
    }
  run_stage \
    "${joint_name}" tinyvit "${model_name}" "${adapter_mode}" \
    "${student_checkpoint}" "${source_checkpoint}" \
    "${RUN_ROOT}/${frozen_name}/checkpoints/last.pt" \
    image_encoder_mask_decoder_memory 1 4 \
    "${encoder_lr}" "${encoder_lr_end}" 5.0e-7 5.0e-8 true "" || {
      record_failure "${joint_name}" "$?"
      return 1
    }
}

run_repvit() {
  local model_name="repvit_m0_9.dist_450e_in1k"
  local student_checkpoint="${SAM2D_ROOT}/checkpoints/repvit/repvit_m0_9.dist_450e_in1k.safetensors"
  local source_root="${SAM2D_ROOT}/runs/repvit_stage1_v1/repvit_m09_proj_sam21l_msehr_cos025_l1010"
  local source_checkpoint
  local hard_videos="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/hardness_base_t4_box/eligible_t8.txt"
  source_checkpoint="$(first_checkpoint \
    "${source_root}/checkpoints/best.pt" \
    "${source_root}/checkpoints/last.pt" \
    "${source_root}/checkpoints/checkpoint.pt")"

  run_stage \
    repvit_P1_encoder_recovery_3ep repvit "${model_name}" projection \
    "${student_checkpoint}" "${source_checkpoint}" "" \
    image_encoder_only 3 2 1.0e-6 1.0e-7 1.0e-6 1.0e-7 true "" || {
      record_failure repvit_P1_encoder_recovery_3ep "$?"
      return 1
    }
  run_stage \
    repvit_P2_joint_frozenbn_2ep repvit "${model_name}" projection \
    "${student_checkpoint}" "${source_checkpoint}" \
    "${RUN_ROOT}/repvit_P1_encoder_recovery_3ep/checkpoints/last.pt" \
    image_encoder_mask_decoder_memory 2 4 \
    3.0e-7 3.0e-8 1.0e-6 1.0e-7 true "" || \
      record_failure repvit_P2_joint_frozenbn_2ep "$?"
  run_stage \
    repvit_P2b_joint_trainbn_1ep repvit "${model_name}" projection \
    "${student_checkpoint}" "${source_checkpoint}" \
    "${RUN_ROOT}/repvit_P1_encoder_recovery_3ep/checkpoints/last.pt" \
    image_encoder_mask_decoder_memory 1 4 \
    3.0e-7 3.0e-8 1.0e-6 1.0e-7 false "" || \
      record_failure repvit_P2b_joint_trainbn_1ep "$?"
  run_stage \
    repvit_P3_decmem_t8_refine_1ep repvit "${model_name}" projection \
    "${student_checkpoint}" "${source_checkpoint}" \
    "${RUN_ROOT}/repvit_P2_joint_frozenbn_2ep/checkpoints/last.pt" \
    mask_decoder_memory 1 8 0 0 5.0e-7 5.0e-8 true \
    "${hard_videos}" || \
      record_failure repvit_P3_decmem_t8_refine_1ep "$?"
}

write_summary() {
  local -a args=()
  local candidate
  if [[ "${#COMPLETED_CANDIDATES[@]}" -eq 0 ]]; then
    echo "No completed candidates; summary not written."
    return 1
  fi
  for candidate in "${COMPLETED_CANDIDATES[@]}"; do
    args+=(--candidate "${candidate}")
  done
  python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${SUMMARY_JSON}" \
    --out-csv "${SUMMARY_CSV}"
}

mkdir -p "${RUN_ROOT}"
for path in \
  "${MANIFEST}" \
  "${SAV_ROOT}/sav_val/sav_val.txt" \
  "${SAV_ROOT}/sav_test/sav_test.txt" \
  "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
  "${SAM2_CHECKPOINT}" \
  "${CONFIG}"; do
  require_path "${path}" || return 1 2>/dev/null || exit 1
done
wandb_preflight || return 1 2>/dev/null || exit 1

if [[ "${LANE}" == "tinyvit" ]]; then
  run_tinyvit_size tv5 || true
  run_tinyvit_size tv11 || true
  run_tinyvit_size tv21 || true
else
  run_repvit
fi
write_summary || FAILED+=("summary:1")

echo
echo "Backbone task expansion lane: ${LANE}"
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY_CSV}"
if [[ -f "${SUMMARY_CSV}" ]]; then
  cat "${SUMMARY_CSV}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed jobs: ${FAILED[*]}"
  return 1 2>/dev/null || exit 1
fi
echo "Lane status: 0"
return 0 2>/dev/null || exit 0
