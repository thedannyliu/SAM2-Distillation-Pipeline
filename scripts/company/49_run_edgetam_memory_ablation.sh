#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-list}"
VARIANT="${2:-}"
VARIANTS=(
  MX1_slot4_decoder_kd_3ep
  MX2_slot8_decoder_kd_3ep
  MX3_slot4_sharedkv_kd_3ep
  MX4_slot8_sharedkv_kd_3ep
  MX5_slot8_decoder_t8_logits2_5ep
  MX6_slot8_sharedkv_t8_mem1_5ep
  MX7_slot8_sharedkv_t8_mem4_5ep
  MX8_slot8_sharedkv_t8_mem1_logits4_5ep
  MX9_slot8_sharedkv_r4_t8_5ep
  MX10_slot8_sharedkv_r8_t8_5ep
  MX11_slot8_sharedkv_r16_t8_5ep
  MX12_slot8_sharedkv_r8_ptr8_t8_5ep
  MX13_slot8_r2_mean_screen3ep
  MX14_slot8_r4_mean_screen3ep
  MX15_slot8_r8_mean_screen3ep
  MX16_slot8_r16_mean_screen3ep
  MX17_slot8_r8_mean_ptr4_screen3ep
  MX18_slot8_r8_mean_ptr8_screen3ep
  MX19_slot8_r8_latest_ptr8_screen3ep
  MX20_slot8_r8_recency050_ptr8_screen3ep
  MX21_slot8_r8_recency025_ptr8_screen3ep
  MX22_slot8_r8_recency075_ptr8_screen3ep
  MX23_slot4_r8_mean_ptr8_screen3ep
  MX24_slot6_r8_mean_ptr8_screen3ep
  MX25_slot8_r8_mean_ptr8_objkd025_screen3ep
  MX26_slot8_r8_mean_ptr8_objkd100_screen3ep
  MX27_slot8_min2_r8_mean_ptr8_screen3ep
  MX28_slot8_min3_r8_mean_ptr8_screen3ep
  MO0_mem4_task_dense8_5ep
  MO1_mem2_task_dense8_5ep
  MO2_mem2_logits_dense8_5ep
  MO3_mem2_memlogits_dense8_5ep
  M0_sam2_mem4
  M1_sam2_mem2
  M2a_edgetam_hybrid2_official
  M2b_edgetam_hybrid2_current
  R0_edgetam_e2e_t4_task
  R1_edgetam_e2e_t4_imgkd
  R2_edgetam_e2e_t4_imgmemkd
  R3_edgetam_e2e_t8_imgmemkd
  C0_coherent_m0mem_align
  C1_partial_m0mem_align
  C2_coherent_m0mem_joint2ep
  C3_coherent_m0mem_staged
  D1_staged_image_align_1ep
  D2_staged_temporal_2ep
  D3_staged_t8_refine_1ep
  J1_joint_behavior_2ep
  J2_joint_temporal_refine_1ep
  J3_joint_t8_refine_1ep
  S0_scratch_temporal_task_2ep
  S1_scratch_behavior_2ep
  S2_scratch_t8_refine_1ep
  W1_official_image_align_2ep
  W2a_official_logits_5ep
  W2b_official_memlogits_5ep
  W2c_official_full_5ep
  W3a_official_logits_t8_3ep
  W3b_official_memlogits_t8_3ep
  W3c_official_full_t8_3ep
  K1a_m0_task_5ep
  K1b_m0_logits_5ep
  K1c_m0_memlogits_5ep
  K1d_m0_full_5ep
  K2a_m0_task_t8_2ep
  K2b_m0_logits_t8_2ep
  K2c_m0_memlogits_t8_2ep
  K2d_m0_full_t8_2ep
  Q0_official_identity_t8_1ep
  Q1_tinyvit_overfit16_t8_500ep
  Q2_tinyvit_paper_scaled_sav_t8_5ep
  EM1_t4_official_temporal_2ep
  EM2_t8_joint_edgetam_5ep
  EM3_t16_memory_refine_2ep
)
FULL_DATA_TOOL="tools/experiments/sam2_full_data_50.py"
if [[ -f "${FULL_DATA_TOOL}" ]]; then
  mapfile -t FULL_DATA_VARIANTS < <(python "${FULL_DATA_TOOL}" list)
  VARIANTS+=("${FULL_DATA_VARIANTS[@]}")
fi

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
SAM2_BPLUS_MODEL_CONFIG="${SAM2_BPLUS_MODEL_CONFIG:-${SAM2_TRAINING_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml}"
SAM2_BPLUS_CHECKPOINT="${SAM2_BPLUS_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_base_plus.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
BEST_TV21_RUN="${BEST_TV21_RUN:-${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/tv21/main}"
BEST_TV21_CHECKPOINT="${BEST_TV21_CHECKPOINT:-${BEST_TV21_RUN}/checkpoints/best.pt}"
BEST_TV21_CONFIG="${BEST_TV21_CONFIG:-${BEST_TV21_RUN}/resolved_config.yaml}"
BEST_TV11_RUN="${BEST_TV11_RUN:-${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/tv11/main}"
BEST_TV11_CHECKPOINT="${BEST_TV11_CHECKPOINT:-${BEST_TV11_RUN}/checkpoints/best.pt}"
TV11_TINYVIT_CHECKPOINT="${TV11_TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors}"
TV11_SOURCE_STAGE1_CHECKPOINT="${TV11_SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_size_scaling/tv11_proj_sam21l_msehr_cos025/checkpoints/best.pt}"
MX2_SLOT8_CHECKPOINT="${MX2_SLOT8_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_object_slots_v1/MX2_slot8_decoder_kd_3ep/main/checkpoints/last.pt}"
MX5_SLOT8_CHECKPOINT="${MX5_SLOT8_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/checkpoints/last.pt}"
MX5_SLOT8_CONFIG="${MX5_SLOT8_CONFIG:-${SAM2D_ROOT}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/resolved_config.yaml}"
A02_BASE_CHECKPOINT="${A02_BASE_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/A02_e2e_t4_official_prompt/main/checkpoints/checkpoint.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${A02_BASE_CHECKPOINT}}"
M0_RUN_DIR="${M0_RUN_DIR:-${SAM2D_ROOT}/runs/edgetam_memory_ablation_v1/M0_sam2_mem4/main}"
M0_CHECKPOINT="${M0_CHECKPOINT:-${M0_RUN_DIR}/checkpoints/last.pt}"
M0_CONFIG="${M0_CONFIG:-${M0_RUN_DIR}/resolved_config.yaml}"
BEHAVIOR_ROOT="${EDGETAM_BEHAVIOR_ROOT:-${SAM2D_ROOT}/runs/edgetam_tinyvit21_behavior_v4}"
E1_CHECKPOINT="${E1_CHECKPOINT:-${BEHAVIOR_ROOT}/E1_a02_official_nonimage/main/checkpoints/last.pt}"
OFFICIAL_EDGETAM_CONFIG="${OFFICIAL_EDGETAM_CONFIG:-${EDGETAM_ROOT}/sam2/configs/edgetam.yaml}"
HARDNESS_ROOT="${MASK_HARDNESS_ROOT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/hardness_base_t4_box}"
if [[ "${VARIANT}" == EM* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/tinyvit21_edgetam_memory_v1"
  DEFAULT_WANDB_PROJECT="tinyvit21-edgetam-memory-v1"
elif [[ "${VARIANT}" == FD* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_full_data_50_v1"
  DEFAULT_WANDB_PROJECT="sam2-full-data-50-v1"
elif [[ "${VARIANT}" == MX1[3-9]_* || "${VARIANT}" == MX2[0-8]_* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_multiplex_overnight_v4"
  DEFAULT_WANDB_PROJECT="sam2-multiplex-overnight-v4"
elif [[ "${VARIANT}" == MX9_* || "${VARIANT}" == MX10_* || \
      "${VARIANT}" == MX11_* || "${VARIANT}" == MX12_* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_object_slots_v3"
  DEFAULT_WANDB_PROJECT="sam2-object-slots-v3"
elif [[ "${VARIANT}" == MX5_* || "${VARIANT}" == MX6_* || \
      "${VARIANT}" == MX7_* || "${VARIANT}" == MX8_* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_object_slots_v2"
  DEFAULT_WANDB_PROJECT="sam2-object-slots-v2"
elif [[ "${VARIANT}" == MX* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_object_slots_v1"
  DEFAULT_WANDB_PROJECT="sam2-object-slots-v1"
elif [[ "${VARIANT}" == MO* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/sam2_multiobject_training_v1"
  DEFAULT_WANDB_PROJECT="sam2-multiobject-training-v1"
elif [[ "${VARIANT}" == C* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/edgetam_memory_recovery_v2"
  DEFAULT_WANDB_PROJECT="edgetam-memory-recovery-v2"
elif [[ "${VARIANT}" == D* || "${VARIANT}" == J* || "${VARIANT}" == S* ]]; then
  DEFAULT_ABLATION_ROOT="${BEHAVIOR_ROOT}"
  DEFAULT_WANDB_PROJECT="edgetam-tinyvit21-behavior-v4"
elif [[ "${VARIANT}" == Q* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/edgetam_recipe_diagnostics_v5"
  DEFAULT_WANDB_PROJECT="edgetam-recipe-diagnostics-v5"
elif [[ "${VARIANT}" == W* || "${VARIANT}" == K* ]]; then
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/weekend_72h_v1/edge"
  DEFAULT_WANDB_PROJECT="weekend-72h-edgetam-v1"
else
  DEFAULT_ABLATION_ROOT="${SAM2D_ROOT}/runs/edgetam_memory_ablation_v1"
  DEFAULT_WANDB_PROJECT="edgetam-memory-ablation-v1"
fi
ABLATION_ROOT="${EDGETAM_MEMORY_ROOT:-${DEFAULT_ABLATION_ROOT}}"
CENTRAL_CSV="${EDGETAM_MEMORY_SUMMARY_CSV:-${ABLATION_ROOT}/summary.csv}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-${DEFAULT_WANDB_PROJECT}}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"
SKIP_DONE="${EDGETAM_MEMORY_SKIP_DONE:-1}"
GATE_MAX_VIDEOS="${EDGETAM_GATE_MAX_VIDEOS:-32}"
GATE_MIN_JF="${EDGETAM_GATE_MIN_JF:-60}"
GATE_MAX_JF_DROP="${EDGETAM_GATE_MAX_JF_DROP:-10}"
GATE_MAX_IMAGE_DROP="${EDGETAM_GATE_MAX_IMAGE_DROP:-0.005}"
GATE_MIN_JF_RETENTION="${EDGETAM_GATE_MIN_JF_RETENTION:-0}"
GATE_ENFORCE="${EDGETAM_GATE_ENFORCE:-1}"
EVAL_MODE="${EDGETAM_EVAL_MODE:-full}"
GATE_REFERENCE_CHECKPOINT="${EDGETAM_GATE_REFERENCE_CHECKPOINT:-${M0_CHECKPOINT}}"
GATE_REFERENCE_CONFIG="${EDGETAM_GATE_REFERENCE_CONFIG:-${M0_CONFIG}}"
GATE_REFERENCE_TAG="${EDGETAM_GATE_REFERENCE_TAG:-m0}"
GATE_REFERENCE_BUCKET_SIZE="${EDGETAM_GATE_REFERENCE_BUCKET_SIZE:-8}"
GATE_REFERENCE_BUCKET_MIN_OBJECTS="${EDGETAM_GATE_REFERENCE_BUCKET_MIN_OBJECTS:-4}"

is_variant() {
  local candidate="$1" item
  for item in "${VARIANTS[@]}"; do
    [[ "${candidate}" == "${item}" ]] && return 0
  done
  return 1
}

is_recovery_variant() {
  [[ "$1" == C* ]]
}

is_behavior_variant() {
  [[ "$1" == D* || "$1" == J* || "$1" == S* || "$1" == W* || "$1" == K* || "$1" == Q* ]]
}

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

require_passed_gate() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[ERROR] Missing prerequisite gate: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "pass":
    raise SystemExit(f"[ERROR] Prerequisite gate did not pass: {path}")
print(f"Prerequisite gate: PASS | {path}")
PY
}

configure_full_data_variant() {
  local rendered key value
  rendered="$(python "${FULL_DATA_TOOL}" env "$1")" || return 1
  while IFS=$'\t' read -r key value; do
    case "${key}" in
      FD_*|TASK_*)
        printf -v "${key}" '%s' "${value}"
        export "${key?}"
        ;;
      *)
        echo "[ERROR] Unsupported full-data setting: ${key}" >&2
        return 2
        ;;
    esac
  done <<< "${rendered}"

  case "${FD_BASE_PROFILE}" in
    mx5)
      export BASE_CHECKPOINT="${MX5_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      ;;
    tv21)
      export BASE_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      ;;
    tv11)
      export BASE_CHECKPOINT="${BEST_TV11_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export SOURCE_STAGE1_CHECKPOINT="${TV11_SOURCE_STAGE1_CHECKPOINT}"
      export BASE_STAGE_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}"
      export TINYVIT_MODEL_NAME="tiny_vit_11m_224.dist_in22k_ft_in1k"
      export TINYVIT_CHECKPOINT="${TV11_TINYVIT_CHECKPOINT}"
      ;;
    *)
      echo "[ERROR] Unknown full-data base profile: ${FD_BASE_PROFILE}" >&2
      return 2
      ;;
  esac

  case "${FD_DATA_COHORT}" in
    all)
      export TASK_VIDEO_IDS_FILE=""
      ;;
    dense4|dense8)
      export TASK_VIDEO_IDS_FILE="${ABLATION_ROOT}/cohorts/${FD_DATA_COHORT}_train_ids.txt"
      ;;
    *)
      echo "[ERROR] Unknown full-data cohort: ${FD_DATA_COHORT}" >&2
      return 2
      ;;
  esac

  if [[ "${TASK_MEMORY_TOPOLOGY}" == "edgetam_hybrid2" ]]; then
    export TASK_TEACHER_MODEL_CONFIG="${OFFICIAL_EDGETAM_CONFIG}"
    export TASK_TEACHER_CHECKPOINT="${EDGETAM_CHECKPOINT}"
  else
    export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
    export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
  fi
}

configure_variant() {
  local local_source=""
  if [[ "$1" == EM* ]]; then
    export TASK_EXPERIMENT_SUITE=tinyvit21_edgetam_memory_v1
  elif [[ "$1" == FD* ]]; then
    export TASK_EXPERIMENT_SUITE=sam2_full_data_50_v1
  elif [[ "$1" == MX1[3-9]_* || "$1" == MX2[0-8]_* ]]; then
    export TASK_EXPERIMENT_SUITE=sam2_multiplex_overnight_v4
  elif [[ "$1" == MX9_* || "$1" == MX10_* || \
        "$1" == MX11_* || "$1" == MX12_* ]]; then
    export TASK_EXPERIMENT_SUITE=sam2_object_slots_v3
  elif [[ "$1" == MX5_* || "$1" == MX6_* || \
        "$1" == MX7_* || "$1" == MX8_* ]]; then
    export TASK_EXPERIMENT_SUITE=sam2_object_slots_v2
  elif [[ "$1" == MX* ]]; then
    export TASK_EXPERIMENT_SUITE=sam2_object_slots_v1
  elif [[ "$1" == MO* ]]; then
    export TASK_EXPERIMENT_SUITE=sam2_multiobject_training_v1
  elif [[ "$1" == Q* ]]; then
    export TASK_EXPERIMENT_SUITE=edgetam_recipe_diagnostics_v5
  elif [[ "$1" == W* || "$1" == K* ]]; then
    export TASK_EXPERIMENT_SUITE=weekend_72h_v1
  elif is_recovery_variant "$1"; then
    export TASK_EXPERIMENT_SUITE=edgetam_memory_recovery_v2
  elif is_behavior_variant "$1"; then
    export TASK_EXPERIMENT_SUITE=edgetam_tinyvit21_behavior_v4
  else
    export TASK_EXPERIMENT_SUITE=edgetam_memory_v1
  fi
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
  export TASK_LAMBDA_TASK=1
  export TASK_LAMBDA_MASK_LOGITS=0
  export TASK_LAMBDA_OBJ_PTR=0
  export TASK_VIDEO_IDS_FILE=""
  export TASK_LOSS_OUTLIER_THRESHOLD=20
  export TASK_NUM_GLOBAL_LATENTS=0
  export TASK_NUM_2D_LATENTS=0
  export TASK_TEACHER_MODEL_CONFIG=""
  export TASK_TEACHER_CHECKPOINT=""
  export TASK_MEMORY_LAYOUT=legacy
  export TASK_OBJECT_SLOT_MODE=none
  export TASK_OBJECT_SLOT_COUNT=0
  export TASK_OBJECT_SLOT_MIN_OBJECTS=4
  export TASK_OBJECT_RESIDUAL_RANK=0
  export TASK_OBJECT_POINTER_RESIDUAL_RANK=0
  export TASK_OBJECT_RESIDUAL_TEMPORAL_POOL=mean
  export TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY=0.5
  export TASK_OFFICIAL_EDGETAM_MODEL=0
  export STUDENT_FAMILY=tinyvit
  export TASK_LOSS_MASK_WEIGHT=20
  export TASK_LOSS_DICE_WEIGHT=1
  export TASK_WEIGHT_DECAY=0.05

  case "$1" in
    EM1_*|EM2_*|EM3_*)
      export BASE_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAIN_BATCH_SIZE=1
      export TASK_MAX_NUM_OBJECTS=3
      export TASK_MEMORY_TOPOLOGY=edgetam_hybrid2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=official_temporal
      export TASK_MEMORY_LAYOUT=official
      export TASK_NUM_GLOBAL_LATENTS=256
      export TASK_NUM_2D_LATENTS=256
      export TASK_LR_WARMUP_FRACTION=0.05
      export TASK_PROB_USE_POINT=0.5
      export TASK_PROB_USE_BOX=0.5
      export TASK_PROB_SAMPLE_GT=0.1
      export TASK_NUM_FRAMES_TO_CORRECT=2
      export TASK_RANDOM_CORRECTION_FRAMES=true
      export TASK_NUM_CORRECTION_POINTS=7
      export TASK_TEACHER_MODEL_CONFIG="${OFFICIAL_EDGETAM_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${EDGETAM_CHECKPOINT}"
      export TASK_LAMBDA_TASK=1
      export TASK_LAMBDA_IMG=0
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=2
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MO*|MX*|FD*)
      export BASE_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAIN_BATCH_SIZE=1
      export TASK_MAX_NUM_OBJECTS=8
      export TASK_TRAINABLE_MODE=mask_decoder_memory
      export TASK_MEMORY_INITIALIZER=current
      export TASK_MEMORY_LAYOUT=legacy
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=5
      export TASK_MEMORY_LR=5.0e-7
      export TASK_MEMORY_LR_END=5.0e-8
      export TASK_MEMORY_AUX_LR=2.5e-7
      export TASK_MEMORY_AUX_LR_END=2.5e-8
      export TASK_LR_WARMUP_FRACTION=0.05
      export TASK_PROB_USE_POINT=0.5
      export TASK_PROB_USE_BOX=0.5
      export TASK_PROB_SAMPLE_GT=0.1
      export TASK_NUM_FRAMES_TO_CORRECT=1
      export TASK_RANDOM_CORRECTION_FRAMES=false
      export TASK_NUM_CORRECTION_POINTS=0
      export TASK_VIDEO_IDS_FILE="${ABLATION_ROOT}/cohorts/dense8_train_ids.txt"
      export TASK_TEACHER_MODEL_CONFIG=""
      export TASK_TEACHER_CHECKPOINT=""
      export TASK_LAMBDA_TASK=1
      export TASK_LAMBDA_IMG=0
      export TASK_LAMBDA_MEM=0
      export TASK_LAMBDA_MASK_LOGITS=0
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
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
    C0_*|C1_*|C2_*|C3_*)
      export TASK_TRAIN_BATCH_SIZE=1
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_TOPOLOGY=edgetam_hybrid2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=official_temporal
      export TASK_MEMORY_LAYOUT=official
      export TASK_NUM_GLOBAL_LATENTS=256
      export TASK_NUM_2D_LATENTS=256
      export TASK_LAMBDA_IMG=0
      export TASK_LAMBDA_MEM=1
      export TASK_TEACHER_MODEL_CONFIG="${M0_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${M0_CHECKPOINT}"
      export EDGETAM_GATE_MAX_VIDEOS="${GATE_MAX_VIDEOS}"
      export EDGETAM_GATE_MIN_JF="${GATE_MIN_JF}"
      ;;
    D1_*|D2_*|D3_*|J1_*|J2_*|J3_*|S0_*|S1_*|S2_*|W*|K*|Q*)
      export BASE_CHECKPOINT="${E1_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAIN_BATCH_SIZE=1
      export TASK_MAX_NUM_OBJECTS=2
      export TASK_MEMORY_TOPOLOGY=edgetam_hybrid2
      export TASK_MEMORY_LAYERS=2
      export TASK_MEMORY_INITIALIZER=current_full
      export TASK_MEMORY_LAYOUT=official
      export TASK_NUM_GLOBAL_LATENTS=256
      export TASK_NUM_2D_LATENTS=256
      export TASK_LR_WARMUP_FRACTION=0.05
      export TASK_PROB_USE_POINT=1.0
      export TASK_PROB_USE_BOX=1.0
      export TASK_PROB_SAMPLE_GT=0.0
      export TASK_NUM_FRAMES_TO_CORRECT=1
      export TASK_RANDOM_CORRECTION_FRAMES=false
      export TASK_NUM_CORRECTION_POINTS=0
      export TASK_TEACHER_MODEL_CONFIG="${OFFICIAL_EDGETAM_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${EDGETAM_CHECKPOINT}"
      export TASK_LAMBDA_TASK=1
      export TASK_LOSS_OUTLIER_THRESHOLD=20
      ;;
  esac

  case "$1" in
    EM1_t4_official_temporal_2ep)
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=2
      export TASK_MEMORY_LR=1.0e-6
      export TASK_MEMORY_LR_END=1.0e-7
      export TASK_MEMORY_AUX_LR=3.0e-7
      export TASK_MEMORY_AUX_LR_END=3.0e-8
      export TASK_PERCEIVER_LR=3.0e-6
      export TASK_PERCEIVER_LR_END=3.0e-7
      ;;
    EM2_t8_joint_edgetam_5ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/EM1_t4_official_temporal_2ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=image_encoder_memory_perceiver
      export TASK_MEMORY_INITIALIZER=current_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=5
      export TASK_ENCODER_LR=1.0e-7
      export TASK_ENCODER_LR_END=1.0e-8
      export TASK_MEMORY_LR=5.0e-7
      export TASK_MEMORY_LR_END=5.0e-8
      export TASK_MEMORY_AUX_LR=1.5e-7
      export TASK_MEMORY_AUX_LR_END=1.5e-8
      export TASK_PERCEIVER_LR=1.5e-6
      export TASK_PERCEIVER_LR_END=1.5e-7
      export TASK_LAMBDA_IMG=1
      ;;
    EM3_t16_memory_refine_2ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/EM2_t8_joint_edgetam_5ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_INITIALIZER=current_full
      export TASK_NUM_FRAMES=16
      export TASK_EPOCHS=2
      export TASK_MEMORY_LR=2.5e-7
      export TASK_MEMORY_LR_END=2.5e-8
      export TASK_MEMORY_AUX_LR=7.5e-8
      export TASK_MEMORY_AUX_LR_END=7.5e-9
      export TASK_PERCEIVER_LR=7.5e-7
      export TASK_PERCEIVER_LR_END=7.5e-8
      export TASK_LAMBDA_MEM=0.25
      export TASK_LAMBDA_MASK_LOGITS=1
      ;;
    FD*)
      configure_full_data_variant "$1" || return $?
      ;;
    MX1_slot4_decoder_kd_3ep)
      export TASK_EPOCHS=3
      export TASK_TRAINABLE_MODE=object_slot_decoder
      export TASK_MEMORY_AUX_LR=3.0e-5
      export TASK_MEMORY_AUX_LR_END=3.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=decoder
      export TASK_OBJECT_SLOT_COUNT=4
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX2_slot8_decoder_kd_3ep)
      export TASK_EPOCHS=3
      export TASK_TRAINABLE_MODE=object_slot_decoder
      export TASK_MEMORY_AUX_LR=3.0e-5
      export TASK_MEMORY_AUX_LR_END=3.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=decoder
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX3_slot4_sharedkv_kd_3ep)
      export TASK_EPOCHS=3
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=1.0e-5
      export TASK_MEMORY_LR_END=1.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=4
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=0.25
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX4_slot8_sharedkv_kd_3ep)
      export TASK_EPOCHS=3
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=1.0e-5
      export TASK_MEMORY_LR_END=1.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=0.25
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX5_slot8_decoder_t8_logits2_5ep)
      export BASE_CHECKPOINT="${MX2_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_EPOCHS=5
      export TASK_NUM_FRAMES=8
      export TASK_TRAINABLE_MODE=object_slot_decoder
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=decoder
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MASK_LOGITS=2
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX6_slot8_sharedkv_t8_mem1_5ep)
      export BASE_CHECKPOINT="${MX2_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_EPOCHS=5
      export TASK_NUM_FRAMES=8
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=1.0e-5
      export TASK_MEMORY_LR_END=1.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=1
      export TASK_LAMBDA_MASK_LOGITS=2
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX7_slot8_sharedkv_t8_mem4_5ep)
      export BASE_CHECKPOINT="${MX2_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_EPOCHS=5
      export TASK_NUM_FRAMES=8
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=1.0e-5
      export TASK_MEMORY_LR_END=1.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=4
      export TASK_LAMBDA_MASK_LOGITS=2
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX8_slot8_sharedkv_t8_mem1_logits4_5ep)
      export BASE_CHECKPOINT="${MX2_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_EPOCHS=5
      export TASK_NUM_FRAMES=8
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=1.0e-5
      export TASK_MEMORY_LR_END=1.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=1
      export TASK_LAMBDA_MASK_LOGITS=4
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    MX9_slot8_sharedkv_r4_t8_5ep|\
    MX10_slot8_sharedkv_r8_t8_5ep|\
    MX11_slot8_sharedkv_r16_t8_5ep|\
    MX12_slot8_sharedkv_r8_ptr8_t8_5ep)
      export BASE_CHECKPOINT="${MX5_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_EPOCHS=5
      export TASK_NUM_FRAMES=8
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=3.0e-5
      export TASK_MEMORY_LR_END=3.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=1
      export TASK_LAMBDA_MASK_LOGITS=2
      export TASK_LAMBDA_OBJ_PTR=0
      case "$1" in
        MX9_*) export TASK_OBJECT_RESIDUAL_RANK=4 ;;
        MX10_*) export TASK_OBJECT_RESIDUAL_RANK=8 ;;
        MX11_*) export TASK_OBJECT_RESIDUAL_RANK=16 ;;
        MX12_*)
          export TASK_OBJECT_RESIDUAL_RANK=8
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          ;;
      esac
      ;;
    MX13_slot8_r2_mean_screen3ep|\
    MX14_slot8_r4_mean_screen3ep|\
    MX15_slot8_r8_mean_screen3ep|\
    MX16_slot8_r16_mean_screen3ep|\
    MX17_slot8_r8_mean_ptr4_screen3ep|\
    MX18_slot8_r8_mean_ptr8_screen3ep|\
    MX19_slot8_r8_latest_ptr8_screen3ep|\
    MX20_slot8_r8_recency050_ptr8_screen3ep|\
    MX21_slot8_r8_recency025_ptr8_screen3ep|\
    MX22_slot8_r8_recency075_ptr8_screen3ep|\
    MX23_slot4_r8_mean_ptr8_screen3ep|\
    MX24_slot6_r8_mean_ptr8_screen3ep|\
    MX25_slot8_r8_mean_ptr8_objkd025_screen3ep|\
    MX26_slot8_r8_mean_ptr8_objkd100_screen3ep|\
    MX27_slot8_min2_r8_mean_ptr8_screen3ep|\
    MX28_slot8_min3_r8_mean_ptr8_screen3ep)
      export BASE_CHECKPOINT="${MX5_SLOT8_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_EPOCHS=3
      export TASK_NUM_FRAMES=8
      export TASK_TRAINABLE_MODE=object_slot_shared_kv
      export TASK_MEMORY_LR=3.0e-5
      export TASK_MEMORY_LR_END=3.0e-6
      export TASK_MEMORY_AUX_LR=1.0e-5
      export TASK_MEMORY_AUX_LR_END=1.0e-6
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      export TASK_OBJECT_SLOT_MODE=shared_kv
      export TASK_OBJECT_SLOT_COUNT=8
      export TASK_OBJECT_SLOT_MIN_OBJECTS=4
      export TASK_OBJECT_RESIDUAL_RANK=8
      export TASK_OBJECT_POINTER_RESIDUAL_RANK=0
      export TASK_OBJECT_RESIDUAL_TEMPORAL_POOL=mean
      export TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY=0.5
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MEM=1
      export TASK_LAMBDA_MASK_LOGITS=2
      export TASK_LAMBDA_OBJ_PTR=0
      case "$1" in
        MX13_*) export TASK_OBJECT_RESIDUAL_RANK=2 ;;
        MX14_*) export TASK_OBJECT_RESIDUAL_RANK=4 ;;
        MX15_*) export TASK_OBJECT_RESIDUAL_RANK=8 ;;
        MX16_*) export TASK_OBJECT_RESIDUAL_RANK=16 ;;
        MX17_*) export TASK_OBJECT_POINTER_RESIDUAL_RANK=4 ;;
        MX18_*) export TASK_OBJECT_POINTER_RESIDUAL_RANK=8 ;;
        MX19_*)
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          export TASK_OBJECT_RESIDUAL_TEMPORAL_POOL=latest
          ;;
        MX20_*)
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          export TASK_OBJECT_RESIDUAL_TEMPORAL_POOL=recency
          ;;
        MX21_*)
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          export TASK_OBJECT_RESIDUAL_TEMPORAL_POOL=recency
          export TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY=0.25
          ;;
        MX22_*)
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          export TASK_OBJECT_RESIDUAL_TEMPORAL_POOL=recency
          export TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY=0.75
          ;;
        MX23_*)
          export TASK_OBJECT_SLOT_COUNT=4
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          ;;
        MX24_*)
          export TASK_OBJECT_SLOT_COUNT=6
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          ;;
        MX25_*)
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          export TASK_LAMBDA_OBJ_PTR=0.25
          ;;
        MX26_*)
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          export TASK_LAMBDA_OBJ_PTR=1
          ;;
        MX27_*)
          export TASK_OBJECT_SLOT_MIN_OBJECTS=2
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          ;;
        MX28_*)
          export TASK_OBJECT_SLOT_MIN_OBJECTS=3
          export TASK_OBJECT_POINTER_RESIDUAL_RANK=8
          ;;
      esac
      ;;
    MO0_mem4_task_dense8_5ep)
      export TASK_MEMORY_TOPOLOGY=standard4
      export TASK_MEMORY_LAYERS=4
      ;;
    MO1_mem2_task_dense8_5ep)
      export TASK_MEMORY_TOPOLOGY=standard2
      export TASK_MEMORY_LAYERS=2
      ;;
    MO2_mem2_logits_dense8_5ep)
      export TASK_MEMORY_TOPOLOGY=standard2
      export TASK_MEMORY_LAYERS=2
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MASK_LOGITS=1
      ;;
    MO3_mem2_memlogits_dense8_5ep)
      export TASK_MEMORY_TOPOLOGY=standard2
      export TASK_MEMORY_LAYERS=2
      export TASK_TEACHER_MODEL_CONFIG="${BEST_TV21_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${BEST_TV21_CHECKPOINT}"
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_MEM=0.5
      ;;
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
    C0_coherent_m0mem_align)
      export TASK_LAMBDA_TASK=0
      ;;
    C1_partial_m0mem_align)
      export TASK_LAMBDA_TASK=0
      export TASK_MEMORY_INITIALIZER=official_pair
      export TASK_MEMORY_LAYOUT=legacy
      ;;
    C2_coherent_m0mem_joint2ep)
      export TASK_EPOCHS=2
      ;;
    C3_coherent_m0mem_staged)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/C0_coherent_m0mem_align/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_MEMORY_INITIALIZER=current_full
      ;;
    D1_staged_image_align_1ep)
      export TASK_TRAINABLE_MODE=image_encoder_only
      export TASK_NUM_FRAMES=2
      export TASK_EPOCHS=1
      export TASK_ENCODER_LR=3.0e-7
      export TASK_ENCODER_LR_END=3.0e-8
      export TASK_MEMORY_AUX_LR=1.0e-7
      export TASK_MEMORY_AUX_LR_END=1.0e-8
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MEM=0
      export TASK_LAMBDA_MASK_LOGITS=1
      ;;
    D2_staged_temporal_2ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/D1_staged_image_align_1ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=2
      export TASK_MEMORY_LR=3.0e-7
      export TASK_MEMORY_LR_END=3.0e-8
      export TASK_MEMORY_AUX_LR=1.0e-7
      export TASK_MEMORY_AUX_LR_END=1.0e-8
      export TASK_PERCEIVER_LR=1.0e-6
      export TASK_PERCEIVER_LR_END=1.0e-7
      export TASK_LAMBDA_IMG=0
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    D3_staged_t8_refine_1ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/D2_staged_temporal_2ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=1
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MEMORY_LR=1.5e-7
      export TASK_MEMORY_LR_END=1.5e-8
      export TASK_MEMORY_AUX_LR=5.0e-8
      export TASK_MEMORY_AUX_LR_END=5.0e-9
      export TASK_PERCEIVER_LR=5.0e-7
      export TASK_PERCEIVER_LR_END=5.0e-8
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    J1_joint_behavior_2ep)
      export TASK_TRAINABLE_MODE=image_encoder_memory_perceiver
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=2
      export TASK_ENCODER_LR=1.5e-7
      export TASK_ENCODER_LR_END=1.5e-8
      export TASK_MEMORY_LR=3.0e-7
      export TASK_MEMORY_LR_END=3.0e-8
      export TASK_MEMORY_AUX_LR=1.0e-7
      export TASK_MEMORY_AUX_LR_END=1.0e-8
      export TASK_PERCEIVER_LR=1.0e-6
      export TASK_PERCEIVER_LR_END=1.0e-7
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    J2_joint_temporal_refine_1ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/J1_joint_behavior_2ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=1
      export TASK_MEMORY_LR=1.5e-7
      export TASK_MEMORY_LR_END=1.5e-8
      export TASK_MEMORY_AUX_LR=5.0e-8
      export TASK_MEMORY_AUX_LR_END=5.0e-9
      export TASK_PERCEIVER_LR=5.0e-7
      export TASK_PERCEIVER_LR_END=5.0e-8
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    J3_joint_t8_refine_1ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/J2_joint_temporal_refine_1ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=1
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MEMORY_LR=1.5e-7
      export TASK_MEMORY_LR_END=1.5e-8
      export TASK_MEMORY_AUX_LR=5.0e-8
      export TASK_MEMORY_AUX_LR_END=5.0e-9
      export TASK_PERCEIVER_LR=5.0e-7
      export TASK_PERCEIVER_LR_END=5.0e-8
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    S0_scratch_temporal_task_2ep)
      export BASE_CHECKPOINT="${A02_BASE_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_INITIALIZER=scratch_temporal
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=2
      export TASK_LR_WARMUP_FRACTION=0.1
      export TASK_MEMORY_LR=3.0e-6
      export TASK_MEMORY_LR_END=3.0e-7
      export TASK_MEMORY_AUX_LR=1.0e-6
      export TASK_MEMORY_AUX_LR_END=1.0e-7
      export TASK_PERCEIVER_LR=1.0e-5
      export TASK_PERCEIVER_LR_END=1.0e-6
      export TASK_TEACHER_MODEL_CONFIG=""
      export TASK_TEACHER_CHECKPOINT=""
      ;;
    S1_scratch_behavior_2ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/S0_scratch_temporal_task_2ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=2
      export TASK_MEMORY_LR=1.0e-6
      export TASK_MEMORY_LR_END=1.0e-7
      export TASK_MEMORY_AUX_LR=3.0e-7
      export TASK_MEMORY_AUX_LR_END=3.0e-8
      export TASK_PERCEIVER_LR=3.0e-6
      export TASK_PERCEIVER_LR_END=3.0e-7
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    S2_scratch_t8_refine_1ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/S1_scratch_behavior_2ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=1
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MEMORY_LR=5.0e-7
      export TASK_MEMORY_LR_END=5.0e-8
      export TASK_MEMORY_AUX_LR=1.5e-7
      export TASK_MEMORY_AUX_LR_END=1.5e-8
      export TASK_PERCEIVER_LR=1.5e-6
      export TASK_PERCEIVER_LR_END=1.5e-7
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0.1
      ;;
    W1_official_image_align_2ep)
      export TASK_TRAINABLE_MODE=image_encoder_only
      export TASK_NUM_FRAMES=2
      export TASK_EPOCHS=2
      export TASK_ENCODER_LR=3.0e-7
      export TASK_ENCODER_LR_END=3.0e-8
      export TASK_MEMORY_AUX_LR=1.0e-7
      export TASK_MEMORY_AUX_LR_END=1.0e-8
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MASK_LOGITS=1
      ;;
    W2a_official_logits_5ep|W2b_official_memlogits_5ep|W2c_official_full_5ep)
      export BASE_CHECKPOINT="${ABLATION_ROOT}/W1_official_image_align_2ep/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=image_encoder_memory_perceiver
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=5
      export TASK_ENCODER_LR=1.0e-7
      export TASK_ENCODER_LR_END=1.0e-8
      export TASK_MEMORY_LR=3.0e-7
      export TASK_MEMORY_LR_END=3.0e-8
      export TASK_MEMORY_AUX_LR=1.0e-7
      export TASK_MEMORY_AUX_LR_END=1.0e-8
      export TASK_PERCEIVER_LR=1.0e-6
      export TASK_PERCEIVER_LR_END=1.0e-7
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MASK_LOGITS=1
      if [[ "$1" == W2b_* || "$1" == W2c_* ]]; then
        export TASK_LAMBDA_MEM=0.5
      fi
      if [[ "$1" == W2c_* ]]; then
        export TASK_LAMBDA_OBJ_PTR=0.1
      fi
      ;;
    W3a_official_logits_t8_3ep|W3b_official_memlogits_t8_3ep|W3c_official_full_t8_3ep)
      local_source="${1/W3/W2}"
      local_source="${local_source/_t8_3ep/_5ep}"
      export BASE_CHECKPOINT="${ABLATION_ROOT}/${local_source}/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=3
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MEMORY_LR=1.5e-7
      export TASK_MEMORY_LR_END=1.5e-8
      export TASK_MEMORY_AUX_LR=5.0e-8
      export TASK_MEMORY_AUX_LR_END=5.0e-9
      export TASK_PERCEIVER_LR=5.0e-7
      export TASK_PERCEIVER_LR_END=5.0e-8
      export TASK_LAMBDA_MASK_LOGITS=1
      if [[ "$1" == W3b_* || "$1" == W3c_* ]]; then
        export TASK_LAMBDA_MEM=0.5
      fi
      if [[ "$1" == W3c_* ]]; then
        export TASK_LAMBDA_OBJ_PTR=0.1
      fi
      ;;
    K1a_m0_task_5ep|K1b_m0_logits_5ep|K1c_m0_memlogits_5ep|K1d_m0_full_5ep)
      export BASE_CHECKPOINT="${A02_BASE_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_INITIALIZER=official_temporal
      export TASK_NUM_FRAMES=4
      export TASK_EPOCHS=5
      export TASK_MEMORY_LR=1.0e-6
      export TASK_MEMORY_LR_END=1.0e-7
      export TASK_MEMORY_AUX_LR=3.0e-7
      export TASK_MEMORY_AUX_LR_END=3.0e-8
      export TASK_PERCEIVER_LR=3.0e-6
      export TASK_PERCEIVER_LR_END=3.0e-7
      export TASK_TEACHER_MODEL_CONFIG="${M0_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${M0_CHECKPOINT}"
      if [[ "$1" == K1b_* || "$1" == K1c_* || "$1" == K1d_* ]]; then
        export TASK_LAMBDA_MASK_LOGITS=1
      fi
      if [[ "$1" == K1c_* || "$1" == K1d_* ]]; then
        export TASK_LAMBDA_MEM=0.5
      fi
      if [[ "$1" == K1d_* ]]; then
        export TASK_LAMBDA_OBJ_PTR=0.1
      fi
      ;;
    K2a_m0_task_t8_2ep|K2b_m0_logits_t8_2ep|K2c_m0_memlogits_t8_2ep|K2d_m0_full_t8_2ep)
      local_source="${1/K2/K1}"
      local_source="${local_source/_t8_2ep/_5ep}"
      export BASE_CHECKPOINT="${ABLATION_ROOT}/${local_source}/main/checkpoints/last.pt"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_INITIALIZER=current_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=2
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MEMORY_LR=5.0e-7
      export TASK_MEMORY_LR_END=5.0e-8
      export TASK_MEMORY_AUX_LR=1.5e-7
      export TASK_MEMORY_AUX_LR_END=1.5e-8
      export TASK_PERCEIVER_LR=1.5e-6
      export TASK_PERCEIVER_LR_END=1.5e-7
      export TASK_TEACHER_MODEL_CONFIG="${M0_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${M0_CHECKPOINT}"
      if [[ "$1" == K2b_* || "$1" == K2c_* || "$1" == K2d_* ]]; then
        export TASK_LAMBDA_MASK_LOGITS=1
      fi
      if [[ "$1" == K2c_* || "$1" == K2d_* ]]; then
        export TASK_LAMBDA_MEM=0.5
      fi
      if [[ "$1" == K2d_* ]]; then
        export TASK_LAMBDA_OBJ_PTR=0.1
      fi
      ;;
    Q0_official_identity_t8_1ep)
      export BASE_CHECKPOINT="${EDGETAM_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_OFFICIAL_EDGETAM_MODEL=1
      export STUDENT_FAMILY=repvit
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_INITIALIZER=current_full
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=1
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MEMORY_LR=1.0e-7
      export TASK_MEMORY_LR_END=1.0e-8
      export TASK_MEMORY_AUX_LR=3.0e-8
      export TASK_MEMORY_AUX_LR_END=3.0e-9
      export TASK_PERCEIVER_LR=3.0e-7
      export TASK_PERCEIVER_LR_END=3.0e-8
      export TASK_TEACHER_MODEL_CONFIG=""
      export TASK_TEACHER_CHECKPOINT=""
      ;;
    Q1_tinyvit_overfit16_t8_500ep)
      export BASE_CHECKPOINT="${A02_BASE_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_SEED=2501072
      export TASK_TRAINABLE_MODE=memory_perceiver_full
      export TASK_MEMORY_INITIALIZER=official_temporal
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=500
      export TASK_VIDEO_IDS_FILE="${ABLATION_ROOT}/overfit16_train_ids.txt"
      export TASK_MEMORY_LR=3.0e-6
      export TASK_MEMORY_LR_END=3.0e-7
      export TASK_MEMORY_AUX_LR=1.0e-6
      export TASK_MEMORY_AUX_LR_END=1.0e-7
      export TASK_PERCEIVER_LR=1.0e-5
      export TASK_PERCEIVER_LR_END=1.0e-6
      export TASK_LAMBDA_MEM=0.5
      export TASK_LAMBDA_MASK_LOGITS=1
      export TASK_LAMBDA_OBJ_PTR=0
      ;;
    Q2_tinyvit_paper_scaled_sav_t8_5ep)
      export BASE_CHECKPOINT="${A02_BASE_CHECKPOINT}"
      export PREVIOUS_TASK_CHECKPOINT="${BASE_CHECKPOINT}"
      export TASK_TRAINABLE_MODE=image_encoder_mask_decoder_memory
      export TASK_MEMORY_INITIALIZER=official_temporal
      export TASK_NUM_FRAMES=8
      export TASK_EPOCHS=5
      export TASK_VIDEO_IDS_FILE="${HARDNESS_ROOT}/eligible_t8.txt"
      export TASK_MAX_NUM_OBJECTS=3
      export TASK_ENCODER_LR=9.375e-7
      export TASK_ENCODER_LR_END=9.375e-8
      export TASK_MEMORY_LR=4.6875e-6
      export TASK_MEMORY_LR_END=4.6875e-7
      export TASK_MEMORY_AUX_LR=4.6875e-6
      export TASK_MEMORY_AUX_LR_END=4.6875e-7
      export TASK_PERCEIVER_LR=4.6875e-6
      export TASK_PERCEIVER_LR_END=4.6875e-7
      export TASK_PROB_USE_POINT=0.5
      export TASK_PROB_USE_BOX=0.5
      export TASK_PROB_SAMPLE_GT=0.1
      export TASK_NUM_FRAMES_TO_CORRECT=2
      export TASK_RANDOM_CORRECTION_FRAMES=true
      export TASK_NUM_CORRECTION_POINTS=7
      export TASK_LAMBDA_IMG=1
      export TASK_LAMBDA_MEM=1
      export TASK_TEACHER_MODEL_CONFIG="${SAM2_BPLUS_MODEL_CONFIG}"
      export TASK_TEACHER_CHECKPOINT="${SAM2_BPLUS_CHECKPOINT}"
      export TASK_LOSS_MASK_WEIGHT=1
      export TASK_LOSS_DICE_WEIGHT=20
      export TASK_WEIGHT_DECAY=0.1
      ;;
    *)
      echo "[ERROR] Unknown EdgeTAM memory variant: $1" >&2
      return 2
      ;;
  esac

  if [[ -n "${TASK_EPOCHS_OVERRIDE:-}" ]]; then
    export TASK_EPOCHS="${TASK_EPOCHS_OVERRIDE}"
  fi
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
  if [[ "${EDGETAM_SKIP_INPUT_AUDIT:-0}" == "1" ]]; then
    echo "Skip per-variant input audit; suite-level audit already passed."
    return 0
  fi
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
  if [[ "${TASK_OFFICIAL_EDGETAM_MODEL}" == "1" ]]; then
    require_path "${OFFICIAL_EDGETAM_CONFIG}" || return 1
  fi
  if [[ "${TASK_LAMBDA_IMG}" != "0" || \
        "${TASK_LAMBDA_MEM}" != "0" || \
        "${TASK_LAMBDA_MASK_LOGITS}" != "0" || \
        "${TASK_LAMBDA_OBJ_PTR}" != "0" ]]; then
    require_path "${TASK_TEACHER_MODEL_CONFIG}" || return 1
    require_path "${TASK_TEACHER_CHECKPOINT}" || return 1
  fi
  if [[ -n "${TASK_VIDEO_IDS_FILE}" ]]; then
    require_path "${TASK_VIDEO_IDS_FILE}" || return 1
  fi
  if [[ "${VARIANT}" == "C3_coherent_m0mem_staged" ]]; then
    require_passed_gate \
      "${ABLATION_ROOT}/C0_coherent_m0mem_align/main/gate_status.json" || return 1
  fi
}

ensure_variant_inputs() {
  if [[ "${VARIANT}" == MO* || "${VARIANT}" == MX* || \
        ( "${VARIANT}" == FD* && "${FD_DATA_COHORT:-all}" == dense* ) ]]; then
    local cohort_root="${ABLATION_ROOT}/cohorts" dense_min=8 cohort_name=dense8
    if [[ "${VARIANT}" == FD* ]]; then
      cohort_name="${FD_DATA_COHORT}"
      dense_min="${FD_DATA_COHORT#dense}"
    fi
    mkdir -p "${cohort_root}"
    exec 7>"${cohort_root}/.${cohort_name}.lock" || return 1
    flock 7 || return 1
    if [[ ! -s "${TASK_VIDEO_IDS_FILE}" ]]; then
      python tools/data/select_sav_dense_training_videos.py \
        --manifest "${MANIFEST}" \
        --sav-root "${SAV_ROOT}" \
        --output-video-ids "${TASK_VIDEO_IDS_FILE}" \
        --out-csv "${cohort_root}/${cohort_name}_index.csv" \
        --out-summary "${cohort_root}/${cohort_name}_summary.json" \
        --min-objects "${dense_min}" \
        --min-dense-frames 4 \
        --target-samples "${MO_TRAIN_SAMPLES:-50337}" \
        --seed "${TASK_SEED}" || return 1
    fi
    flock -u 7
  fi
  if [[ "${VARIANT}" == "Q1_tinyvit_overfit16_t8_500ep" ]]; then
    mkdir -p "${ABLATION_ROOT}"
    python tools/experiments/sample_video_gate.py \
      --input "${HARDNESS_ROOT}/eligible_t8.txt" \
      --output "${TASK_VIDEO_IDS_FILE}" \
      --count 16
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
    OFFICIAL_EDGETAM_CONFIG="${OFFICIAL_EDGETAM_CONFIG}" \
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
  MAX_VIDEOS=0 \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  SKIP_DONE="${eval_skip_done}" \
  CLEAN_PREDICTIONS=1 \
    scripts/company/25_benchmark_stage1_sav_test.sh
}

run_gate_benchmark() {
  local name="$1" checkpoint="$2" model_config="$3" run_dir="$4"
  local benchmark_root="$5"
  MODEL_FAMILY=edgetam \
  STAGE1_CHECKPOINT="${checkpoint}" \
  EDGETAM_ROOT="${EDGETAM_ROOT}" \
  EDGETAM_CONFIG="${model_config}" \
  SAM2_ROOT="${SAM2_TRAINING_ROOT}" \
  EXPERIMENT="${name}" \
  RUN_DIR="${run_dir}" \
  BENCH_ROOT="${benchmark_root}" \
  AGGREGATE_CSV="${ABLATION_ROOT}/gate_metrics.csv" \
  SAV_ROOT="${SAV_ROOT}" \
  SAV_SPLIT=sav_val \
  VIDEO_LIST_FILE="${ABLATION_ROOT}/gate_sav_val_${GATE_MAX_VIDEOS}.txt" \
  MAX_VIDEOS=0 \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  SKIP_DONE="${SKIP_DONE}" \
  CLEAN_PREDICTIONS=1 \
    scripts/company/25_benchmark_stage1_sav_test.sh
}

ensure_gate_reference() {
  local reference_dir="$1" reference_benchmark="$2"
  require_path "${GATE_REFERENCE_CHECKPOINT}" || return 1
  require_path "${GATE_REFERENCE_CONFIG}" || return 1
  mkdir -p "${reference_dir}"
  exec 6>"${ABLATION_ROOT}/.gate_reference.lock" || return 1
  flock 6 || return 1
  python tools/experiments/sample_video_gate.py \
    --input "${SAV_ROOT}/sav_val/sav_val.txt" \
    --output "${ABLATION_ROOT}/gate_sav_val_${GATE_MAX_VIDEOS}.txt" \
    --count "${GATE_MAX_VIDEOS}" || return 1
  echo "===== ${GATE_REFERENCE_TAG} fixed mini-val reference (${GATE_MAX_VIDEOS} videos) ====="
  VOS_BUCKET_SIZE="${GATE_REFERENCE_BUCKET_SIZE}" \
  VOS_BUCKET_MIN_OBJECTS="${GATE_REFERENCE_BUCKET_MIN_OBJECTS}" \
    run_gate_benchmark \
    "${GATE_REFERENCE_TAG}_gate_reference" \
    "${GATE_REFERENCE_CHECKPOINT}" \
    "${GATE_REFERENCE_CONFIG}" \
    "${reference_dir}" \
    "${reference_benchmark}" || return 1
  flock -u 6
}

evaluate_gate() {
  local name="$1" variant_dir="${ABLATION_ROOT}/$1"
  local run_dir="${variant_dir}/main"
  local reference_dir="${ABLATION_ROOT}/_gate_reference_${GATE_REFERENCE_TAG}"
  local reference_benchmark="${reference_dir}/sav_val_gate${GATE_MAX_VIDEOS}_box_benchmark"
  local candidate_benchmark="${run_dir}/sav_val_gate${GATE_MAX_VIDEOS}_box_benchmark"
  local reference_metrics="${reference_benchmark}/metrics.csv"
  local candidate_metrics="${candidate_benchmark}/metrics.csv"
  local gate_status=0

  ensure_gate_reference "${reference_dir}" "${reference_benchmark}" || return 1

  echo "===== Temporal compatibility gate: ${name} ====="
  run_gate_benchmark \
    "${name}_gate" \
    "${run_dir}/checkpoints/last.pt" \
    "${run_dir}/resolved_config.yaml" \
    "${run_dir}" \
    "${candidate_benchmark}" || return 1

  python tools/experiments/check_sav_memory_gate.py \
    --metrics "${candidate_metrics}" \
    --reference-metrics "${reference_metrics}" \
    --out-json "${run_dir}/gate_status.json" \
    --min-jf "${GATE_MIN_JF}" \
    --min-jf-retention "${GATE_MIN_JF_RETENTION}" \
    --max-jf-drop "${GATE_MAX_JF_DROP}" \
    --max-miou-drop "${GATE_MAX_IMAGE_DROP}" \
    --max-ap-drop "${GATE_MAX_IMAGE_DROP}" || gate_status="$?"

  if [[ "${WANDB_MODE}" == "online" ]]; then
    env -u WANDB_RUN_ID python tools/train/log_task_eval_to_wandb.py \
      --run-file "${run_dir}/wandb/wandb_run.json" \
      --metrics "sav_val_gate${GATE_MAX_VIDEOS}=${candidate_metrics}" || return 1
  fi
  record_summary "${variant_dir}" "${run_dir}"
  if [[ "${gate_status}" -ne 0 ]]; then
    echo "[GATE FAIL] ${name} failed the temporal compatibility gate." >&2
    if [[ "${GATE_ENFORCE}" == "1" ]]; then
      return "${gate_status}"
    fi
    echo "[CONTINUE] Gate enforcement is disabled for this screen."
  fi
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
      echo "Initializer/layout: ${TASK_MEMORY_INITIALIZER}/${TASK_MEMORY_LAYOUT}"
      echo "Trainable mode: ${TASK_TRAINABLE_MODE}"
      echo "Epochs/max objects: ${TASK_EPOCHS}/${TASK_MAX_NUM_OBJECTS}"
      echo "T/global batch: ${TASK_NUM_FRAMES}/$((TASK_TRAIN_BATCH_SIZE * NPROC))"
      echo "Prompt point/box/GT: ${TASK_PROB_USE_POINT}/${TASK_PROB_USE_BOX}/${TASK_PROB_SAMPLE_GT}"
      echo "Correction frames/points: ${TASK_NUM_FRAMES_TO_CORRECT}/${TASK_NUM_CORRECTION_POINTS}"
      echo "Loss task/image/memory/logits/obj: ${TASK_LAMBDA_TASK}/${TASK_LAMBDA_IMG}/${TASK_LAMBDA_MEM}/${TASK_LAMBDA_MASK_LOGITS}/${TASK_LAMBDA_OBJ_PTR}"
      echo "Object residual spatial/pointer rank: ${TASK_OBJECT_RESIDUAL_RANK}/${TASK_OBJECT_POINTER_RESIDUAL_RANK}"
      echo "Object residual temporal pool/decay: ${TASK_OBJECT_RESIDUAL_TEMPORAL_POOL}/${TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY}"
      echo "Object slot mode/count/min: ${TASK_OBJECT_SLOT_MODE}/${TASK_OBJECT_SLOT_COUNT}/${TASK_OBJECT_SLOT_MIN_OBJECTS}"
      echo "Base: ${BASE_CHECKPOINT}"
      echo "Teacher: ${TASK_TEACHER_CHECKPOINT:-none}"
    fi
    ;;
  run)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Set a valid variant after run" >&2
      STATUS=2
    else
      configure_variant "${VARIANT}" || STATUS="$?"
      if [[ "${STATUS}" -eq 0 ]]; then
        acquire_lock && ensure_edgetam_checkpoint && ensure_variant_inputs && validate_common_paths && audit_inputs && \
          record_summary "${ABLATION_ROOT}/${VARIANT}" "${ABLATION_ROOT}/${VARIANT}/main" && \
          train_variant "${VARIANT}"
        STATUS="$?"
        if [[ "${STATUS}" -eq 0 ]] && {
          is_recovery_variant "${VARIANT}" || [[ "${EVAL_MODE}" == "gate" ]]
        }; then
          evaluate_gate "${VARIANT}"
          STATUS="$?"
        fi
        if [[ "${STATUS}" -eq 0 && "${EVAL_MODE}" == "full" ]]; then
          evaluate_variant "${VARIANT}"
          STATUS="$?"
        elif [[ "${STATUS}" -eq 0 && "${EVAL_MODE}" != "gate" && \
                "${EVAL_MODE}" != "none" ]]; then
          echo "[ERROR] EDGETAM_EVAL_MODE must be full, gate, or none" >&2
          STATUS=2
        fi
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
