#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
ACTION="${2:-run}"
case "${LANE}" in
  edge_official|edge_compression|tinyvit|repvit) ;;
  *)
    echo "Usage: $0 {edge_official|edge_compression|tinyvit|repvit} {describe|run}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
case "${ACTION}" in
  describe|run) ;;
  *)
    echo "Usage: $0 {edge_official|edge_compression|tinyvit|repvit} {describe|run}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "[ERROR] Weekend lanes require exactly four GPUs: ${GPUS}" >&2
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
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/weekend_72h_v1/${LANE}}"
LOG_ROOT="${LOG_ROOT:-/user-volume/weekend_72h_logs/${LANE}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-weekend-72h-${LANE}-v1}"
EDGE_FOLLOWUP_MODE="${EDGE_FOLLOWUP_MODE:-full}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"
EDGE_K2_MIN_VAL_JF="${EDGE_K2_MIN_VAL_JF:-57.0}"
EDGE_K2_MIN_VAL_MIOU="${EDGE_K2_MIN_VAL_MIOU:-0.8355}"
EDGE_K2_MIN_VAL_AP="${EDGE_K2_MIN_VAL_AP:-0.7117}"
FAILED=()
COMPLETED_CANDIDATES=()

EDGE_OFFICIAL_VARIANTS=(
  W1_official_image_align_2ep
  W2a_official_logits_5ep
  W2b_official_memlogits_5ep
  W2c_official_full_5ep
  W3a_official_logits_t8_3ep
  W3b_official_memlogits_t8_3ep
  W3c_official_full_t8_3ep
)
EDGE_COMPRESSION_VARIANTS=(
  K1a_m0_task_5ep
  K1b_m0_logits_5ep
  K1c_m0_memlogits_5ep
  K1d_m0_full_5ep
  K2a_m0_task_t8_2ep
  K2b_m0_logits_t8_2ep
  K2c_m0_memlogits_t8_2ep
  K2d_m0_full_t8_2ep
)
case "${EDGE_FOLLOWUP_MODE}" in
  full) ;;
  core)
    EDGE_OFFICIAL_VARIANTS=(
      W1_official_image_align_2ep
      W2a_official_logits_5ep
      W2b_official_memlogits_5ep
      W3a_official_logits_t8_3ep
      W3b_official_memlogits_t8_3ep
    )
    EDGE_COMPRESSION_VARIANTS=(
      K1b_m0_logits_5ep
      K1c_m0_memlogits_5ep
      K2b_m0_logits_t8_2ep
      K2c_m0_memlogits_t8_2ep
    )
    ;;
  *)
    echo "[ERROR] EDGE_FOLLOWUP_MODE must be full or core" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

describe_lane() {
  echo "Weekend lane: ${LANE}"
  echo "Hardware: one independent 4xH100 node"
  echo "Selection: full SA-V val J&F; test is descriptive only"
  echo "Tracking: W&B ${WANDB_PROJECT}; mode ${WANDB_MODE}"
  echo "Retention: only last.pt and best.pt are physical task checkpoints"
  echo "Run root: ${RUN_ROOT}"
  echo "EdgeTAM follow-up mode: ${EDGE_FOLLOWUP_MODE}"
  case "${LANE}" in
    edge_official)
      if [[ "${EDGE_FOLLOWUP_MODE}" == "core" ]]; then
        echo "Budget: 24 T4-equivalent SA-V epochs plus five full val/test evaluations"
      else
        echo "Budget: 34 T4-equivalent SA-V epochs plus seven full val/test evaluations"
      fi
      echo "Question: can image alignment plus official temporal behavior targets repair the strict transplant?"
      printf '  %s\n' "${EDGE_OFFICIAL_VARIANTS[@]}"
      ;;
    edge_compression)
      if [[ "${EDGE_FOLLOWUP_MODE}" == "core" ]]; then
        echo "Budget: 18 T4-equivalent SA-V epochs plus four full val/test evaluations"
      else
        echo "Budget: 36 T4-equivalent SA-V epochs plus eight full val/test evaluations"
      fi
      echo "Question: which same-interface M0 behavior target makes 4-to-2 memory compression learnable?"
      if [[ "${EDGE_FOLLOWUP_MODE}" == "core" ]]; then
        echo "K2 gate: parent val J&F/mIoU/AP >= ${EDGE_K2_MIN_VAL_JF}/${EDGE_K2_MIN_VAL_MIOU}/${EDGE_K2_MIN_VAL_AP}"
      fi
      printf '  %s\n' "${EDGE_COMPRESSION_VARIANTS[@]}"
      ;;
    tinyvit)
      echo "Budget: 30 T4-equivalent SA-V epochs plus nine full val/test evaluations"
      echo "Question: for 5M/11M/21M, does frozen decoder-memory or low-LR joint continuation win before T8 refinement?"
      echo "  Per size: independent T4 3ep decoder-memory and joint forks; val-selected T8 2ep continuation"
      ;;
    repvit)
      echo "Budget: 24 T4-equivalent SA-V epochs plus six full val/test evaluations"
      echo "Question: is the remaining RepViT gap encoder-, decoder-, memory-, or coupling-limited?"
      echo "  Four independent 5ep forks; val-selected T8 3ep continuation; low-LR joint 3ep finish"
      ;;
  esac
  echo "Observed runtime estimate: 3-4 h per T4-equivalent epoch, excluding evaluation"
}

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

record_failure() {
  FAILED+=("$1:$2")
  echo "$1 status: $2"
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

run_edge_lane() {
  local -a variants=()
  local variant log status parent metrics
  if [[ "${LANE}" == "edge_official" ]]; then
    variants=("${EDGE_OFFICIAL_VARIANTS[@]}")
  else
    variants=("${EDGE_COMPRESSION_VARIANTS[@]}")
  fi
  mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
  for variant in "${variants[@]}"; do
    parent=""
    if [[ "${EDGE_FOLLOWUP_MODE}" == "core" ]]; then
      case "${variant}" in
        K2b_m0_logits_t8_2ep) parent=K1b_m0_logits_5ep ;;
        K2c_m0_memlogits_t8_2ep) parent=K1c_m0_memlogits_5ep ;;
      esac
    fi
    if [[ -n "${parent}" ]]; then
      metrics="${RUN_ROOT}/${parent}/main/sav_val_box_benchmark/metrics.csv"
      if ! python tools/experiments/check_task_metric_gate.py \
        --metrics "${metrics}" \
        --min-jf "${EDGE_K2_MIN_VAL_JF}" \
        --min-miou "${EDGE_K2_MIN_VAL_MIOU}" \
        --min-ap "${EDGE_K2_MIN_VAL_AP}"; then
        echo "[SKIP] ${variant}: ${parent} failed or lacks the K2 continuation gate."
        continue
      fi
    fi
    log="${LOG_ROOT}/${variant}_$(date +%Y%m%d_%H%M%S).log"
    echo
    echo "================================================================"
    echo "Starting: ${variant}"
    echo "Log: ${log}"
    echo "================================================================"
    env \
      WANDB_MODE="${WANDB_MODE}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      GPUS="${GPUS}" \
      FULL_EVAL_GPUS="${FULL_EVAL_GPUS}" \
      EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
      EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
      scripts/company/49_run_edgetam_memory_ablation.sh run "${variant}" \
        2>&1 | tee -a "${log}"
    status="${PIPESTATUS[0]}"
    if [[ "${status}" -ne 0 ]]; then
      record_failure "${variant}" "${status}"
    fi
  done
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
    scripts/company/49_run_edgetam_memory_ablation.sh summarize || \
      record_failure summary "$?"
}

run_task_stage() {
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
  local seed="${15}"
  local stage_dir="${RUN_ROOT}/${name}"
  local checkpoint_dir="${stage_dir}/checkpoints"
  local log
  log="${LOG_ROOT}/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo
  echo "================================================================"
  echo "Starting: ${name}"
  echo "Family/model: ${student_family} / ${model_name}"
  echo "Mode: ${trainable_mode}; T${frames}; epochs: ${epochs}; seed: ${seed}"
  echo "Log: ${log}"
  echo "================================================================"

  if [[ -f "${stage_dir}/.pipeline_complete" && \
        -f "${stage_dir}/sav_val_box_benchmark/metrics.csv" && \
        -f "${stage_dir}/sav_test_box_benchmark/metrics.csv" ]]; then
    echo "skip completed formal stage: ${name}"
    COMPLETED_CANDIDATES+=("${name}=${stage_dir}")
    return 0
  fi
  require_path "${student_checkpoint}" || return 1
  require_path "${source_stage1_checkpoint}" || return 1
  require_path "${previous_task_checkpoint}" || return 1

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
    TASK_EXPERIMENT_SUITE=weekend_72h_v1 \
    TASK_MASK_ABLATION_V2=1 \
    TASK_SEED="${seed}" \
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

select_candidate() {
  local prefix="$1"
  shift
  local -a args=()
  local spec
  for spec in "$@"; do
    args+=(--candidate "${spec}")
  done
  python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${RUN_ROOT}/${prefix}_selection.json" \
    --out-csv "${RUN_ROOT}/${prefix}_summary.csv" \
    --print-selected
}

selected_dir() {
  local selected="$1"
  shift
  local spec name
  for spec in "$@"; do
    name="${spec%%=*}"
    if [[ "${name}" == "${selected}" ]]; then
      echo "${spec#*=}"
      return 0
    fi
  done
  return 1
}

write_backbone_summary() {
  local -a args=()
  local spec
  if [[ "${#COMPLETED_CANDIDATES[@]}" -eq 0 ]]; then
    echo "[ERROR] No completed candidates for summary" >&2
    return 1
  fi
  for spec in "${COMPLETED_CANDIDATES[@]}"; do
    args+=(--candidate "${spec}")
  done
  python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${RUN_ROOT}/selection.json" \
    --out-csv "${RUN_ROOT}/summary.csv"
}

run_tinyvit_size() {
  local size="$1"
  local model_name adapter_mode student_checkpoint source_checkpoint base_task
  local encoder_lr encoder_lr_end head_lr head_lr_end
  local frozen_name joint_name refine_name selected previous
  local -a candidates=()
  case "${size}" in
    tv5)
      model_name="tiny_vit_5m_224.dist_in22k_ft_in1k"
      adapter_mode="residual_dwconv"
      student_checkpoint="${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors"
      source_checkpoint="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_extra_adapter_cos/tv5_adapter_sam21l_msehr/checkpoints/best.pt"
      base_task="${SAM2D_ROOT}/runs/tinyvit_capacity_freeze_v2/tv5_F2_joint_low_1ep/checkpoints/last.pt"
      encoder_lr="1.0e-7"
      encoder_lr_end="1.0e-8"
      head_lr="3.0e-7"
      head_lr_end="3.0e-8"
      ;;
    tv11)
      model_name="tiny_vit_11m_224.dist_in22k_ft_in1k"
      adapter_mode="projection"
      student_checkpoint="${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors"
      source_checkpoint="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_size_scaling/tv11_proj_sam21l_msehr_cos025/checkpoints/best.pt"
      base_task="${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/tv11/main/checkpoints/best.pt"
      encoder_lr="7.5e-8"
      encoder_lr_end="7.5e-9"
      head_lr="3.0e-7"
      head_lr_end="3.0e-8"
      ;;
    tv21)
      model_name="tiny_vit_21m_512.dist_in22k_ft_in1k"
      adapter_mode="projection"
      student_checkpoint="${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
      source_checkpoint="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt"
      base_task="${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt"
      encoder_lr="5.0e-8"
      encoder_lr_end="5.0e-9"
      head_lr="2.5e-7"
      head_lr_end="2.5e-8"
      ;;
  esac
  frozen_name="${size}_W1_decmem_t4_3ep"
  joint_name="${size}_W2_joint_t4_3ep"
  refine_name="${size}_W3_selected_t8_2ep"

  if run_task_stage \
    "${frozen_name}" tinyvit "${model_name}" "${adapter_mode}" \
    "${student_checkpoint}" "${source_checkpoint}" "${base_task}" \
    mask_decoder_memory 3 4 0 0 \
    "${head_lr}" "${head_lr_end}" 250107256; then
    candidates+=("${frozen_name}=${RUN_ROOT}/${frozen_name}")
  else
    record_failure "${frozen_name}" "$?"
  fi
  if run_task_stage \
    "${joint_name}" tinyvit "${model_name}" "${adapter_mode}" \
    "${student_checkpoint}" "${source_checkpoint}" "${base_task}" \
    image_encoder_mask_decoder_memory 3 4 \
    "${encoder_lr}" "${encoder_lr_end}" \
    "${head_lr}" "${head_lr_end}" 250107256; then
    candidates+=("${joint_name}=${RUN_ROOT}/${joint_name}")
  else
    record_failure "${joint_name}" "$?"
  fi
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    record_failure "${size}_selection" 1
    return 1
  fi
  selected="$(select_candidate "${size}_fork" "${candidates[@]}")" || return 1
  previous="$(selected_dir "${selected}" "${candidates[@]}")/checkpoints/last.pt" || return 1
  run_task_stage \
    "${refine_name}" tinyvit "${model_name}" "${adapter_mode}" \
    "${student_checkpoint}" "${source_checkpoint}" "${previous}" \
    mask_decoder_memory 2 8 0 0 \
    "${head_lr}" "${head_lr_end}" 250107257 || \
      record_failure "${refine_name}" "$?"
}

run_tinyvit_lane() {
  run_tinyvit_size tv5 || true
  run_tinyvit_size tv11 || true
  run_tinyvit_size tv21 || true
  write_backbone_summary || record_failure summary "$?"
}

run_repvit_lane() {
  local model_name="repvit_m0_9.dist_450e_in1k"
  local student_checkpoint="${SAM2D_ROOT}/checkpoints/repvit/repvit_m0_9.dist_450e_in1k.safetensors"
  local source_root="${SAM2D_ROOT}/runs/repvit_stage1_v1/repvit_m09_proj_sam21l_msehr_cos025_l1010"
  local source_checkpoint="${source_root}/checkpoints/best.pt"
  local base_task="${SAM2D_ROOT}/runs/repvit_task_finetune_v2/repvit_P3_decmem_t8_refine_1ep/checkpoints/last.pt"
  local selected previous
  local -a candidates=()
  local -a stage_specs=(
    "repvit_W1_encoder_t2_5ep|image_encoder_only|5|2|5.0e-7|5.0e-8|3.0e-7|3.0e-8|250107256"
    "repvit_W2_decoder_t2_5ep|mask_decoder_only|5|2|0|0|5.0e-7|5.0e-8|250107256"
    "repvit_W3_decmem_t4_5ep|mask_decoder_memory|5|4|0|0|5.0e-7|5.0e-8|250107256"
    "repvit_W4_joint_t4_5ep|image_encoder_mask_decoder_memory|5|4|1.5e-7|1.5e-8|5.0e-7|5.0e-8|250107256"
  )
  local spec name mode epochs frames enc_lr enc_end other_lr other_end seed

  if [[ ! -f "${source_checkpoint}" ]]; then
    source_checkpoint="${source_root}/checkpoints/last.pt"
  fi
  for spec in "${stage_specs[@]}"; do
    IFS='|' read -r name mode epochs frames enc_lr enc_end other_lr other_end seed <<< "${spec}"
    if run_task_stage \
      "${name}" repvit "${model_name}" projection \
      "${student_checkpoint}" "${source_checkpoint}" "${base_task}" \
      "${mode}" "${epochs}" "${frames}" \
      "${enc_lr}" "${enc_end}" "${other_lr}" "${other_end}" "${seed}"; then
      candidates+=("${name}=${RUN_ROOT}/${name}")
    else
      record_failure "${name}" "$?"
    fi
  done
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    record_failure repvit_selection 1
    return 1
  fi
  selected="$(select_candidate repvit_fork "${candidates[@]}")" || return 1
  previous="$(selected_dir "${selected}" "${candidates[@]}")/checkpoints/last.pt" || return 1
  run_task_stage \
    repvit_W5_selected_t8_3ep repvit "${model_name}" projection \
    "${student_checkpoint}" "${source_checkpoint}" "${previous}" \
    mask_decoder_memory 3 8 0 0 2.5e-7 2.5e-8 250107257 || \
      record_failure repvit_W5_selected_t8_3ep "$?"
  previous="${RUN_ROOT}/repvit_W5_selected_t8_3ep/checkpoints/last.pt"
  if [[ -f "${previous}" ]]; then
    run_task_stage \
      repvit_W6_joint_low_t4_3ep repvit "${model_name}" projection \
      "${student_checkpoint}" "${source_checkpoint}" "${previous}" \
      image_encoder_mask_decoder_memory 3 4 \
      7.5e-8 7.5e-9 2.5e-7 2.5e-8 250107258 || \
        record_failure repvit_W6_joint_low_t4_3ep "$?"
  else
    record_failure repvit_W6_joint_low_t4_3ep 1
  fi
  write_backbone_summary || record_failure summary "$?"
}

if [[ "${ACTION}" == "describe" ]]; then
  describe_lane
  if [[ "${LANE}" == edge_* ]]; then
    variants=("${EDGE_OFFICIAL_VARIANTS[@]}")
    [[ "${LANE}" == "edge_compression" ]] && \
      variants=("${EDGE_COMPRESSION_VARIANTS[@]}")
    for variant in "${variants[@]}"; do
      EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
        scripts/company/49_run_edgetam_memory_ablation.sh describe "${variant}" || \
          return 1 2>/dev/null || exit 1
    done
  fi
  return 0 2>/dev/null || exit 0
fi

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
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
describe_lane

case "${LANE}" in
  edge_official|edge_compression) run_edge_lane ;;
  tinyvit) run_tinyvit_lane ;;
  repvit) run_repvit_lane ;;
esac

echo
echo "Weekend lane: ${LANE}"
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
