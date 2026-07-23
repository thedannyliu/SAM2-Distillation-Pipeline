#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

SIZE="${1:-}"
ACTION="${2:-all}"
case "${SIZE}" in
  tv5|tv11|tv21) ;;
  *)
    echo "Usage: $0 {tv5|tv11|tv21} {describe|all|summary}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
case "${ACTION}" in
  describe|all|summary) ;;
  *)
    echo "Usage: $0 {tv5|tv11|tv21} {describe|all|summary}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
if [[ "${NPROC}" -ne 4 ]]; then
  echo "[ERROR] ${SIZE} max-J&F lane requires exactly four GPUs: ${GPUS}" >&2
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
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/${SIZE}}"
MAIN_DIR="${RUN_ROOT}/main"
WANDB_PROJECT="${WANDB_PROJECT:-tinyvit-max-jf-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"

case "${SIZE}" in
  tv5)
    TINYVIT_MODEL_NAME="tiny_vit_5m_224.dist_in22k_ft_in1k"
    TINYVIT_ADAPTER_MODE="residual_dwconv"
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors}"
    SOURCE_RUN="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_extra_adapter_cos/tv5_adapter_sam21l_msehr"
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SOURCE_RUN}/checkpoints/best.pt}"
    BASE_CANDIDATE_NAME="tv5_stage1_best"
    BASE_CANDIDATE_DIR="${SOURCE_RUN}"
    STAGE1_NAME="tv5_S1_encoder_t2_2ep"
    STAGE1_MODE="image_encoder_only"
    STAGE1_EPOCHS=2
    STAGE1_FRAMES=2
    STAGE1_ENCODER_LR=1.0e-6
    STAGE1_ENCODER_LR_END=1.0e-7
    STAGE1_HEAD_LR=1.0e-6
    STAGE1_HEAD_LR_END=1.0e-7
    STAGE1_PREVIOUS_CHECKPOINT=""
    STAGE2_ENCODER_LR=3.0e-7
    STAGE2_ENCODER_LR_END=3.0e-8
    JOINT_HEAD_LR=1.0e-6
    JOINT_HEAD_LR_END=1.0e-7
    REFINE_HEAD_LR=5.0e-7
    REFINE_HEAD_LR_END=5.0e-8
    ;;
  tv11)
    TINYVIT_MODEL_NAME="tiny_vit_11m_224.dist_in22k_ft_in1k"
    TINYVIT_ADAPTER_MODE="projection"
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors}"
    SOURCE_RUN="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_size_scaling/tv11_proj_sam21l_msehr_cos025"
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SOURCE_RUN}/checkpoints/best.pt}"
    BASE_CANDIDATE_NAME="tv11_stage1_best"
    BASE_CANDIDATE_DIR="${SOURCE_RUN}"
    STAGE1_NAME="tv11_S1_encoder_t2_2ep"
    STAGE1_MODE="image_encoder_only"
    STAGE1_EPOCHS=2
    STAGE1_FRAMES=2
    STAGE1_ENCODER_LR=8.0e-7
    STAGE1_ENCODER_LR_END=8.0e-8
    STAGE1_HEAD_LR=1.0e-6
    STAGE1_HEAD_LR_END=1.0e-7
    STAGE1_PREVIOUS_CHECKPOINT=""
    STAGE2_ENCODER_LR=2.5e-7
    STAGE2_ENCODER_LR_END=2.5e-8
    JOINT_HEAD_LR=1.0e-6
    JOINT_HEAD_LR_END=1.0e-7
    REFINE_HEAD_LR=5.0e-7
    REFINE_HEAD_LR_END=5.0e-8
    ;;
  tv21)
    TINYVIT_MODEL_NAME="tiny_vit_21m_512.dist_in22k_ft_in1k"
    TINYVIT_ADAPTER_MODE="projection"
    TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
    SOURCE_RUN="${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025"
    SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SOURCE_RUN}/checkpoints/best.pt}"
    BASE_CANDIDATE_NAME="tv21_A02_best"
    BASE_CANDIDATE_DIR="${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/A02_e2e_t4_official_prompt/main"
    STAGE1_NAME="tv21_S1_e2e_t4_continue_1ep"
    STAGE1_MODE="image_encoder_mask_decoder_memory"
    STAGE1_EPOCHS=1
    STAGE1_FRAMES=4
    STAGE1_ENCODER_LR=1.5e-7
    STAGE1_ENCODER_LR_END=1.5e-8
    STAGE1_HEAD_LR=5.0e-7
    STAGE1_HEAD_LR_END=5.0e-8
    STAGE1_PREVIOUS_CHECKPOINT="${BASE_CANDIDATE_DIR}/checkpoints/checkpoint.pt"
    STAGE2_ENCODER_LR=7.5e-8
    STAGE2_ENCODER_LR_END=7.5e-9
    JOINT_HEAD_LR=5.0e-7
    JOINT_HEAD_LR_END=5.0e-8
    REFINE_HEAD_LR=2.5e-7
    REFINE_HEAD_LR_END=2.5e-8
    ;;
esac

STAGE2_NAME="${SIZE}_S2_e2e_t4_low_1ep"
STAGE2_MODE="image_encoder_mask_decoder_memory"
STAGE2_EPOCHS=1
STAGE2_FRAMES=4
STAGE2_HEAD_LR="${JOINT_HEAD_LR}"
STAGE2_HEAD_LR_END="${JOINT_HEAD_LR_END}"

STAGE3_NAME="${SIZE}_S3_decmem_t4_refine_1ep"
STAGE3_MODE="mask_decoder_memory"
STAGE3_EPOCHS=1
STAGE3_FRAMES=4
STAGE3_ENCODER_LR=0
STAGE3_ENCODER_LR_END=0
STAGE3_HEAD_LR="${REFINE_HEAD_LR}"
STAGE3_HEAD_LR_END="${REFINE_HEAD_LR_END}"

CANDIDATE_NAMES=(
  "${BASE_CANDIDATE_NAME}"
  "${STAGE1_NAME}"
  "${STAGE2_NAME}"
  "${STAGE3_NAME}"
)
CANDIDATE_DIRS=(
  "${BASE_CANDIDATE_DIR}"
  "${RUN_ROOT}/${STAGE1_NAME}"
  "${RUN_ROOT}/${STAGE2_NAME}"
  "${RUN_ROOT}/${STAGE3_NAME}"
)

export GPUS FULL_EVAL_GPUS SAM2D_ROOT SAV_ROOT MANIFEST
export SAM2_TRAINING_ROOT SAM2_CHECKPOINT CONFIG RUN_ROOT
export TINYVIT_MODEL_NAME TINYVIT_ADAPTER_MODE TINYVIT_CHECKPOINT
export SOURCE_STAGE1_CHECKPOINT WANDB_PROJECT WANDB_MODE
export TASK_NUM_WORKERS PRINT_EVERY LOG_EVERY
export STAGE1_NAME STAGE1_MODE STAGE1_EPOCHS STAGE1_FRAMES
export STAGE1_ENCODER_LR STAGE1_ENCODER_LR_END
export STAGE1_HEAD_LR STAGE1_HEAD_LR_END STAGE1_PREVIOUS_CHECKPOINT
export STAGE2_NAME STAGE2_MODE STAGE2_EPOCHS STAGE2_FRAMES
export STAGE2_ENCODER_LR STAGE2_ENCODER_LR_END
export STAGE2_HEAD_LR STAGE2_HEAD_LR_END
export STAGE3_NAME STAGE3_MODE STAGE3_EPOCHS STAGE3_FRAMES
export STAGE3_ENCODER_LR STAGE3_ENCODER_LR_END
export STAGE3_HEAD_LR STAGE3_HEAD_LR_END
export TASK_MASK_ABLATION_V2=1
export TASK_TRAIN_BATCH_SIZE=1
export TASK_MAX_NUM_OBJECTS=2
export TASK_FREEZE_BATCHNORM=true
export TASK_LR_WARMUP_FRACTION=0
export TASK_PROB_USE_POINT=0.5
export TASK_PROB_USE_BOX=0.5
export TASK_PROB_SAMPLE_GT=0.1
export TASK_NUM_FRAMES_TO_CORRECT=2
export TASK_RANDOM_CORRECTION_FRAMES=true
export TASK_NUM_INIT_COND_FRAMES=1
export TASK_RANDOM_INIT_COND_FRAMES=false
export TASK_NUM_CORRECTION_POINTS=7
export TASK_EXPORT_STAGE_CHECKPOINT=1
export TASK_EVAL_SPLITS=sav_val,sav_test
export SKIP_DONE=1

describe() {
  echo "TinyViT max-J&F lane: ${SIZE}"
  echo "Model: ${TINYVIT_MODEL_NAME}; adapter: ${TINYVIT_ADAPTER_MODE}"
  echo "Stage 1 source: ${SOURCE_STAGE1_CHECKPOINT}"
  echo "Selection baseline: ${BASE_CANDIDATE_NAME} | ${BASE_CANDIDATE_DIR}"
  echo "S1: ${STAGE1_NAME} | ${STAGE1_MODE} | T${STAGE1_FRAMES} | ${STAGE1_EPOCHS} epoch(s)"
  echo "S2: ${STAGE2_NAME} | ${STAGE2_MODE} | T${STAGE2_FRAMES} | ${STAGE2_EPOCHS} epoch"
  echo "S3: ${STAGE3_NAME} | ${STAGE3_MODE} | T${STAGE3_FRAMES} | ${STAGE3_EPOCHS} epoch"
  echo "Selection: maximum full SA-V val J&F; test is never used for ranking"
  echo "W&B: ${WANDB_PROJECT}; run root: ${RUN_ROOT}"
}

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

validate_inputs() {
  local path
  for path in \
    "${MANIFEST}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt" \
    "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
    "${SAM2_CHECKPOINT}" \
    "${TINYVIT_CHECKPOINT}" \
    "${SOURCE_STAGE1_CHECKPOINT}" \
    "${BASE_CANDIDATE_DIR}/sav_val_box_benchmark/metrics.csv" \
    "${BASE_CANDIDATE_DIR}/sav_test_box_benchmark/metrics.csv" \
    "${CONFIG}"; do
    require_path "${path}" || return 1
  done
  if [[ -n "${STAGE1_PREVIOUS_CHECKPOINT}" ]]; then
    require_path "${STAGE1_PREVIOUS_CHECKPOINT}" || return 1
  fi
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

selection_args() {
  local index
  for index in "${!CANDIDATE_NAMES[@]}"; do
    printf '%s\n' \
      --candidate \
      "${CANDIDATE_NAMES[index]}=${CANDIDATE_DIRS[index]}"
  done
}

write_summary() {
  local -a args=()
  while IFS= read -r value; do
    args+=("${value}")
  done < <(selection_args)
  python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${RUN_ROOT}/selection.json" \
    --out-csv "${RUN_ROOT}/summary.csv"
}

finalize_retention() {
  local selected_name selected_dir selected_checkpoint last_checkpoint
  local index
  local -a args=()
  while IFS= read -r value; do
    args+=("${value}")
  done < <(selection_args)
  selected_name="$(python tools/train/select_task_checkpoint_by_val.py \
    "${args[@]}" \
    --out-json "${RUN_ROOT}/selection.json" \
    --out-csv "${RUN_ROOT}/summary.csv" \
    --print-selected)" || return 1
  selected_dir=""
  for index in "${!CANDIDATE_NAMES[@]}"; do
    if [[ "${CANDIDATE_NAMES[index]}" == "${selected_name}" ]]; then
      selected_dir="${CANDIDATE_DIRS[index]}"
      break
    fi
  done
  [[ -n "${selected_dir}" ]] || return 1

  selected_checkpoint="${selected_dir}/checkpoints/stage.pt"
  [[ -f "${selected_checkpoint}" ]] || \
    selected_checkpoint="${selected_dir}/checkpoints/best.pt"
  [[ -f "${selected_checkpoint}" ]] || \
    selected_checkpoint="${selected_dir}/checkpoints/checkpoint.pt"
  last_checkpoint="${RUN_ROOT}/${STAGE3_NAME}/checkpoints/checkpoint.pt"
  require_path "${selected_checkpoint}" || return 1
  require_path "${last_checkpoint}" || return 1

  mkdir -p "${MAIN_DIR}/checkpoints" "${MAIN_DIR}/wandb"
  cp -p --reflink=auto "${last_checkpoint}" "${MAIN_DIR}/checkpoints/last.pt" || return 1
  if [[ "${selected_checkpoint}" -ef "${last_checkpoint}" ]]; then
    ln -sfn last.pt "${MAIN_DIR}/checkpoints/best.pt"
  else
    cp -p --reflink=auto "${selected_checkpoint}" "${MAIN_DIR}/checkpoints/best.pt" || return 1
  fi
  ln -sfn best.pt "${MAIN_DIR}/checkpoints/checkpoint.pt"
  ln -sfn best.pt "${MAIN_DIR}/checkpoints/stage.pt"

  rm -rf \
    "${MAIN_DIR}/sav_val_box_benchmark" \
    "${MAIN_DIR}/sav_test_box_benchmark"
  cp -a \
    "${selected_dir}/sav_val_box_benchmark" \
    "${MAIN_DIR}/sav_val_box_benchmark" || return 1
  cp -a \
    "${selected_dir}/sav_test_box_benchmark" \
    "${MAIN_DIR}/sav_test_box_benchmark" || return 1
  for path in \
    training_status.json \
    training_model_summary.json \
    resolved_config.yaml; do
    if [[ -f "${selected_dir}/${path}" ]]; then
      cp "${selected_dir}/${path}" "${MAIN_DIR}/${path}" || return 1
    fi
  done
  if [[ -f "${selected_dir}/wandb/wandb_run.json" ]]; then
    cp \
      "${selected_dir}/wandb/wandb_run.json" \
      "${MAIN_DIR}/wandb/wandb_run.json" || return 1
  fi
  python - \
    "${RUN_ROOT}/selection.json" \
    "${MAIN_DIR}/training_status.json" \
    "${selected_name}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["selected_candidate"] = sys.argv[3]
payload["status"] = "complete"
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "status": "complete",
            "selected_candidate": sys.argv[3],
            "selection_metric": "full_sav_val_J&F",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  touch "${RUN_ROOT}/.pipeline_complete"
  echo "Selected by full val J&F: ${selected_name}"
  echo "Best checkpoint: ${MAIN_DIR}/checkpoints/best.pt"
  echo "Last checkpoint: ${MAIN_DIR}/checkpoints/last.pt"
}

if [[ "${ACTION}" == "describe" ]]; then
  describe
  return 0 2>/dev/null || exit 0
fi

if [[ "${ACTION}" == "summary" ]]; then
  write_summary
  STATUS="$?"
  echo "Summary status: ${STATUS}"
  echo "Summary CSV: ${RUN_ROOT}/summary.csv"
  return "${STATUS}" 2>/dev/null || exit "${STATUS}"
fi

describe
if [[ -f "${RUN_ROOT}/.pipeline_complete" ]]; then
  echo "skip completed max-J&F pipeline: ${SIZE}"
  cat "${RUN_ROOT}/summary.csv"
  return 0 2>/dev/null || exit 0
fi

validate_inputs || return 1 2>/dev/null || exit 1
wandb_preflight || return 1 2>/dev/null || exit 1
mkdir -p "${RUN_ROOT}"

scripts/company/39_run_sam2_task_finetune_3stage.sh audit || \
  return 1 2>/dev/null || exit 1
scripts/company/39_run_sam2_task_finetune_3stage.sh stage1 || \
  return 1 2>/dev/null || exit 1
scripts/company/39_run_sam2_task_finetune_3stage.sh stage2 || \
  return 1 2>/dev/null || exit 1
scripts/company/39_run_sam2_task_finetune_3stage.sh stage3 || \
  return 1 2>/dev/null || exit 1
finalize_retention || return 1 2>/dev/null || exit 1
write_summary || return 1 2>/dev/null || exit 1

echo "TinyViT max-J&F status: 0"
echo "Size: ${SIZE}"
echo "Summary CSV: ${RUN_ROOT}/summary.csv"
cat "${RUN_ROOT}/summary.csv"
return 0 2>/dev/null || exit 0
