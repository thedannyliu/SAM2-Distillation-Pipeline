#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

MODE="${1:-all}"
GPUS="${GPUS:-0,1,2,3}"
DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/repvit_stage1_v1}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet}"
SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
REPVIT_ROOT="${REPVIT_ROOT:-${CHECKPOINT_ROOT}/repvit}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${CHECKPOINT_ROOT}/sam2.1/sam2.1_hiera_large.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-distill-repvit-stage1-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRINT_EVERY="${PRINT_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-30}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
SKIP_DONE="${SKIP_DONE:-1}"

M09_NAME="repvit_m0_9.dist_450e_in1k"
M09_CKPT="${REPVIT_ROOT}/${M09_NAME}.safetensors"
M23_NAME="repvit_m2_3.dist_450e_in1k"
M23_CKPT="${REPVIT_ROOT}/${M23_NAME}.safetensors"

usage() {
  echo "Usage: scripts/company/38_run_repvit_sam21l_stage1.sh smoke|train|eval|all"
}

check_inputs() {
  local path missing=0
  for path in \
    "${MANIFEST}" \
    "${SAM2_CHECKPOINT}" \
    "${M09_CKPT}" \
    "${M23_CKPT}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt"; do
    if [[ ! -e "${path}" ]]; then
      echo "missing required input: ${path}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

smoke_model() {
  local model_name="$1" checkpoint="$2"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" \
    python tools/train/smoke_repvit_stage1.py \
      --model-name "${model_name}" \
      --checkpoint "${checkpoint}" \
      --device cuda
}

step_plan() {
  local batch_size="$1"
  python - "${MANIFEST}" "${batch_size}" "${GPUS}" <<'PY'
import math
import sys
import pandas as pd

manifest, batch_size, gpus = sys.argv[1:]
train_rows = int((pd.read_parquet(manifest, columns=["split"])["split"] == "train").sum())
world_size = len([part for part in gpus.split(",") if part.strip()])
steps_per_epoch = math.ceil(train_rows / (int(batch_size) * world_size))
print(f"{steps_per_epoch} {steps_per_epoch * 5}")
PY
}

train_model() {
  local experiment="$1" model_name="$2" checkpoint="$3" batch_size="$4" lr="$5"
  local run_dir="${RUN_ROOT}/${experiment}" steps_per_epoch max_steps plan
  plan="$(step_plan "${batch_size}")" || return 1
  read -r steps_per_epoch max_steps <<< "${plan}"
  echo "===== train ${experiment}: steps/epoch=${steps_per_epoch}, max_steps=${max_steps} ====="
  DATA_ROOT="${DATA_ROOT}" \
  SAM2D_ROOT="${SAM2D_ROOT}" \
  SAM2_UPSTREAM="${SAM2_UPSTREAM:-/user-volume/repo/facebookresearch-sam2}" \
  MANIFEST="${MANIFEST}" \
  RUN_DIR="${run_dir}" \
  GPUS="${GPUS}" \
  SAM2_CONFIG="${SAM2_CONFIG}" \
  SAM2_CKPT="${SAM2_CHECKPOINT}" \
  STUDENT_FAMILY=repvit \
  STUDENT_CKPT="${checkpoint}" \
  STUDENT_MODEL_NAME="${model_name}" \
  ADAPTER_MODE=projection \
  BATCH_SIZE="${batch_size}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  MAX_STEPS="${max_steps}" \
  LR="${lr}" \
  WEIGHT_DECAY=0.05 \
  PROJECTION_WARMUP_STEPS=2000 \
  LR_WARMUP_STEPS=2000 \
  MAX_GRAD_NORM=1.0 \
  TRAIN_SPLIT=train \
  VAL_SPLIT=val_sav \
  MAX_TRAIN_ITEMS="" \
  MAX_VAL_ITEMS=0 \
  VAL_MAX_BATCHES=0 \
  LAMBDA_MSE=1.0 \
  LAMBDA_HR=1.0 \
  LAMBDA_COS=0.25 \
  LAMBDA_L1=0.10 \
  AMP_DTYPE=bf16 \
  TEACHER_AMP_DTYPE=bf16 \
  TRAIN_SEED=250107256 \
  LOG_EVERY="${LOG_EVERY}" \
  PRINT_EVERY="${PRINT_EVERY}" \
  EVAL_EVERY="${steps_per_epoch}" \
  SAVE_EVERY="${steps_per_epoch}" \
  SAVE_STEP_CHECKPOINTS=0 \
  RESUME=auto \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_NAME="${experiment}" \
  scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh train || return 1
  python - "${run_dir}" "${max_steps}" <<'PY'
import sys
from pathlib import Path
import torch

run_dir = Path(sys.argv[1])
target = int(sys.argv[2])
for name in ("best.pt", "last.pt"):
    path = run_dir / "checkpoints" / name
    if not path.is_file():
        raise SystemExit(f"missing completed checkpoint: {path}")
last = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
step = int(last.get("step", 0))
if step < target:
    raise SystemExit(f"training incomplete: step={step}, target={target}")
print(f"training complete: {run_dir.name} step={step} target={target}")
PY
}

eval_split() {
  local experiment="$1" model_name="$2" checkpoint="$3" split="$4"
  local run_dir="${RUN_ROOT}/${experiment}"
  echo "===== full ${split} ${experiment} ====="
  MODEL_FAMILY=sam2 \
  STUDENT_FAMILY=repvit \
  STUDENT_CHECKPOINT="${checkpoint}" \
  STUDENT_MODEL_NAME="${model_name}" \
  EXPERIMENT="${experiment}" \
  RUN_DIR="${run_dir}" \
  SAV_ROOT="${SAV_ROOT}" \
  SAV_SPLIT="${split}" \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  MAX_VIDEOS=0 \
  MAX_IMAGE_OBJECTS=0 \
  SKIP_DONE="${SKIP_DONE}" \
  scripts/company/25_benchmark_stage1_sav_test.sh
}

eval_model() {
  local experiment="$1" model_name="$2" checkpoint="$3"
  local best="${RUN_ROOT}/${experiment}/checkpoints/best.pt"
  if [[ ! -s "${best}" ]]; then
    echo "missing best checkpoint for evaluation: ${best}" >&2
    return 1
  fi
  eval_split "${experiment}" "${model_name}" "${checkpoint}" sav_val || return 1
  eval_split "${experiment}" "${model_name}" "${checkpoint}" sav_test
}

run_train_matrix() {
  train_model repvit_m09_proj_sam21l_msehr_cos025_l1010 "${M09_NAME}" "${M09_CKPT}" 8 1e-4 || return 1
  train_model repvit_m23_proj_sam21l_msehr_cos025_l1010 "${M23_NAME}" "${M23_CKPT}" 4 5e-5
}

run_eval_matrix() {
  eval_model repvit_m09_proj_sam21l_msehr_cos025_l1010 "${M09_NAME}" "${M09_CKPT}" || return 1
  eval_model repvit_m23_proj_sam21l_msehr_cos025_l1010 "${M23_NAME}" "${M23_CKPT}"
}

check_inputs || {
  echo "RepViT Stage 1 input check failed" >&2
  exit 1
}
mkdir -p "${RUN_ROOT}"

case "${MODE}" in
  smoke)
    smoke_model "${M09_NAME}" "${M09_CKPT}" || exit 1
    smoke_model "${M23_NAME}" "${M23_CKPT}"
    ;;
  train)
    run_train_matrix
    ;;
  eval)
    run_eval_matrix
    ;;
  all)
    smoke_model "${M09_NAME}" "${M09_CKPT}" || exit 1
    smoke_model "${M23_NAME}" "${M23_CKPT}" || exit 1
    train_model repvit_m09_proj_sam21l_msehr_cos025_l1010 "${M09_NAME}" "${M09_CKPT}" 8 1e-4 || exit 1
    eval_model repvit_m09_proj_sam21l_msehr_cos025_l1010 "${M09_NAME}" "${M09_CKPT}" || exit 1
    train_model repvit_m23_proj_sam21l_msehr_cos025_l1010 "${M23_NAME}" "${M23_CKPT}" 4 5e-5 || exit 1
    eval_model repvit_m23_proj_sam21l_msehr_cos025_l1010 "${M23_NAME}" "${M23_CKPT}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
