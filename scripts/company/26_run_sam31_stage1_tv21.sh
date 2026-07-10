#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAM3_ROOT="${SAM3_ROOT:-${DATA_ROOT}/sam3}"
SAM3_UPSTREAM="${SAM3_UPSTREAM:-/user-volume/repo/facebookresearch-sam3}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps.parquet}"
SAM31_CKPT="${SAM31_CKPT:-${SAM3_ROOT}/checkpoints/sam3.1/sam3.1_multiplex.pt}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
RUN_DIR="${RUN_DIR:-${SAM2D_ROOT}/runs/sam31_stage1/tv21m_adapter_mse_cos025_5ep_v1}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-300000}"
MAX_VAL_ITEMS="${MAX_VAL_ITEMS:-1240}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PROJECTION_WARMUP_STEPS="${PROJECTION_WARMUP_STEPS:-2000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
LAMBDA_MSE="${LAMBDA_MSE:-1.0}"
LAMBDA_COS="${LAMBDA_COS:-0.25}"
ADAPTER_MODE="${ADAPTER_MODE:-residual_dwconv}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TEACHER_AMP_DTYPE="${TEACHER_AMP_DTYPE:-bf16}"
LOG_EVERY="${LOG_EVERY:-30}"
PRINT_EVERY="${PRINT_EVERY:-300}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
RESUME="${RESUME:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-sam31-distill-stage1}"
WANDB_NAME="${WANDB_NAME:-tv21m-adapter-sam31-mse-cos025-5ep-v1}"
NO_WANDB="${NO_WANDB:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/26_run_sam31_stage1_tv21.sh setup
  scripts/company/26_run_sam31_stage1_tv21.sh inspect
  scripts/company/26_run_sam31_stage1_tv21.sh smoke
  scripts/company/26_run_sam31_stage1_tv21.sh train

The formal default uses a deterministic 300k-frame SA-V subset for five epochs,
an online SAM3.1 vision-trunk teacher, TinyViT-21M ImageNet initialization,
projection plus a BN-free residual adapter, and MSE + 0.25 cosine loss.
Only checkpoints/best.pt and checkpoints/last.pt are retained.
EOF
}

nproc_from_gpus() {
  python - "${GPUS}" <<'PY'
import sys
print(len([item for item in sys.argv[1].split(",") if item.strip()]))
PY
}

check_paths() {
  local missing=0
  for path in "${MANIFEST}" "${SAM31_CKPT}" "${TINYVIT_CKPT}"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing required file: ${path}" >&2
      missing=1
    fi
  done
  if [[ ! -f "${SAM3_UPSTREAM}/sam3/model_builder.py" ]]; then
    echo "missing official SAM3 checkout: ${SAM3_UPSTREAM}" >&2
    missing=1
  fi
  [[ "${missing}" -eq 0 ]]
}

setup_env() {
  if [[ ! -d "${SAM3_UPSTREAM}/.git" ]]; then
    mkdir -p "$(dirname "${SAM3_UPSTREAM}")"
    git clone https://github.com/facebookresearch/sam3.git "${SAM3_UPSTREAM}"
  fi
  python -m pip install --user -r requirements-sam31-stage1.txt
  PYTHONPATH="${SAM3_UPSTREAM}:${PYTHONPATH:-}" python - <<'PY'
import torch
from sam3.model_builder import _create_vit_backbone
print("torch:", torch.__version__)
print("sam3 import: PASS")
print("builder:", _create_vit_backbone.__name__)
PY
}

inspect() {
  check_paths
  PYTHONPATH="${SAM3_UPSTREAM}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" \
    python tools/train/smoke_sam31_stage1.py \
      --teacher-checkpoint "${SAM31_CKPT}" \
      --tinyvit-checkpoint "${TINYVIT_CKPT}" \
      --adapter-mode "${ADAPTER_MODE}"
}

compute_schedule() {
  local nproc
  nproc="$(nproc_from_gpus)"
  python - "${MANIFEST}" "${MAX_TRAIN_ITEMS}" "${EPOCHS}" "${BATCH_SIZE}" "${nproc}" <<'PY'
import math
import sys
import pandas as pd

manifest, max_items, epochs, batch_size, world_size = sys.argv[1:]
df = pd.read_parquet(manifest) if manifest.endswith(".parquet") else pd.read_csv(manifest)
train_items = int((df["split"] == "train").sum())
if int(max_items) > 0:
    train_items = min(train_items, int(max_items))
steps_per_epoch = math.ceil(train_items / (int(batch_size) * int(world_size)))
print(math.ceil(steps_per_epoch * float(epochs)), steps_per_epoch)
PY
}

train() {
  local nproc computed_steps steps_per_epoch resume_args=() wandb_args=()
  check_paths
  nproc="$(nproc_from_gpus)"
  read -r computed_steps steps_per_epoch < <(compute_schedule)
  MAX_STEPS="${MAX_STEPS:-${computed_steps}}"
  mkdir -p "${RUN_DIR}"
  if [[ "${RESUME}" == "auto" && -f "${RUN_DIR}/checkpoints/last.pt" ]]; then
    resume_args=(--resume "${RUN_DIR}/checkpoints/last.pt")
  elif [[ "${RESUME}" != "auto" && -n "${RESUME}" ]]; then
    resume_args=(--resume "${RESUME}")
  fi
  if [[ "${NO_WANDB}" == "1" ]]; then
    wandb_args=(--no-wandb)
  fi
  PYTHONPATH="${SAM3_UPSTREAM}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun --standalone --nproc-per-node "${nproc}" \
      tools/train/train_sam31_stage1_online_teacher.py \
      --manifest "${MANIFEST}" \
      --teacher-checkpoint "${SAM31_CKPT}" \
      --tinyvit-checkpoint "${TINYVIT_CKPT}" \
      --out-dir "${RUN_DIR}" \
      --adapter-mode "${ADAPTER_MODE}" \
      --train-split train \
      --val-split val_sav \
      --max-train-items "${MAX_TRAIN_ITEMS}" \
      --max-val-items "${MAX_VAL_ITEMS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-steps "${MAX_STEPS}" \
      --lr "${LR}" \
      --min-lr "${MIN_LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --projection-warmup-steps "${PROJECTION_WARMUP_STEPS}" \
      --lr-warmup-steps "${LR_WARMUP_STEPS}" \
      --lambda-mse "${LAMBDA_MSE}" \
      --lambda-cos "${LAMBDA_COS}" \
      --amp-dtype "${AMP_DTYPE}" \
      --teacher-amp-dtype "${TEACHER_AMP_DTYPE}" \
      --log-every "${LOG_EVERY}" \
      --print-every "${PRINT_EVERY}" \
      --eval-every "${steps_per_epoch}" \
      --save-every "${steps_per_epoch}" \
      --val-max-batches "${VAL_MAX_BATCHES}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${WANDB_NAME}" \
      "${resume_args[@]}" \
      "${wandb_args[@]}"
}

smoke() {
  RUN_DIR="${RUN_DIR}_smoke" \
  GPUS="${GPUS%%,*}" \
  MAX_TRAIN_ITEMS=64 \
  MAX_VAL_ITEMS=32 \
  BATCH_SIZE=1 \
  MAX_STEPS=20 \
  PROJECTION_WARMUP_STEPS=5 \
  LR_WARMUP_STEPS=5 \
  VAL_MAX_BATCHES=2 \
  PRINT_EVERY=1 \
  LOG_EVERY=1 \
  WANDB_NAME="${WANDB_NAME}-smoke" \
  RESUME="" \
    "${BASH_SOURCE[0]}" train
}

case "${1:-}" in
  setup) setup_env ;;
  inspect) inspect ;;
  smoke) smoke ;;
  train) train ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
