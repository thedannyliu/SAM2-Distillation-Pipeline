#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
DEFAULT_MANIFEST="${SAM2D_ROOT}/manifests/stage1_vbal16_6fps.parquet"
MANIFEST="${MANIFEST:-${DEFAULT_MANIFEST}}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-distill-sav-stage1-ablation-v2}"

EXPERIMENT="${EXPERIMENT:-}"
GPUS="${GPUS:-0,1,2,3}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PROJECTION_WARMUP_STEPS="${PROJECTION_WARMUP_STEPS:-2000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_VAL_ITEMS="${MAX_VAL_ITEMS:-4000}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-50}"
LOG_EVERY="${LOG_EVERY:-10}"
RESUME="${RESUME:-auto}"
SAVE_STEP_CHECKPOINTS="${SAVE_STEP_CHECKPOINTS:-0}"

usage() {
  cat <<'EOF'
Usage:
  EXPERIMENT=<name> GPUS=0,1,2,3 scripts/company/19_run_sav_stage1_ablation.sh

Priority experiment names:
  tv21_proj_sam21l_msehr
  tv21_proj_sam21l_msehr_cos025
  tv21_adapter_sam21l_msehr
  tv21_proj_sam21bplus_msehr
  tv11_proj_sam21l_msehr
  tv5_proj_sam21l_msehr
  tv11_proj_sam21l_msehr_cos025
  tv5_proj_sam21l_msehr_cos025
  tv21_proj_sam21l_image_only
  tv21_proj_sam21l_hr025
  tv21_proj_sam21l_msehr_l1_025
  tv21_proj_sam21l_msehr_cos1
  tv21_adapter_sam21l_msehr_cos025
  tv11_adapter_sam21l_msehr
  tv5_adapter_sam21l_msehr
  tv11_proj_sam21bplus_msehr
  tv5_proj_sam21bplus_msehr
  tv21_proj_sam21l_msehr_seed2
  tv21_proj_sam21l_msehr_vbal64
EOF
}

nproc_from_gpus() {
  python - "${GPUS}" <<'PY'
import sys
print(len([part for part in sys.argv[1].split(",") if part.strip()]))
PY
}

steps_for_epochs() {
  python - "${MANIFEST}" "${EPOCHS}" "${BATCH_SIZE}" "$(nproc_from_gpus)" <<'PY'
import math
import sys
import pandas as pd
manifest, epochs, batch, nproc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
df = pd.read_parquet(manifest) if manifest.endswith(".parquet") else pd.read_csv(manifest)
train = int((df["split"] == "train").sum())
global_batch = batch * nproc
steps_per_epoch = max(math.ceil(train / global_batch), 1)
max_steps = max(math.ceil(steps_per_epoch * epochs), 1)
save_every = max(math.ceil(steps_per_epoch / 2), 1)
print(f"{max_steps} {save_every}")
PY
}

set_tinyvit() {
  case "$1" in
    tv21)
      TINYVIT_MODEL_NAME="tiny_vit_21m_512.dist_in22k_ft_in1k"
      TINYVIT_CKPT="${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
      BATCH_SIZE="${BATCH_SIZE:-4}"
      ;;
    tv11)
      TINYVIT_MODEL_NAME="tiny_vit_11m_224.dist_in22k_ft_in1k"
      TINYVIT_CKPT="${CHECKPOINT_ROOT}/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors"
      BATCH_SIZE="${BATCH_SIZE:-8}"
      ;;
    tv5)
      TINYVIT_MODEL_NAME="tiny_vit_5m_224.dist_in22k_ft_in1k"
      TINYVIT_CKPT="${CHECKPOINT_ROOT}/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors"
      BATCH_SIZE="${BATCH_SIZE:-16}"
      ;;
    *)
      echo "unsupported tinyvit size: $1" >&2
      exit 2
      ;;
  esac
}

set_teacher() {
  case "$1" in
    sam21l)
      SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
      SAM2_CKPT="${CHECKPOINT_ROOT}/sam2.1/sam2.1_hiera_large.pt"
      ;;
    sam21bplus)
      SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_b+.yaml"
      SAM2_CKPT="${CHECKPOINT_ROOT}/sam2.1/sam2.1_hiera_base_plus.pt"
      ;;
    *)
      echo "unsupported teacher: $1" >&2
      exit 2
      ;;
  esac
}

set_loss() {
  LAMBDA_MSE=1.0
  LAMBDA_HR=1.0
  LAMBDA_COS=0.0
  LAMBDA_L1=0.0
  case "$1" in
    msehr) ;;
    msehr_cos025) LAMBDA_COS=0.25 ;;
    image_only) LAMBDA_HR=0.0 ;;
    hr025) LAMBDA_HR=0.25 ;;
    msehr_l1_025) LAMBDA_L1=0.25 ;;
    msehr_cos1) LAMBDA_COS=1.0 ;;
    *)
      echo "unsupported loss: $1" >&2
      exit 2
      ;;
  esac
}

configure_experiment() {
  case "${EXPERIMENT}" in
    tv21_proj_sam21l_msehr) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr ;;
    tv21_proj_sam21l_msehr_cos025) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr_cos025 ;;
    tv21_adapter_sam21l_msehr) set_tinyvit tv21; ADAPTER_MODE=residual_dwconv; set_teacher sam21l; set_loss msehr ;;
    tv21_proj_sam21bplus_msehr) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21bplus; set_loss msehr ;;
    tv11_proj_sam21l_msehr) set_tinyvit tv11; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr ;;
    tv5_proj_sam21l_msehr) set_tinyvit tv5; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr ;;
    tv11_proj_sam21l_msehr_cos025) set_tinyvit tv11; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr_cos025 ;;
    tv5_proj_sam21l_msehr_cos025) set_tinyvit tv5; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr_cos025 ;;
    tv21_proj_sam21l_image_only) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss image_only ;;
    tv21_proj_sam21l_hr025) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss hr025 ;;
    tv21_proj_sam21l_msehr_l1_025) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr_l1_025 ;;
    tv21_proj_sam21l_msehr_cos1) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr_cos1 ;;
    tv21_adapter_sam21l_msehr_cos025) set_tinyvit tv21; ADAPTER_MODE=residual_dwconv; set_teacher sam21l; set_loss msehr_cos025 ;;
    tv11_adapter_sam21l_msehr) set_tinyvit tv11; ADAPTER_MODE=residual_dwconv; set_teacher sam21l; set_loss msehr ;;
    tv5_adapter_sam21l_msehr) set_tinyvit tv5; ADAPTER_MODE=residual_dwconv; set_teacher sam21l; set_loss msehr ;;
    tv11_proj_sam21bplus_msehr) set_tinyvit tv11; ADAPTER_MODE=projection; set_teacher sam21bplus; set_loss msehr ;;
    tv5_proj_sam21bplus_msehr) set_tinyvit tv5; ADAPTER_MODE=projection; set_teacher sam21bplus; set_loss msehr ;;
    tv21_proj_sam21l_msehr_seed2) set_tinyvit tv21; ADAPTER_MODE=projection; set_teacher sam21l; set_loss msehr; SEED="${SEED:-250107257}" ;;
    tv21_proj_sam21l_msehr_vbal64)
      set_tinyvit tv21
      ADAPTER_MODE=projection
      set_teacher sam21l
      set_loss msehr
      if [[ "${MANIFEST}" == "${DEFAULT_MANIFEST}" ]]; then
        MANIFEST="${SAM2D_ROOT}/manifests/stage1_vbal64_6fps.parquet"
      fi
      ;;
    ""|-h|--help) usage; exit 0 ;;
    *) usage; echo "unknown EXPERIMENT=${EXPERIMENT}" >&2; exit 2 ;;
  esac
}

configure_experiment
read -r computed_max_steps computed_save_every < <(steps_for_epochs)
MAX_STEPS="${MAX_STEPS:-${computed_max_steps}}"
SAVE_EVERY="${SAVE_EVERY:-${computed_save_every}}"
EVAL_EVERY="${EVAL_EVERY:-${computed_save_every}}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${EXPERIMENT}}"
WANDB_NAME="${WANDB_NAME:-${EXPERIMENT}}"

DATA_ROOT="${DATA_ROOT}" \
SAM2D_ROOT="${SAM2D_ROOT}" \
MANIFEST="${MANIFEST}" \
RUN_DIR="${RUN_DIR}" \
GPUS="${GPUS}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
MAX_STEPS="${MAX_STEPS}" \
LR="${LR}" \
WEIGHT_DECAY="${WEIGHT_DECAY}" \
PROJECTION_WARMUP_STEPS="${PROJECTION_WARMUP_STEPS}" \
LR_WARMUP_STEPS="${LR_WARMUP_STEPS}" \
MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
LAMBDA_MSE="${LAMBDA_MSE}" \
LAMBDA_HR="${LAMBDA_HR}" \
LAMBDA_COS="${LAMBDA_COS}" \
LAMBDA_L1="${LAMBDA_L1}" \
SEED="${SEED:-250107256}" \
ADAPTER_MODE="${ADAPTER_MODE}" \
TINYVIT_MODEL_NAME="${TINYVIT_MODEL_NAME}" \
TINYVIT_CKPT="${TINYVIT_CKPT}" \
SAM2_CONFIG="${SAM2_CONFIG}" \
SAM2_CKPT="${SAM2_CKPT}" \
TRAIN_SPLIT=train \
VAL_SPLIT=val_sav \
MAX_VAL_ITEMS="${MAX_VAL_ITEMS}" \
VAL_MAX_BATCHES="${VAL_MAX_BATCHES}" \
LOG_EVERY="${LOG_EVERY}" \
SAVE_EVERY="${SAVE_EVERY}" \
SAVE_STEP_CHECKPOINTS="${SAVE_STEP_CHECKPOINTS}" \
EVAL_EVERY="${EVAL_EVERY}" \
RESUME="${RESUME}" \
NO_WANDB="${NO_WANDB:-0}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_NAME="${WANDB_NAME}" \
scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh train
