#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam31_stage1_ablation_v1/node2_interface}"
WANDB_PROJECT="${WANDB_PROJECT:-sam31-distill-stage1-ablation-v1}"
GPUS="${GPUS:-0,1,2,3}"

run_experiment() {
  local name="$1" adapter="$2" projection_warmup="$3"
  echo "===== ${name} ====="
  DATA_ROOT="${DATA_ROOT}" \
  MANIFEST="${MANIFEST}" \
  GPUS="${GPUS}" \
  BATCH_SIZE=4 \
  NUM_WORKERS=16 \
  MAX_TRAIN_ITEMS=0 \
  EPOCHS=5 \
  LR=1e-4 \
  MIN_LR=1e-6 \
  PROJECTION_WARMUP_STEPS="${projection_warmup}" \
  LR_WARMUP_STEPS=2000 \
  LAMBDA_MSE=1.0 \
  LAMBDA_COS=0.25 \
  LAMBDA_RELATION=0.0 \
  VAL_MAX_BATCHES=0 \
  ADAPTER_MODE="${adapter}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_NAME="${name}" \
  RUN_DIR="${RUN_ROOT}/${name}" \
  RESUME=auto \
  NO_WANDB=0 \
    scripts/company/26_run_sam31_stage1_tv21.sh train
}

run_experiment n2_projection_cos025_ft_w2k projection 2000
run_experiment n2_adapter_cos025_frozen residual_dwconv 999999999
run_experiment n2_adapter_cos025_ft_w0 residual_dwconv 0
