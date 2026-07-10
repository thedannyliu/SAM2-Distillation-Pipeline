#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam31_stage1_ablation_v1/node3_relations}"
WANDB_PROJECT="${WANDB_PROJECT:-sam31-distill-stage1-ablation-v1}"
GPUS="${GPUS:-0,1,2,3}"

run_experiment() {
  local name="$1" cosine="$2" relation="$3"
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
  PROJECTION_WARMUP_STEPS=2000 \
  LR_WARMUP_STEPS=2000 \
  LAMBDA_MSE=1.0 \
  LAMBDA_COS="${cosine}" \
  LAMBDA_RELATION="${relation}" \
  RELATION_GRID_SIZE=18 \
  VAL_MAX_BATCHES=0 \
  ADAPTER_MODE=residual_dwconv \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_NAME="${name}" \
  RUN_DIR="${RUN_ROOT}/${name}" \
  RESUME=auto \
  NO_WANDB=0 \
    scripts/company/26_run_sam31_stage1_tv21.sh train
}

run_experiment n3_cos150_adapter_ft_w2k 1.5 0.0
run_experiment n3_relation010_adapter_ft_w2k 0.0 0.1
run_experiment n3_cos025_relation010_adapter_ft_w2k 0.25 0.1
