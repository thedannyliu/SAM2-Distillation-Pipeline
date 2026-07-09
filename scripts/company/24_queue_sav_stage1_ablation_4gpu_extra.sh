#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-0,1,2,3}"
QUEUE_NAME="${QUEUE_NAME:-4gpu_extra_adapter_cos}"
RUN_ROOT="${RUN_ROOT:-/group-volume/danny-dataset/sam2_distill/runs/sav_stage1_ablation_v2/${QUEUE_NAME}}"

EXPERIMENTS=(
  tv11_adapter_sam21l_msehr
  tv5_adapter_sam21l_msehr
  tv21_proj_sam21l_msehr_cos1
)

echo "queue=${QUEUE_NAME}"
echo "gpus=${GPUS}"
printf 'experiments:\n'
printf '  %s\n' "${EXPERIMENTS[@]}"

for experiment in "${EXPERIMENTS[@]}"; do
  echo "==== start ${experiment} on GPUS=${GPUS}"
  EXPERIMENT="${experiment}" \
  GPUS="${GPUS}" \
  RUN_ROOT="${RUN_ROOT}" \
  RUN_DIR="${RUN_ROOT}/${experiment}" \
  WANDB_NAME="${QUEUE_NAME}_${experiment}" \
  NO_WANDB=0 \
  SAVE_STEP_CHECKPOINTS=0 \
  BATCH_SIZE= \
  MAX_STEPS= \
  SAVE_EVERY= \
  EVAL_EVERY= \
  scripts/company/19_run_sav_stage1_ablation.sh
done
