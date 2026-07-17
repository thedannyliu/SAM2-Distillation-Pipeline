#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-all}"
DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"

export RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v2}"
export WANDB_PROJECT="${WANDB_PROJECT:-sam2-task-finetune-tv21-v2}"

export STAGE1_NAME="stage1_encoder_task_2ep_v2"

export STAGE2_NAME="stage2_decoder_only_task_1ep_v2"
export STAGE2_MODE="mask_decoder_only"
export STAGE2_EPOCHS=1
export STAGE2_HEAD_LR=5.0e-7
export STAGE2_HEAD_LR_END=1.0e-7

export STAGE3_NAME="stage3_encoder_decoder_memory_task_1ep_v2"

export SMOKE_NAME="smoke_decoder_only_v2"
export SMOKE_MODE="mask_decoder_only"
export SMOKE_HEAD_LR=5.0e-7
export SMOKE_HEAD_LR_END=1.0e-7

echo "Task fine-tuning recipe: v2"
echo "Stage 1: encoder only, 2 epochs"
echo "Stage 2: mask decoder only, 1 epoch, LR 5e-7 -> 1e-7"
echo "Stage 3: encoder + decoder + memory, 1 epoch"
echo "Run root: ${RUN_ROOT}"
echo "W&B project: ${WANDB_PROJECT}"

scripts/company/39_run_sam2_task_finetune_3stage.sh "${ACTION}"
STATUS="$?"
echo "SAM2 task fine-tuning v2 status: ${STATUS}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
