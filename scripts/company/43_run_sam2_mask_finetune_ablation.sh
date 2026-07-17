#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

VARIANT="${1:-list}"
VARIANTS=(
  decoder_lr2e7
  decoder_lr5e7
  decoder_lr2e6
  encdec_low_frozenbn
  encdec_low_trainbn
  decoder_lr5e7_boxonly
)

if [[ "${VARIANT}" == "list" ]]; then
  printf '%s\n' "${VARIANTS[@]}"
  return 0 2>/dev/null || exit 0
fi

if [[ "${VARIANT}" == "all" ]]; then
  STATUS=0
  for item in "${VARIANTS[@]}"; do
    "${BASH_SOURCE[0]}" "${item}"
    STATUS="$?"
    if [[ "${STATUS}" -ne 0 ]]; then
      echo "[ERROR] mask ablation failed: ${item}" >&2
      break
    fi
  done
  return "${STATUS}" 2>/dev/null || exit "${STATUS}"
fi

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
ABLATION_ROOT="${MASK_ABLATION_ROOT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v1}"
BASE_CHECKPOINT="${MASK_ABLATION_BASE_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_task_finetune_tv21_v2/stage1_encoder_task_2ep_v2/checkpoints/checkpoint.pt}"
BASE_STAGE_NAME="shared_stage1_encoder_task_2ep_v2"
WAIT_SECONDS="${MASK_ABLATION_WAIT_SECONDS:-43200}"

export STAGE2_MODE=mask_decoder_only
export STAGE2_EPOCHS=1
export STAGE2_FRAMES=2
export STAGE2_ENCODER_LR=1.0e-7
export STAGE2_ENCODER_LR_END=2.0e-8
export STAGE2_HEAD_LR=5.0e-7
export STAGE2_HEAD_LR_END=1.0e-7
export TASK_FREEZE_BATCHNORM=true
export TASK_NUM_CORRECTION_POINTS=1

case "${VARIANT}" in
  decoder_lr2e7)
    export STAGE2_HEAD_LR=2.0e-7
    export STAGE2_HEAD_LR_END=5.0e-8
    ;;
  decoder_lr5e7)
    ;;
  decoder_lr2e6)
    export STAGE2_HEAD_LR=2.0e-6
    export STAGE2_HEAD_LR_END=2.0e-7
    ;;
  encdec_low_frozenbn)
    export STAGE2_MODE=image_encoder_mask_decoder
    ;;
  encdec_low_trainbn)
    export STAGE2_MODE=image_encoder_mask_decoder
    export TASK_FREEZE_BATCHNORM=false
    ;;
  decoder_lr5e7_boxonly)
    export TASK_NUM_CORRECTION_POINTS=0
    ;;
  *)
    echo "[ERROR] Unknown mask ablation: ${VARIANT}" >&2
    printf 'Valid variants:\n%s\n' "${VARIANTS[*]}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

checkpoint_ready() {
  python - "${BASE_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except (FileNotFoundError, EOFError, RuntimeError):
    raise SystemExit(1)
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) >= 2 else 1)
PY
}

if ! checkpoint_ready; then
  echo "Waiting up to ${WAIT_SECONDS}s for shared Stage-1 checkpoint:"
  echo "  ${BASE_CHECKPOINT}"
  DEADLINE=$(( $(date +%s) + WAIT_SECONDS ))
  while ! checkpoint_ready; do
    if [[ "$(date +%s)" -ge "${DEADLINE}" ]]; then
      echo "[ERROR] Timed out waiting for ${BASE_CHECKPOINT}" >&2
      return 1 2>/dev/null || exit 1
    fi
    sleep 300
  done
fi

export RUN_ROOT="${ABLATION_ROOT}/${VARIANT}"
export WANDB_PROJECT="${MASK_ABLATION_WANDB_PROJECT:-sam2-mask-finetune-ablation-v1}"
export WANDB_LOSS_EMA_BETA=0.98
export SKIP_DONE="${MASK_ABLATION_SKIP_DONE:-1}"
export STAGE1_NAME="${BASE_STAGE_NAME}"
export STAGE2_NAME="mask_${VARIANT}"
unset WANDB_RUN_ID

mkdir -p "${RUN_ROOT}/${BASE_STAGE_NAME}/checkpoints"
ln -sfn "${BASE_CHECKPOINT}" \
  "${RUN_ROOT}/${BASE_STAGE_NAME}/checkpoints/checkpoint.pt"

echo "============================================================"
echo "Mask fine-tuning ablation: ${VARIANT}"
echo "Base checkpoint: ${BASE_CHECKPOINT}"
echo "Mode: ${STAGE2_MODE}"
echo "Encoder LR: ${STAGE2_ENCODER_LR} -> ${STAGE2_ENCODER_LR_END}"
echo "Head LR: ${STAGE2_HEAD_LR} -> ${STAGE2_HEAD_LR_END}"
echo "Freeze BatchNorm: ${TASK_FREEZE_BATCHNORM}"
echo "Correction points: ${TASK_NUM_CORRECTION_POINTS}"
echo "Run root: ${RUN_ROOT}"
echo "W&B project: ${WANDB_PROJECT}"
echo "============================================================"

scripts/company/39_run_sam2_task_finetune_3stage.sh stage2
STATUS="$?"
echo "Mask fine-tuning ablation status: ${STATUS}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
