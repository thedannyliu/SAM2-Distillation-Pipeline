#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SA1B_ROOT="${SA1B_ROOT:-${DATA_ROOT}/SA-1B/hf_hdtech_sa1b_online_v1}"
SAM2_UPSTREAM="${SAM2_UPSTREAM:-/user-volume/repo/facebookresearch-sam2}"

HF_REPO_ID="${HF_REPO_ID:-hdtech/SA-1B}"
HF_REVISION="${HF_REVISION:-main}"
HF_SPLIT="${HF_SPLIT:-train}"
HF_MAX_IMAGES="${HF_MAX_IMAGES:-25000}"
HF_MAX_GB="${HF_MAX_GB:-0}"
HF_SHUFFLE_BUFFER_SIZE="${HF_SHUFFLE_BUFFER_SIZE:-10000}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-sam2_stage1_hf_sa1b_online_tinyvit21m_v1}"

MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/hf_sa1b_online_tinyvit21m_v1.parquet}"
RUN_DIR="${RUN_DIR:-${SAM2D_ROOT}/runs/stage1_online_teacher_hf_sa1b_tinyvit21m}"
GPUS="${GPUS:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CKPT="${SAM2_CKPT:-${CHECKPOINT_ROOT}/sam2.1/sam2.1_hiera_large.pt}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-10000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PROJECTION_WARMUP_STEPS="${PROJECTION_WARMUP_STEPS:-1000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-25}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TEACHER_AMP_DTYPE="${TEACHER_AMP_DTYPE:-bf16}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-}"
MAX_VAL_ITEMS="${MAX_VAL_ITEMS:-1000}"
RESUME="${RESUME:-auto}"
LAMBDA_MSE="${LAMBDA_MSE:-1.0}"
LAMBDA_L1="${LAMBDA_L1:-0.0}"
LAMBDA_COS="${LAMBDA_COS:-1.0}"
LAMBDA_HR="${LAMBDA_HR:-1.0}"

WANDB_PROJECT="${WANDB_PROJECT:-sam2-distill-stage1-online-teacher}"
WANDB_NAME="${WANDB_NAME:-hf-sa1b-online-teacher-tinyvit21m}"
NO_WANDB="${NO_WANDB:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh setup
  scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh download
  scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh train
  scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh all

This Stage 1 run trains TinyViT-21M from online SAM2.1-Hiera-L teacher features.
Teacher embeddings are not written to disk.
EOF
}

nproc_from_gpus() {
  python - "${GPUS}" <<'PY'
import sys
print(len([part for part in sys.argv[1].split(",") if part.strip()]))
PY
}

setup_env() {
  python -m pip install --user -r requirements-stage1.txt
  python -m pip install --user datasets fvcore iopath
}

download() {
  python tools/data/download_hf_sa1b_images.py \
    --repo-id "${HF_REPO_ID}" \
    --revision "${HF_REVISION}" \
    --split "${HF_SPLIT}" \
    --out-root "${SA1B_ROOT}" \
    --manifest "${MANIFEST}" \
    --max-images "${HF_MAX_IMAGES}" \
    --max-gb "${HF_MAX_GB}" \
    --seed "${SEED}" \
    --val-fraction "${VAL_FRACTION}" \
    --shuffle-buffer-size "${HF_SHUFFLE_BUFFER_SIZE}" \
    --resume
}

check_train_inputs() {
  local missing=0
  for path in "${MANIFEST}" "${SAM2_CKPT}" "${TINYVIT_CKPT}"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing required training input: ${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    cat >&2 <<EOF

Fix:
  cd /user-volume/repo/SAM2-Distillation-Pipeline
  scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh download

If weights are missing:
  bash scripts/company/01_download_weights.sh --out ${CHECKPOINT_ROOT}

Current manifest path:
  MANIFEST=${MANIFEST}
EOF
    exit 1
  fi
}

train() {
  local nproc args
  check_train_inputs
  nproc="$(nproc_from_gpus)"
  mkdir -p "${RUN_DIR}"
  args=(
    --manifest "${MANIFEST}"
    --teacher-config "${SAM2_CONFIG}"
    --teacher-checkpoint "${SAM2_CKPT}"
    --tinyvit-checkpoint "${TINYVIT_CKPT}"
    --out-dir "${RUN_DIR}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --max-steps "${MAX_STEPS}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --projection-warmup-steps "${PROJECTION_WARMUP_STEPS}"
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --lambda-mse "${LAMBDA_MSE}"
    --lambda-l1 "${LAMBDA_L1}"
    --lambda-cos "${LAMBDA_COS}"
    --lambda-hr "${LAMBDA_HR}"
    --amp-dtype "${AMP_DTYPE}"
    --teacher-amp-dtype "${TEACHER_AMP_DTYPE}"
    --log-every "${LOG_EVERY}"
    --save-every "${SAVE_EVERY}"
    --eval-every "${EVAL_EVERY}"
    --val-max-batches "${VAL_MAX_BATCHES}"
    --max-val-items "${MAX_VAL_ITEMS}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-name "${WANDB_NAME}"
  )
  if [[ -n "${MAX_TRAIN_ITEMS}" ]]; then
    args+=(--max-train-items "${MAX_TRAIN_ITEMS}")
  fi
  if [[ "${RESUME}" == "auto" && -f "${RUN_DIR}/checkpoints/last.pt" ]]; then
    args+=(--resume "${RUN_DIR}/checkpoints/last.pt")
  elif [[ "${RESUME}" != "auto" && -n "${RESUME}" ]]; then
    args+=(--resume "${RESUME}")
  fi
  if [[ "${NO_WANDB}" -eq 1 ]]; then
    args+=(--no-wandb)
  fi
  PYTHONPATH="${SAM2_UPSTREAM}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --standalone \
    --nproc-per-node "${nproc}" \
    tools/train/train_stage1_online_teacher.py \
    "${args[@]}"
}

case "${1:-}" in
  setup)
    setup_env
    ;;
  download)
    download
    ;;
  train)
    train
    ;;
  all)
    setup_env
    download
    train
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
