#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
IMAGE_ROOT="${IMAGE_ROOT:-/group-volume/danny-dataset/SA-1B/images_3pct}"
SOURCE="${SOURCE:-sa1b}"
SAMPLE_PERCENT="${SAMPLE_PERCENT:-100}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-sam2_stage1_sa1b_3pct_v1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SKIP_FILE_SHA256="${SKIP_FILE_SHA256:-1}"

MANIFEST="${MANIFEST:-${ROOT}/manifests/sa1b_3pct_v1.parquet}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/cache/stage1_teacher/sam2p1_large_sa1b_3pct_v1}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/stage1_mse_sa1b_3pct_8xh100}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PROJECTION_WARMUP_STEPS="${PROJECTION_WARMUP_STEPS:-2000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NONFINITE_LOSS="${NONFINITE_LOSS:-error}"
SHARD_SIZE="${SHARD_SIZE:-512}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-100}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

SAM2_CKPT="${ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt"
TINYVIT_CKPT="${ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/05_run_stage1_large_mse_8xh100.sh manifest
  scripts/company/05_run_stage1_large_mse_8xh100.sh plan-cache
  scripts/company/05_run_stage1_large_mse_8xh100.sh cache
  scripts/company/05_run_stage1_large_mse_8xh100.sh train
  scripts/company/05_run_stage1_large_mse_8xh100.sh all

Important environment overrides:
  SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill
  IMAGE_ROOT=/group-volume/danny-dataset/SA-1B/images_3pct
  GPUS=0,1,2,3,4,5,6,7
  SAMPLE_PERCENT=100
  VAL_FRACTION=0.1
  SKIP_FILE_SHA256=1
  MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sa1b_3pct_v1.parquet
  CACHE_ROOT=/group-volume/danny-dataset/sam2_distill/cache/stage1_teacher/sam2p1_large_sa1b_3pct_v1
  RUN_DIR=/group-volume/danny-dataset/sam2_distill/runs/stage1_mse_sa1b_3pct_8xh100
  BATCH_SIZE=8
  MAX_STEPS=100000
  PROJECTION_WARMUP_STEPS=2000
  LR_WARMUP_STEPS=2000
  MAX_GRAD_NORM=1.0
EOF
}

nproc_from_gpus() {
  python - "${GPUS}" <<'PY'
import sys
print(len([part for part in sys.argv[1].split(",") if part.strip()]))
PY
}

manifest() {
  ARGS=(
    --source "${SOURCE}"
    --image-root "${IMAGE_ROOT}"
    --sample-percent "${SAMPLE_PERCENT}"
    --val-fraction "${VAL_FRACTION}"
    --seed "${SEED}"
    --out "${MANIFEST}"
  )
  if [[ "${SKIP_FILE_SHA256}" -eq 1 ]]; then
    ARGS+=(--skip-file-sha256)
  fi
  python tools/data/build_image_manifest.py \
    "${ARGS[@]}"
}

plan_cache() {
  python tools/cache/plan_cache_shards.py \
    --manifest "${MANIFEST}" \
    --shard-size "${SHARD_SIZE}" \
    --num-jobs 1
}

cache() {
  bash scripts/company/03_cache_teacher_embeddings.sh \
    --manifest "${MANIFEST}" \
    --teacher large \
    --root "${ROOT}" \
    --out "${CACHE_ROOT}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${CACHE_NUM_WORKERS}" \
    --shard-size "${SHARD_SIZE}" \
    --gpus "${GPUS}"
}

train() {
  NPROC="$(nproc_from_gpus)"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --standalone \
    --nproc-per-node "${NPROC}" \
    tools/train/train_stage1.py \
      --manifest "${MANIFEST}" \
      --train-split train \
      --val-split val_sa1b \
      --cache-root "${CACHE_ROOT}" \
      --tinyvit-checkpoint "${TINYVIT_CKPT}" \
      --out-dir "${RUN_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-steps "${MAX_STEPS}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --projection-warmup-steps "${PROJECTION_WARMUP_STEPS}" \
      --lr-warmup-steps "${LR_WARMUP_STEPS}" \
      --max-grad-norm "${MAX_GRAD_NORM}" \
      --nonfinite-loss "${NONFINITE_LOSS}" \
      --lambda-mse 1.0 \
      --lambda-l1 0.0 \
      --lambda-cos 0.0 \
      --lambda-hr 1.0 \
      --amp-dtype "${AMP_DTYPE}" \
      --val-max-batches "${VAL_MAX_BATCHES}" \
      --eval-every "${EVAL_EVERY}" \
      --save-every "${SAVE_EVERY}" \
      --wandb-project "${WANDB_PROJECT:-sam2-distill-stage1}" \
      --wandb-name "${WANDB_NAME:-stage1-mse-sa1b-8xh100}"
}

case "${1:-}" in
  manifest)
    manifest
    ;;
  plan-cache)
    plan_cache
    ;;
  cache)
    cache
    ;;
  train)
    train
    ;;
  all)
    manifest
    plan_cache
    cache
    train
    ;;
  *)
    usage
    exit 2
    ;;
esac
