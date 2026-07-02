#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAM2D_ROOT:-/danny-dataset/sam2_distill}"
COCO_RAW="${COCO_RAW:-/danny-dataset/coco2017_raw}"
PILOT_ROOT="${ROOT}/pilot/coco_1000train_100val"
RUN_DIR="${ROOT}/runs/stage1_coco_pilot"
GPUS="${GPUS:-0,1}"
TRAIN_COUNT="${TRAIN_COUNT:-1000}"
VAL_COUNT="${VAL_COUNT:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-1000}"
SHARD_SIZE="${SHARD_SIZE:-128}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"

SAM2_CKPT="${ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt"
TINYVIT_CKPT="${ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
MANIFEST="${PILOT_ROOT}/manifests/coco_pilot_${TRAIN_COUNT}train_${VAL_COUNT}val.parquet"
BOXES="${PILOT_ROOT}/manifests/coco_pilot_boxes.jsonl"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/04_run_coco_stage1_pilot.sh prepare
  scripts/company/04_run_coco_stage1_pilot.sh cache
  scripts/company/04_run_coco_stage1_pilot.sh train
  scripts/company/04_run_coco_stage1_pilot.sh benchmark
  scripts/company/04_run_coco_stage1_pilot.sh all

Environment overrides:
  SAM2D_ROOT=/danny-dataset/sam2_distill
  COCO_RAW=/danny-dataset/coco2017_raw
  GPUS=0,1
  BATCH_SIZE=4
  MAX_STEPS=1000
EOF
}

prepare() {
  python tools/data/prepare_coco_pilot.py \
    --coco-root "${COCO_RAW}" \
    --out-root "${PILOT_ROOT}" \
    --train-count "${TRAIN_COUNT}" \
    --val-count "${VAL_COUNT}" \
    --remove-archives \
    --remove-extracted-images
}

cache() {
  bash scripts/company/03_cache_teacher_embeddings.sh \
    --manifest "${MANIFEST}" \
    --teacher large \
    --root "${ROOT}" \
    --out "${PILOT_ROOT}/teacher_cache/sam2p1_large" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${CACHE_NUM_WORKERS}" \
    --shard-size "${SHARD_SIZE}" \
    --gpus "${GPUS}"
}

train() {
  NPROC="$(python - "${GPUS}" <<'PY'
import sys
print(len([part for part in sys.argv[1].split(",") if part.strip()]))
PY
)"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --standalone \
    --nproc-per-node "${NPROC}" \
    tools/train/train_stage1.py \
      --manifest "${MANIFEST}" \
      --cache-root "${PILOT_ROOT}/teacher_cache/sam2p1_large" \
      --tinyvit-checkpoint "${TINYVIT_CKPT}" \
      --out-dir "${RUN_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${TRAIN_NUM_WORKERS}" \
      --max-steps "${MAX_STEPS}" \
      --wandb-project "${WANDB_PROJECT:-sam2-distill-stage1}" \
      --wandb-name "${WANDB_NAME:-coco-pilot-stage1}"
}

benchmark() {
  python tools/benchmark/benchmark_student_sam2_masks.py \
    --boxes "${BOXES}" \
    --split val \
    --student-checkpoint "${RUN_DIR}/checkpoints/last.pt" \
    --tinyvit-checkpoint "${TINYVIT_CKPT}" \
    --sam2-checkpoint "${SAM2_CKPT}" \
    --out-dir "${RUN_DIR}/benchmark_val" \
    --limit "${VAL_COUNT}" \
    --save-overlays 50 \
    --device cuda
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  cache)
    cache
    ;;
  train)
    train
    ;;
  benchmark)
    benchmark
    ;;
  all)
    prepare
    cache
    train
    benchmark
    ;;
  *)
    usage
    exit 2
    ;;
esac
