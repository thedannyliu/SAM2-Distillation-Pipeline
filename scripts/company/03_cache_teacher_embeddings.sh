#!/usr/bin/env bash
set -euo pipefail

MANIFEST=""
TEACHER="large"
OUT="/danny-dataset/sam2_distill/cache/stage1_teacher/sam2p1_large_sa1b_1pct_v1"
ROOT="/danny-dataset/sam2_distill"
BATCH_SIZE=8
NUM_WORKERS=8
SHARD_SIZE=512
LIMIT=""
START_SHARD=0
NUM_SHARDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --teacher)
      TEACHER="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --shard-size)
      SHARD_SIZE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --start-shard)
      START_SHARD="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MANIFEST}" ]]; then
  echo "--manifest is required" >&2
  exit 2
fi

case "${TEACHER}" in
  large)
    CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
    CHECKPOINT="${ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt"
    ;;
  base_plus|b+)
    CONFIG="configs/sam2.1/sam2.1_hiera_b+.yaml"
    CHECKPOINT="${ROOT}/checkpoints/sam2.1/sam2.1_hiera_base_plus.pt"
    ;;
  *)
    echo "Unsupported teacher: ${TEACHER}" >&2
    exit 2
    ;;
esac

ARGS=(
  --manifest "${MANIFEST}"
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --out "${OUT}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --shard-size "${SHARD_SIZE}"
  --start-shard "${START_SHARD}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${NUM_SHARDS}" ]]; then
  ARGS+=(--num-shards "${NUM_SHARDS}")
fi

python tools/cache/cache_teacher_image_outputs.py "${ARGS[@]}"
