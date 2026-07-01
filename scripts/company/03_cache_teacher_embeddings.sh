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
SHARD_IDS=""
GPUS=""
DEVICE="auto"
OVERWRITE=0

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
    --shard-ids)
      SHARD_IDS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
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
  --device "${DEVICE}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${NUM_SHARDS}" ]]; then
  ARGS+=(--num-shards "${NUM_SHARDS}")
fi

if [[ -n "${SHARD_IDS}" ]]; then
  ARGS+=(--shard-ids "${SHARD_IDS}")
fi

if [[ "${OVERWRITE}" -eq 1 ]]; then
  ARGS+=(--overwrite)
fi

if [[ -n "${GPUS}" ]]; then
  NPROC="$(python - "${GPUS}" <<'PY'
import sys

gpus = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
print(len(gpus))
PY
)"
  if [[ "${NPROC}" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
      --standalone \
      --nproc-per-node "${NPROC}" \
      tools/cache/cache_teacher_image_outputs.py "${ARGS[@]}"
  else
    CUDA_VISIBLE_DEVICES="${GPUS}" python tools/cache/cache_teacher_image_outputs.py "${ARGS[@]}"
  fi
else
  python tools/cache/cache_teacher_image_outputs.py "${ARGS[@]}"
fi
