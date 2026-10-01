#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

BUCKET="${BUCKET:-sdp-ril}"
PREFIX="${PREFIX:-danny-dataset/SA-V/sav_train/}"
MANIFEST="${MANIFEST:-/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet}"
OUT_ROOT="${OUT_ROOT:-/group-volume/danny-dataset/sam2_distill/data/sav_task_annotations}"
NUM_WORKERS="${NUM_WORKERS:-32}"
JSON_SAMPLES="${JSON_SAMPLES:-200}"

python tools/data/sync_sav_task_annotations_from_s3.py \
  --bucket "${BUCKET}" \
  --prefix "${PREFIX}" \
  --manifest "${MANIFEST}" \
  --out-root "${OUT_ROOT}" \
  --workers "${NUM_WORKERS}" \
  --json-samples "${JSON_SAMPLES}"
status=$?

if [[ "${status}" -ne 0 ]]; then
  echo "SA-V task annotation sync failed with status ${status}" >&2
  exit "${status}"
fi

echo "SA-V task annotations: ${OUT_ROOT}/sav_train"
echo "Only manifest-selected *_manual.json files were stored; no MP4 or auto JSON was copied."
