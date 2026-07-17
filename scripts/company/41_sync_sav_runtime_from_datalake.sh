#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

BUCKET="${BUCKET:-sdp-ril}"
SOURCE_ROOT="${SOURCE_ROOT:-danny-dataset/SA-V}"
OUT_ROOT="${OUT_ROOT:-/group-volume/danny-dataset/SA-V}"
NUM_WORKERS="${NUM_WORKERS:-32}"
RESERVE_GIB="${RESERVE_GIB:-5}"

python tools/data/sync_sav_runtime_from_s3.py \
  --bucket "${BUCKET}" \
  --source-root "${SOURCE_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --workers "${NUM_WORKERS}" \
  --reserve-gib "${RESERVE_GIB}"
status=$?

if [[ "${status}" -ne 0 ]]; then
  echo "SA-V runtime data sync failed with status ${status}" >&2
  exit "${status}"
fi

echo "SA-V runtime data ready under ${OUT_ROOT}"
