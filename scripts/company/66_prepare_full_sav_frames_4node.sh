#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
NODE="${2:-}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
RAW_ROOT="${RAW_ROOT:-${SAV_ROOT}/sav_train}"
CACHE_ROOT="${CACHE_ROOT:-${SAV_ROOT}/sav_train_6fps_full}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
NUM_WORKERS="${NUM_WORKERS:-64}"
NUM_SHARDS=4
MARKER_ROOT="${CACHE_ROOT}/.four_node_shards"

shard_manifest() {
  local index="$1"
  echo "${MANIFEST%.parquet}.shard_$(printf '%03d' "${index}").parquet"
}

run_node() {
  local node="$1" index shard_manifest_path marker
  if [[ ! "${node}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] Node must be 1 through 4" >&2
    return 2
  fi
  index=$((node - 1))
  shard_manifest_path="$(shard_manifest "${index}")"
  marker="${MARKER_ROOT}/shard_$(printf '%03d' "${index}").complete"
  mkdir -p "${MARKER_ROOT}" "$(dirname "${MANIFEST}")" "${CACHE_ROOT}"
  if [[ -f "${marker}" && -f "${shard_manifest_path}" ]]; then
    echo "Skip completed shard ${index}: ${shard_manifest_path}"
    return 0
  fi
  echo "Prepare SA-V frame shard ${index}/${NUM_SHARDS} with ${NUM_WORKERS} workers"
  DATA_ROOT="/group-volume/danny-dataset" \
  SAV_ROOT="${SAV_ROOT}" \
  TRAIN_ROOT="${RAW_ROOT}" \
  OUT_ROOT="${CACHE_ROOT}" \
  MANIFEST="${shard_manifest_path}" \
  TRAIN_FRAMES_PER_VIDEO=1000000 \
  VAL_FRAMES_PER_VIDEO=0 \
  TEST_FRAMES_PER_VIDEO=0 \
  MAX_TRAIN_VIDEOS=0 \
  NUM_WORKERS="${NUM_WORKERS}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  SHARD_INDEX="${index}" \
  JPEG_QUALITY=90 \
    scripts/company/18_prepare_sav_stage1_frame_cache.sh || return $?
  touch "${marker}"
  echo "Completed shard ${index}: ${shard_manifest_path}"
}

merge_shards() {
  local index path marker inputs=()
  for index in 0 1 2 3; do
    path="$(shard_manifest "${index}")"
    marker="${MARKER_ROOT}/shard_$(printf '%03d' "${index}").complete"
    if [[ ! -f "${marker}" || ! -f "${path}" ]]; then
      echo "[ERROR] Shard ${index} is incomplete: ${path}" >&2
      return 1
    fi
    inputs+=("${path}")
  done
  python - "${MANIFEST}" "${inputs[@]}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

output = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:]]
frame = pd.concat([pd.read_parquet(path) for path in inputs], ignore_index=True)
duplicates = int(frame["sample_id"].duplicated().sum())
if duplicates:
    raise SystemExit(
        f"Refusing to merge manifests with {duplicates} duplicate sample IDs"
    )
frame = frame.sort_values(
    ["split", "video_id", "frame_idx_24fps"]
).reset_index(drop=True)
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(".tmp.parquet")
frame.to_parquet(temporary, index=False)
temporary.replace(output)
print(json.dumps({
    "status": "pass",
    "manifest": str(output),
    "rows": len(frame),
    "videos": int(frame["video_id"].nunique()),
    "inputs": [str(path) for path in inputs],
}, indent=2))
PY
}

show_status() {
  local index path marker state rows
  for index in 0 1 2 3; do
    path="$(shard_manifest "${index}")"
    marker="${MARKER_ROOT}/shard_$(printf '%03d' "${index}").complete"
    state="INCOMPLETE"
    [[ -f "${marker}" && -f "${path}" ]] && state="COMPLETE"
    rows="-"
    if [[ -f "${path}" ]]; then
      rows="$(python - "${path}" <<'PY'
import sys
import pandas as pd
print(len(pd.read_parquet(sys.argv[1], columns=["sample_id"])))
PY
)"
    fi
    printf 'shard=%d state=%-10s rows=%s manifest=%s\n' \
      "${index}" "${state}" "${rows}" "${path}"
  done
  if [[ -f "${MANIFEST}" ]]; then
    echo "Merged manifest: ${MANIFEST}"
  else
    echo "Merged manifest pending: ${MANIFEST}"
  fi
}

STATUS=0
case "${ACTION}" in
  describe)
    echo "Four-node full SA-V 6fps-aligned frame extraction"
    echo "Nodes 1-4: deterministic shards 0-3, ${NUM_WORKERS} workers each"
    echo "Raw root: ${RAW_ROOT}"
    echo "Shared frame cache: ${CACHE_ROOT}/JPEGImages"
    echo "Merged manifest: ${MANIFEST}"
    echo "Existing frames are skipped; completed shards use marker files."
    ;;
  node)
    run_node "${NODE}" || STATUS="$?"
    ;;
  merge)
    merge_shards || STATUS="$?"
    ;;
  status)
    show_status || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|node 1|node 2|node 3|node 4|status|merge}" >&2
    STATUS=2
    ;;
esac

echo "Four-node SA-V frame preparation status: ${STATUS}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
