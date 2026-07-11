#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet}"
NUM_WORKERS="${NUM_WORKERS:-64}"

python tools/data/rebase_sav_stage1_manifest.py \
  --source-manifest "${SOURCE_MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --sav-root "${SAV_ROOT}" \
  --workers "${NUM_WORKERS}"

echo "mounted manifest: ${OUTPUT_MANIFEST}"
