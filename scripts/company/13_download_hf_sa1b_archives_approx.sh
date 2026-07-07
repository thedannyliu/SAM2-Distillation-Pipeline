#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SA1B_ROOT="${SA1B_ROOT:-${DATA_ROOT}/SA-1B/hf_archives_300k_v1}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
HF_REPO_ID="${HF_REPO_ID:-hdtech/SA-1B}"
HF_REVISION="${HF_REVISION:-main}"
TARGET_IMAGES="${TARGET_IMAGES:-300000}"
SELECTION_SEED="${SELECTION_SEED:-sam2_stage1_hf_sa1b_archives_300k_v1}"
SELECTION_MODE="${SELECTION_MODE:-hash}"
MAX_ARCHIVES="${MAX_ARCHIVES:-}"
RAW_ROOT="${RAW_ROOT:-${SA1B_ROOT}/raw_archives}"
IMAGE_ROOT="${IMAGE_ROOT:-${SA1B_ROOT}/images}"
METADATA_ROOT="${METADATA_ROOT:-${SA1B_ROOT}/manifests}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/hf_sa1b_archives_300k_tinyvit21m_v1.parquet}"
VAL_FRACTION="${VAL_FRACTION:-0.02}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
DRY_RUN="${DRY_RUN:-0}"
NUM_MANIFEST_WORKERS="${NUM_MANIFEST_WORKERS:-64}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/13_download_hf_sa1b_archives_approx.sh

Downloads HF SA-1B zip archives, extracts images, and stops once the local image
count reaches TARGET_IMAGES. This avoids row-level HF streaming.

Key environment variables:
  DATA_ROOT=/group-volume/danny-dataset
  SA1B_ROOT=$DATA_ROOT/SA-1B/hf_archives_300k_v1
  HF_REPO_ID=hdtech/SA-1B
  TARGET_IMAGES=300000
  SELECTION_SEED=sam2_stage1_hf_sa1b_archives_300k_v1
  SELECTION_MODE=hash        # hash or first
  MAX_ARCHIVES=              # optional cap for dry/smoke tests
  KEEP_ARCHIVES=0            # delete zip after successful extraction
  MANIFEST=$DATA_ROOT/sam2_distill/manifests/hf_sa1b_archives_300k_tinyvit21m_v1.parquet
  NUM_MANIFEST_WORKERS=64
  DRY_RUN=0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${RAW_ROOT}" "${IMAGE_ROOT}" "${METADATA_ROOT}" "$(dirname "${MANIFEST}")"
ARCHIVE_LIST="${METADATA_ROOT}/hf_archive_selection_${TARGET_IMAGES}_${SELECTION_MODE}.txt"
PROVENANCE="${METADATA_ROOT}/hf_archive_selection_${TARGET_IMAGES}_${SELECTION_MODE}.json"
DONE_ROOT="${METADATA_ROOT}/download_done_${TARGET_IMAGES}_${SELECTION_MODE}"
mkdir -p "${DONE_ROOT}"

python - "${HF_REPO_ID}" "${HF_REVISION}" "${SELECTION_SEED}" "${SELECTION_MODE}" "${MAX_ARCHIVES}" "${ARCHIVE_LIST}" "${PROVENANCE}" <<'PY'
import hashlib
import json
import os
import re
import sys

from huggingface_hub import HfApi

repo_id, revision, seed, mode, max_archives_s, out_list, provenance = sys.argv[1:8]
if mode not in {"hash", "first"}:
    raise SystemExit("SELECTION_MODE must be hash or first")
max_archives = int(max_archives_s) if max_archives_s else None

files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
archives = [
    path for path in files
    if path.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
]
archives = [
    path for path in archives
    if re.search(r"(sa-?1b|part|sa_|\d)", os.path.basename(path).lower())
]
if not archives:
    raise SystemExit(f"No archives found in HF dataset {repo_id}@{revision}")

if mode == "hash":
    archives = sorted(
        archives,
        key=lambda path: hashlib.sha256(f"{seed}|{repo_id}|{revision}|{path}".encode()).hexdigest(),
    )
else:
    archives = sorted(archives)

selected = archives[:max_archives] if max_archives is not None else archives
with open(out_list, "w", encoding="utf-8") as out:
    for path in selected:
        out.write(path + "\n")

with open(provenance, "w", encoding="utf-8") as out:
    json.dump(
        {
            "repo_id": repo_id,
            "revision": revision,
            "seed": seed,
            "selection_mode": mode,
            "total_archives": len(archives),
            "selected_archives": len(selected),
            "max_archives": max_archives,
            "archive_list": os.path.abspath(out_list),
        },
        out,
        indent=2,
        sort_keys=True,
    )
    out.write("\n")

print(f"HF archives: {len(archives)}")
print(f"Selected archives: {len(selected)}")
print(f"Archive list: {out_list}")
print(f"Provenance: {provenance}")
PY

echo "SA1B_ROOT=${SA1B_ROOT}"
echo "RAW_ROOT=${RAW_ROOT}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "TARGET_IMAGES=${TARGET_IMAGES}"
echo "KEEP_ARCHIVES=${KEEP_ARCHIVES}"
echo "MANIFEST=${MANIFEST}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY_RUN=1; first selected archives:"
  sed -n '1,20p' "${ARCHIVE_LIST}"
  exit 0
fi

count_images() {
  find "${IMAGE_ROOT}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' '
}

extract_archive() {
  local archive="$1"
  case "${archive}" in
    *.zip)
      python - "${archive}" "${IMAGE_ROOT}" <<'PY'
import sys
import zipfile

archive, image_root = sys.argv[1:3]
with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        lower = info.filename.lower()
        if lower.endswith((".jpg", ".jpeg", ".png")):
            zf.extract(info, image_root)
PY
      ;;
    *)
      tar -xf "${archive}" -C "${IMAGE_ROOT}" --wildcards --no-anchored \
        '*.jpg' '*.jpeg' '*.png'
      ;;
  esac
}

while IFS= read -r hf_path; do
  [[ -n "${hf_path}" ]] || continue
  shard_name="$(basename "${hf_path}")"
  done_file="${DONE_ROOT}/${shard_name}.done"
  current_count="$(count_images)"
  if [[ "${current_count}" -ge "${TARGET_IMAGES}" ]]; then
    echo "target reached: ${current_count} images"
    break
  fi
  if [[ -f "${done_file}" ]]; then
    echo "skip done ${hf_path}"
    continue
  fi

  echo "download ${hf_path}"
  HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
  hf download "${HF_REPO_ID}" "${hf_path}" \
    --repo-type dataset \
    --revision "${HF_REVISION}" \
    --local-dir "${RAW_ROOT}" \
    --resume-download

  archive="${RAW_ROOT}/${hf_path}"
  if [[ ! -f "${archive}" ]]; then
    archive="${RAW_ROOT}/${shard_name}"
  fi
  if [[ ! -s "${archive}" ]]; then
    echo "missing downloaded archive: ${archive}" >&2
    exit 3
  fi

  echo "extract ${archive}"
  extract_archive "${archive}"
  touch "${done_file}"
  if [[ "${KEEP_ARCHIVES}" -eq 0 ]]; then
    rm -f "${archive}"
  fi
  echo "image_count=$(count_images)"
done < "${ARCHIVE_LIST}"

echo "Build manifest"
python tools/data/build_image_manifest.py \
  --source hf_sa1b_archives_approx \
  --image-root "${IMAGE_ROOT}" \
  --sample-percent 100 \
  --seed "${SELECTION_SEED}" \
  --val-fraction "${VAL_FRACTION}" \
  --skip-file-sha256 \
  --num-workers "${NUM_MANIFEST_WORKERS}" \
  --out "${MANIFEST}"

echo "Done"
echo "images=$(count_images)"
echo "image_root=${IMAGE_ROOT}"
echo "manifest=${MANIFEST}"
