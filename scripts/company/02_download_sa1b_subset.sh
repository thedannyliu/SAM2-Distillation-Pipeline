#!/usr/bin/env bash
set -euo pipefail

SA1B_ROOT="${SA1B_ROOT:-/group-volume/danny-dataset/SA-1B}"
LINK_FILE="${SA1B_LINK_FILE:-${SA1B_ROOT}/sa1b_links.txt}"
LINK_URL="${SA1B_LINK_URL:-}"
REFRESH_LINK_FILE="${REFRESH_LINK_FILE:-0}"
RAW_ROOT="${SA1B_RAW_ROOT:-${SA1B_ROOT}/raw_3pct}"
IMAGE_ROOT="${IMAGE_ROOT:-${SA1B_ROOT}/images_3pct}"
ANNOTATION_ROOT="${SA1B_ANNOTATION_ROOT:-${SA1B_ROOT}/annotations_3pct}"
METADATA_ROOT="${SA1B_METADATA_ROOT:-${SA1B_ROOT}/manifests}"
PERCENT="${SA1B_DOWNLOAD_PERCENT:-3}"
WORKERS="${SA1B_DOWNLOAD_WORKERS:-4}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
EXTRACT_ANNOTATIONS="${EXTRACT_ANNOTATIONS:-0}"
SELECTION_MODE="${SA1B_SELECTION_MODE:-hash}"
MAX_SHARDS="${SA1B_MAX_SHARDS:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/02_download_sa1b_subset.sh

Required:
  Put the official SA-1B URL list at:
    /group-volume/danny-dataset/SA-1B/sa1b_links.txt
  or set SA1B_LINK_URL to a URL that serves that text file.

Default behavior:
  - deterministically selects 3% of shards from the link list
  - downloads archives to /group-volume/danny-dataset/SA-1B/raw_3pct
  - extracts images to /group-volume/danny-dataset/SA-1B/images_3pct
  - removes archives after successful extraction
  - removes the raw archive directory if it is empty
  - keeps only the original link list plus a small selected-shard TSV and JSON provenance file
  - does not extract mask JSON annotations unless EXTRACT_ANNOTATIONS=1

Environment overrides:
  SA1B_ROOT=/group-volume/danny-dataset/SA-1B
  SA1B_LINK_FILE=$SA1B_ROOT/sa1b_links.txt
  SA1B_LINK_URL=              # optional URL for the official link-list .txt
  REFRESH_LINK_FILE=0         # set 1 to re-download SA1B_LINK_FILE from SA1B_LINK_URL
  SA1B_DOWNLOAD_PERCENT=3
  SA1B_DOWNLOAD_WORKERS=4
  SA1B_SELECTION_MODE=hash      # hash or first
  SA1B_MAX_SHARDS=              # optional hard cap for smoke tests
  SA1B_RAW_ROOT=$SA1B_ROOT/raw_3pct
  IMAGE_ROOT=$SA1B_ROOT/images_3pct
  SA1B_ANNOTATION_ROOT=$SA1B_ROOT/annotations_3pct
  SA1B_METADATA_ROOT=$SA1B_ROOT/manifests
  EXTRACT_ANNOTATIONS=0
  KEEP_ARCHIVES=0
  DRY_RUN=0

Examples:
  SA1B_LINK_URL='https://...' DRY_RUN=1 scripts/company/02_download_sa1b_subset.sh
  SA1B_LINK_URL='https://...' scripts/company/02_download_sa1b_subset.sh
  DRY_RUN=1 scripts/company/02_download_sa1b_subset.sh
  SA1B_DOWNLOAD_WORKERS=8 scripts/company/02_download_sa1b_subset.sh
  SA1B_MAX_SHARDS=2 DRY_RUN=1 scripts/company/02_download_sa1b_subset.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$(dirname "${LINK_FILE}")"
if [[ -n "${LINK_URL}" && "${REFRESH_LINK_FILE}" -eq 1 ]]; then
  rm -f "${LINK_FILE}"
fi
if [[ ! -f "${LINK_FILE}" && -n "${LINK_URL}" ]]; then
  echo "download SA-1B link list ${LINK_URL}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${LINK_FILE}" "${LINK_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 -o "${LINK_FILE}" "${LINK_URL}"
  else
    echo "Need wget or curl to download SA1B_LINK_URL." >&2
    exit 127
  fi
fi

if [[ ! -f "${LINK_FILE}" ]]; then
  cat >&2 <<EOF
Missing SA-1B link file:
  ${LINK_FILE}

Download/obtain the official SA-1B link list after accepting Meta's dataset terms,
then save it at the path above, set SA1B_LINK_FILE=/path/to/sa1b_links.txt,
or set SA1B_LINK_URL=https://... to let this script download the link list.
EOF
  exit 2
fi

mkdir -p "${RAW_ROOT}" "${IMAGE_ROOT}" "${ANNOTATION_ROOT}" "${METADATA_ROOT}"
SELECTION_FILE="${METADATA_ROOT}/sa1b_download_selected_${PERCENT}pct_${SELECTION_MODE}.tsv"
PROVENANCE_FILE="${METADATA_ROOT}/sa1b_download_selected_${PERCENT}pct_${SELECTION_MODE}.json"
DONE_ROOT="${METADATA_ROOT}/download_done_${PERCENT}pct_${SELECTION_MODE}"
mkdir -p "${DONE_ROOT}"

python - "${LINK_FILE}" "${PERCENT}" "${SELECTION_MODE}" "${MAX_SHARDS}" "${SELECTION_FILE}" "${PROVENANCE_FILE}" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from urllib.parse import urlparse

link_file, percent_s, mode, max_shards_s, out_path, provenance_path = sys.argv[1:7]
archive_suffixes = (".tar", ".tar.gz", ".tgz", ".zip")
percent = float(percent_s)
if not (0 < percent <= 100):
    raise SystemExit(f"SA1B_DOWNLOAD_PERCENT must be in (0, 100], got {percent}")
if mode not in {"first", "hash"}:
    raise SystemExit(f"SA1B_SELECTION_MODE must be 'first' or 'hash', got {mode!r}")
max_shards = int(max_shards_s) if max_shards_s else None

def parse_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = re.split(r"[\s,]+", line)
    url_idx = next((idx for idx, part in enumerate(parts) if part.startswith(("http://", "https://"))), None)
    if url_idx is None:
        return None
    url = parts[url_idx]
    candidates = [part for idx, part in enumerate(parts) if idx != url_idx and part]
    filename = candidates[0] if candidates else os.path.basename(urlparse(url).path)
    filename = os.path.basename(filename)
    if not filename:
        raise ValueError(f"could not derive filename from line: {line}")
    if not filename.lower().endswith(archive_suffixes):
        return None
    return filename, url

records = []
with open(link_file, "r", encoding="utf-8") as handle:
    for raw in handle:
        parsed = parse_line(raw)
        if parsed is not None:
            records.append(parsed)

if not records:
    raise SystemExit(f"no downloadable URLs found in {link_file}")

if mode == "hash":
    ordered = sorted(records, key=lambda item: hashlib.sha256("\t".join(item).encode()).hexdigest())
else:
    ordered = records

count = max(1, math.ceil(len(ordered) * percent / 100.0))
if max_shards is not None:
    count = min(count, max_shards)
selected = ordered[:count]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as out:
    for filename, url in selected:
        out.write(f"{filename}\t{url}\n")

with open(provenance_path, "w", encoding="utf-8") as out:
    json.dump(
        {
            "link_file": os.path.abspath(link_file),
            "link_file_sha256": hashlib.sha256(open(link_file, "rb").read()).hexdigest(),
            "total_link_records": len(records),
            "selected_shards": len(selected),
            "percent": percent,
            "selection_mode": mode,
            "max_shards": max_shards,
            "selection_file": os.path.abspath(out_path),
        },
        out,
        indent=2,
        sort_keys=True,
    )
    out.write("\n")

print(f"SA-1B link records: {len(records)}")
print(f"Selected shards: {len(selected)} ({percent:g}% mode={mode})")
print(f"Selection file: {out_path}")
print(f"Provenance file: {provenance_path}")
PY

echo "SA1B_ROOT=${SA1B_ROOT}"
echo "RAW_ROOT=${RAW_ROOT}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "ANNOTATION_ROOT=${ANNOTATION_ROOT}"
echo "METADATA_ROOT=${METADATA_ROOT}"
echo "DONE_ROOT=${DONE_ROOT}"
echo "KEEP_ARCHIVES=${KEEP_ARCHIVES}"
echo "EXTRACT_ANNOTATIONS=${EXTRACT_ANNOTATIONS}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY_RUN=1; selected shards:"
  sed -n '1,20p' "${SELECTION_FILE}"
  exit 0
fi

download_archive() {
  local filename="$1"
  local url="$2"
  local archive="${RAW_ROOT}/${filename}"
  local partial="${archive}.part"

  if [[ -s "${archive}" ]]; then
    echo "download-skip existing ${archive}"
    return 0
  fi

  echo "download ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -c --auto-file-renaming=false --allow-overwrite=true \
      -d "${RAW_ROOT}" -o "${filename}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${archive}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --continue-at - -o "${archive}" "${url}"
  else
    echo "Need aria2c, wget, or curl for downloads." >&2
    return 127
  fi

  if [[ -e "${partial}" && ! -e "${archive}" ]]; then
    mv "${partial}" "${archive}"
  fi
}

extract_archive() {
  local filename="$1"
  local archive="${RAW_ROOT}/${filename}"
  local shard="${filename}"
  shard="${shard%.tar}"
  shard="${shard%.tar.gz}"
  shard="${shard%.tgz}"
  shard="${shard%.zip}"
  local done_file="${DONE_ROOT}/${shard}.done"

  if [[ -f "${done_file}" ]]; then
    echo "extract-skip ${filename}"
    return 0
  fi
  if [[ ! -s "${archive}" ]]; then
    echo "archive missing after download: ${archive}" >&2
    return 3
  fi

  echo "extract images ${archive}"
  case "${archive}" in
    *.zip)
      python - "${archive}" "${IMAGE_ROOT}" "${ANNOTATION_ROOT}" "${EXTRACT_ANNOTATIONS}" <<'PY'
import os
import sys
import zipfile

archive, image_root, annotation_root, extract_annotations = sys.argv[1:5]
extract_annotations = extract_annotations == "1"
with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        lower = info.filename.lower()
        if lower.endswith((".jpg", ".jpeg", ".png")):
            zf.extract(info, image_root)
        elif extract_annotations and lower.endswith(".json"):
            zf.extract(info, annotation_root)
PY
      ;;
    *)
      tar -xf "${archive}" -C "${IMAGE_ROOT}" --wildcards --no-anchored \
        '*.jpg' '*.jpeg' '*.png'
      if [[ "${EXTRACT_ANNOTATIONS}" -eq 1 ]]; then
        tar -xf "${archive}" -C "${ANNOTATION_ROOT}" --wildcards --no-anchored '*.json' || true
      fi
      ;;
  esac

  touch "${done_file}"
  if [[ "${KEEP_ARCHIVES}" -eq 0 ]]; then
    rm -f "${archive}"
  fi
}

process_one() {
  local line="$1"
  local filename url
  filename="${line%%$'\t'*}"
  url="${line#*$'\t'}"
  download_archive "${filename}" "${url}"
  extract_archive "${filename}"
}

export RAW_ROOT IMAGE_ROOT ANNOTATION_ROOT DONE_ROOT KEEP_ARCHIVES EXTRACT_ANNOTATIONS
export -f download_archive extract_archive process_one

if [[ "${WORKERS}" -le 1 ]]; then
  while IFS= read -r line; do
    [[ -n "${line}" ]] && process_one "${line}"
  done < "${SELECTION_FILE}"
else
  xargs -r -d '\n' -P "${WORKERS}" -I{} bash -lc 'process_one "$1"' _ "{}" < "${SELECTION_FILE}"
fi

if [[ "${KEEP_ARCHIVES}" -eq 0 ]]; then
  find "${RAW_ROOT}" -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' -o -name '*.part' \) -delete
  rmdir "${RAW_ROOT}" 2>/dev/null || true
fi

echo "Downloaded/extracted selected SA-1B subset."
echo "Selection/provenance kept for reproducibility:"
echo "  ${SELECTION_FILE}"
echo "  ${PROVENANCE_FILE}"
echo "Image count:"
find "${IMAGE_ROOT}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
