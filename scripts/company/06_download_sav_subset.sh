#!/usr/bin/env bash
set -euo pipefail

SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAV_URL_LIST="${SAV_URL_LIST:-${SAV_ROOT}/manifests/sav_download_urls.txt}"
RAW_ROOT="${SAV_RAW_ROOT:-${SAV_ROOT}/_downloads_300g}"
METADATA_ROOT="${SAV_METADATA_ROOT:-${SAV_ROOT}/manifests}"
DONE_ROOT="${SAV_DONE_ROOT:-${METADATA_ROOT}/download_extract_done_300g}"
BUDGET_GB="${SAV_BUDGET_GB:-300}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/06_download_sav_subset.sh

Required:
  Accept Meta's SA-V dataset terms and save the current signed URL list at:
    /group-volume/danny-dataset/SA-V/manifests/sav_download_urls.txt

Default behavior:
  - reads SA-V signed URLs from SAV_URL_LIST
  - prioritizes val/test URLs before train URLs
  - downloads archives under /group-volume/danny-dataset/SA-V/_downloads_300g
  - extracts each archive into /group-volume/danny-dataset/SA-V
  - removes each archive after successful extraction
  - stops when the existing dataset plus the next known-size archive would pass
    SAV_BUDGET_GB, default 300

Environment overrides:
  SAV_ROOT=/group-volume/danny-dataset/SA-V
  SAV_URL_LIST=$SAV_ROOT/manifests/sav_download_urls.txt
  SAV_RAW_ROOT=$SAV_ROOT/_downloads_300g
  SAV_METADATA_ROOT=$SAV_ROOT/manifests
  SAV_DONE_ROOT=$SAV_METADATA_ROOT/download_extract_done_300g
  SAV_BUDGET_GB=300
  KEEP_ARCHIVES=0
  DRY_RUN=0

Examples:
  DRY_RUN=1 scripts/company/06_download_sav_subset.sh
  SAV_BUDGET_GB=300 scripts/company/06_download_sav_subset.sh
  KEEP_ARCHIVES=1 SAV_BUDGET_GB=300 scripts/company/06_download_sav_subset.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${SAV_URL_LIST}" ]]; then
  cat >&2 <<EOF
Missing SA-V URL list:
  ${SAV_URL_LIST}

Visit https://ai.meta.com/datasets/segment-anything-video-downloads/,
accept the dataset terms, and paste the current signed download URLs into the
file above. Do not commit signed URLs to git.
EOF
  exit 2
fi

mkdir -p "${RAW_ROOT}" "${METADATA_ROOT}" "${DONE_ROOT}"

CLEAN_URL_LIST="${METADATA_ROOT}/sav_download_urls_clean.tsv"
PROVENANCE_FILE="${METADATA_ROOT}/sav_download_300g_provenance.txt"

python - "${SAV_URL_LIST}" "${CLEAN_URL_LIST}" <<'PY'
import os
import re
import sys
from urllib.parse import urlparse, unquote

src, dst = sys.argv[1:3]

def parse_line(raw):
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    parts = re.split(r"[\s,]+", line)
    url_idx = next(
        (idx for idx, part in enumerate(parts) if part.startswith(("http://", "https://"))),
        None,
    )
    if url_idx is None:
        return None
    url = parts[url_idx]
    filename = ""
    for idx, part in enumerate(parts[:-1]):
        if part in {"-O", "-o", "--output-document", "--out"} and idx + 1 != url_idx:
            filename = parts[idx + 1]
            break
    if not filename:
        candidates = [
            part
            for idx, part in enumerate(parts)
            if idx != url_idx and part and not part.startswith("-") and part not in {"wget", "curl", "aria2c"}
        ]
        filename = candidates[0] if candidates else os.path.basename(urlparse(url).path)
    filename = os.path.basename(unquote(filename))
    if not filename:
        filename = f"sav_archive_{abs(hash(url))}.download"
    return filename, url

def priority(item):
    filename, url = item
    key = f"{filename} {url}".lower()
    if "val" in key:
        return (0, filename)
    if "test" in key:
        return (1, filename)
    return (2, filename)

records = []
seen = set()
with open(src, "r", encoding="utf-8") as handle:
    for raw in handle:
        parsed = parse_line(raw)
        if parsed is None:
            continue
        if parsed[1] in seen:
            continue
        seen.add(parsed[1])
        records.append(parsed)

if not records:
    raise SystemExit(f"no downloadable URLs found in {src}")

records = sorted(records, key=priority)
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as out:
    for filename, url in records:
        out.write(f"{filename}\t{url}\n")

print(f"SA-V URL records: {len(records)}")
print(f"Clean URL list: {dst}")
PY

budget_bytes="$(python - "${BUDGET_GB}" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024 * 1024 * 1024))
PY
)"

used_bytes() {
  du -sb "${SAV_ROOT}" 2>/dev/null | awk '{print $1 + 0}'
}

remote_size_bytes() {
  local url="$1"
  if ! command -v curl >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  curl -L --silent --show-error --head --fail --retry 3 "${url}" \
    | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub("\r", "", $2); size=$2} END{print size}'
}

download_archive() {
  local filename="$1"
  local url="$2"
  local archive="${RAW_ROOT}/${filename}"

  if [[ -s "${archive}" ]]; then
    echo "download-skip existing ${archive}"
    return 0
  fi

  echo "download ${filename}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c --auto-file-renaming=false --allow-overwrite=true \
      -d "${RAW_ROOT}" -o "${filename}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${archive}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --continue-at - -o "${archive}" "${url}"
  else
    echo "Need aria2c, wget, or curl for downloads." >&2
    return 127
  fi
}

extract_archive() {
  local filename="$1"
  local archive="${RAW_ROOT}/${filename}"
  local shard="${filename}"
  shard="${shard%.tar}"
  shard="${shard%.tar.gz}"
  shard="${shard%.tgz}"
  shard="${shard%.tar.xz}"
  shard="${shard%.txz}"
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

  echo "extract ${archive} -> ${SAV_ROOT}"
  case "${archive}" in
    *.zip)
      python - "${archive}" "${SAV_ROOT}" <<'PY'
import sys
import zipfile

archive, root = sys.argv[1:3]
with zipfile.ZipFile(archive) as zf:
    zf.extractall(root)
PY
      ;;
    *.tar|*.tar.gz|*.tgz|*.tar.xz|*.txz)
      tar -xf "${archive}" -C "${SAV_ROOT}"
      ;;
    *)
      echo "Unsupported archive extension: ${archive}" >&2
      return 4
      ;;
  esac

  touch "${done_file}"
  if [[ "${KEEP_ARCHIVES}" -eq 0 ]]; then
    rm -f "${archive}"
  fi
}

{
  echo "SAV_ROOT=${SAV_ROOT}"
  echo "SAV_URL_LIST=${SAV_URL_LIST}"
  echo "RAW_ROOT=${RAW_ROOT}"
  echo "BUDGET_GB=${BUDGET_GB}"
  echo "KEEP_ARCHIVES=${KEEP_ARCHIVES}"
  echo "CLEAN_URL_LIST=${CLEAN_URL_LIST}"
} | tee "${PROVENANCE_FILE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY_RUN=1; first selected URLs:"
  sed -n '1,20p' "${CLEAN_URL_LIST}"
  usage_human="$(du -sh "${SAV_ROOT}" 2>/dev/null | awk '{print $1}')"
  echo "Current usage: ${usage_human:-0}"
  exit 0
fi

while IFS=$'\t' read -r filename url; do
  [[ -z "${filename}" || -z "${url}" ]] && continue

  current="$(used_bytes)"
  if [[ "${current}" -ge "${budget_bytes}" ]]; then
    echo "SA-V budget reached: ${current} / ${budget_bytes} bytes"
    break
  fi

  size="$(remote_size_bytes "${url}" || true)"
  if [[ -n "${size}" && "${size}" =~ ^[0-9]+$ ]]; then
    if (( current + size > budget_bytes )); then
      echo "skip ${filename}: current ${current} + remote ${size} exceeds ${budget_bytes}"
      continue
    fi
  else
    echo "remote size unknown for ${filename}; proceeding with current usage guard"
  fi

  download_archive "${filename}" "${url}"
  extract_archive "${filename}"
  du -sh "${SAV_ROOT}"
done < "${CLEAN_URL_LIST}"

if [[ "${KEEP_ARCHIVES}" -eq 0 ]]; then
  find "${RAW_ROOT}" -type f \( \
    -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o \
    -name '*.tar.xz' -o -name '*.txz' -o -name '*.zip' -o -name '*.part' \
  \) -delete
  rmdir "${RAW_ROOT}" 2>/dev/null || true
fi

echo "SA-V download/extract finished."
echo "Dataset usage:"
du -sh "${SAV_ROOT}"
echo "Sanity checks:"
find "${SAV_ROOT}" -maxdepth 4 -type f -name '*.mp4' | head || true
find "${SAV_ROOT}" -maxdepth 5 -type f -name '*_manual.json' | head || true
find "${SAV_ROOT}" -maxdepth 5 -type f \( -name 'sav_val.txt' -o -name 'sav_test.txt' \) | head || true
