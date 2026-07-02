#!/usr/bin/env bash
set -euo pipefail

SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAV_URL_LIST="${SAV_URL_LIST:-${SAV_ROOT}/manifests/sav_download_urls.txt}"
DEFAULT_SAV_LINK_URL="https://scontent-atl3-3.xx.fbcdn.net/m1/v/t6/An-njBc2M67ccobCOmd1fazC7fyC7aAPok19RCKBztvMXLPRK7AcAcya6fCJkYUIs-I_Qzp29deRSsnkN0C_T0Yvk3yjMdg0PGMiwVB6Qa7EDYb2.txt?_nc_gid=uqdK-mFhKk_sF3lbnwTryA&_nc_oc=Adp7zfeanqQs2lbnLXbtT15QYJj304D480-PFxGAMnw6RNdy6VbPDZ_sVKtNuhEPAEk&ccb=10-5&oh=00_AQA-4SCaEuKOP8Iz1FmZxKxpVSxtfmO1u_KG08L76gg6eQ&oe=6A6E2A18&_nc_sid=7b5a27"
SAV_LINK_URL="${SAV_LINK_URL:-${DEFAULT_SAV_LINK_URL}}"
REFRESH_SAV_URL_LIST="${REFRESH_SAV_URL_LIST:-0}"
RAW_ROOT="${SAV_RAW_ROOT:-${SAV_ROOT}/_downloads_300g}"
METADATA_ROOT="${SAV_METADATA_ROOT:-${SAV_ROOT}/manifests}"
DONE_ROOT="${SAV_DONE_ROOT:-${METADATA_ROOT}/download_extract_done_300g}"
BUDGET_GB="${SAV_BUDGET_GB:-300}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
DRY_RUN="${DRY_RUN:-0}"
SHOW_URLS="${SHOW_URLS:-0}"
TRAIN_PERCENT="${SAV_TRAIN_PERCENT:-}"
SELECTION_SEED="${SAV_SELECTION_SEED:-sav_train_1pct_v1}"
INCLUDE_EVAL_SPLITS="${SAV_INCLUDE_EVAL_SPLITS:-1}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/06_download_sav_subset.sh

Required:
  Accept Meta's SA-V dataset terms. The script has a default signed fbcdn .txt
  link-list URL, and you can override it with SAV_LINK_URL when it expires.
  Alternatively, save the current signed URL list at:
    /group-volume/danny-dataset/SA-V/manifests/sav_download_urls.txt

Default behavior:
  - downloads SAV_LINK_URL into SAV_URL_LIST when SAV_URL_LIST is missing
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
  SAV_LINK_URL=$DEFAULT_SAV_LINK_URL    # override with a refreshed signed fbcdn .txt URL
  REFRESH_SAV_URL_LIST=0                # set 1 to replace SAV_URL_LIST from SAV_LINK_URL
  SAV_RAW_ROOT=$SAV_ROOT/_downloads_300g
  SAV_METADATA_ROOT=$SAV_ROOT/manifests
  SAV_DONE_ROOT=$SAV_METADATA_ROOT/download_extract_done_300g
  SAV_BUDGET_GB=300
  SAV_TRAIN_PERCENT=                  # set 1 for deterministic ~1% train archive selection
  SAV_SELECTION_SEED=sav_train_1pct_v1
  SAV_INCLUDE_EVAL_SPLITS=1           # keep val/test archives when selecting train subset
  KEEP_ARCHIVES=0
  DRY_RUN=0
  SHOW_URLS=0                           # set 1 to print signed archive URLs during dry-run

Examples:
  REFRESH_SAV_URL_LIST=1 DRY_RUN=1 scripts/company/06_download_sav_subset.sh
  SAV_TRAIN_PERCENT=1 REFRESH_SAV_URL_LIST=1 DRY_RUN=1 scripts/company/06_download_sav_subset.sh
  SAV_TRAIN_PERCENT=1 REFRESH_SAV_URL_LIST=1 SAV_BUDGET_GB=300 scripts/company/06_download_sav_subset.sh
  REFRESH_SAV_URL_LIST=1 SAV_BUDGET_GB=300 scripts/company/06_download_sav_subset.sh
  SAV_LINK_URL='<refreshed fbcdn .txt URL>' REFRESH_SAV_URL_LIST=1 DRY_RUN=1 scripts/company/06_download_sav_subset.sh
  KEEP_ARCHIVES=1 SAV_BUDGET_GB=300 scripts/company/06_download_sav_subset.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$(dirname "${SAV_URL_LIST}")"

if [[ -n "${SAV_LINK_URL}" && "${REFRESH_SAV_URL_LIST}" -eq 1 ]]; then
  rm -f "${SAV_URL_LIST}"
fi

if [[ ! -f "${SAV_URL_LIST}" && -n "${SAV_LINK_URL}" ]]; then
  echo "download SA-V link list to ${SAV_URL_LIST}"
  if command -v wget >/dev/null 2>&1 && wget -q --timeout=30 --tries=2 -O "${SAV_URL_LIST}" "${SAV_LINK_URL}"; then
    true
  elif command -v curl >/dev/null 2>&1 && curl -L --fail --retry 2 --connect-timeout 30 --max-time 120 --silent --show-error -o "${SAV_URL_LIST}" "${SAV_LINK_URL}"; then
    true
  elif command -v python >/dev/null 2>&1 && python - "${SAV_LINK_URL}" "${SAV_URL_LIST}" <<'PY'
import sys
import urllib.request

url, out_path = sys.argv[1:3]
request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(request, timeout=30) as response, open(out_path, "wb") as out:
        out.write(response.read())
except Exception as exc:
    print(f"python download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    true
  else
    rm -f "${SAV_URL_LIST}"
    cat >&2 <<EOF
Failed to download the SA-V link list into:
  ${SAV_URL_LIST}

This usually means the default fbcdn URL expired, the company network blocked
the request, or this environment lacks wget/curl/python download support.
Refresh the link from Meta and rerun with:
  SAV_LINK_URL='<refreshed fbcdn .txt URL>' REFRESH_SAV_URL_LIST=1 scripts/company/06_download_sav_subset.sh
EOF
    exit 3
  fi
fi

if [[ ! -f "${SAV_URL_LIST}" ]]; then
  cat >&2 <<EOF
Missing SA-V URL list:
  ${SAV_URL_LIST}

Visit https://ai.meta.com/datasets/segment-anything-video-downloads/,
accept the dataset terms, then either set:
  SAV_LINK_URL='<current signed fbcdn .txt URL>'
or paste the current signed download URLs into the file above.
Replace DEFAULT_SAV_LINK_URL or override SAV_LINK_URL when the default expires.
EOF
  exit 2
fi

mkdir -p "${RAW_ROOT}" "${METADATA_ROOT}" "${DONE_ROOT}"

CLEAN_URL_LIST="${METADATA_ROOT}/sav_download_urls_clean.tsv"
SELECTED_URL_LIST="${METADATA_ROOT}/sav_download_urls_selected.tsv"
PROVENANCE_FILE="${METADATA_ROOT}/sav_download_selection_provenance.json"

python - "${SAV_URL_LIST}" "${CLEAN_URL_LIST}" "${SELECTED_URL_LIST}" "${PROVENANCE_FILE}" "${TRAIN_PERCENT}" "${SELECTION_SEED}" "${INCLUDE_EVAL_SPLITS}" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from urllib.parse import urlparse, unquote

src, clean_dst, selected_dst, provenance_dst, train_percent_s, seed, include_eval_s = sys.argv[1:8]
include_eval = include_eval_s == "1"
archive_suffixes = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".zip")

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
    if not filename.lower().endswith(archive_suffixes):
        return None
    return filename, url

def priority(item):
    filename, url = item
    key = filename.lower()
    if key.startswith("sav_val") or "_val" in key:
        return (0, filename)
    if key.startswith("sav_test") or "_test" in key:
        return (1, filename)
    return (2, filename)

def split_name(filename):
    key = filename.lower()
    if key.startswith("sav_val") or "_val" in key:
        return "val"
    if key.startswith("sav_test") or "_test" in key:
        return "test"
    return "train"

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
os.makedirs(os.path.dirname(clean_dst), exist_ok=True)
with open(clean_dst, "w", encoding="utf-8") as out:
    for filename, url in records:
        out.write(f"{filename}\t{url}\n")

train_records = [record for record in records if split_name(record[0]) == "train"]
eval_records = [record for record in records if split_name(record[0]) != "train"]
if train_percent_s:
    train_percent = float(train_percent_s)
    if not (0 < train_percent <= 100):
        raise SystemExit(f"SAV_TRAIN_PERCENT must be in (0, 100], got {train_percent}")
    selected_train_count = max(1, math.ceil(len(train_records) * train_percent / 100.0))
    selected_train = sorted(
        train_records,
        key=lambda item: hashlib.sha256(f"{seed}|{item[0]}".encode("utf-8")).hexdigest(),
    )[:selected_train_count]
    selected_train = sorted(selected_train, key=priority)
    selected_records = (eval_records if include_eval else []) + selected_train
else:
    train_percent = None
    selected_train_count = len(train_records)
    selected_train = train_records
    selected_records = records

with open(selected_dst, "w", encoding="utf-8") as out:
    for filename, url in selected_records:
        out.write(f"{filename}\t{url}\n")

provenance = {
    "source_url_list": os.path.abspath(src),
    "source_url_list_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
    "clean_url_list": os.path.abspath(clean_dst),
    "selected_url_list": os.path.abspath(selected_dst),
    "total_archives": len(records),
    "train_archives": len(train_records),
    "eval_archives": len(eval_records),
    "selected_archives": len(selected_records),
    "selected_train_archives": len(selected_train),
    "train_percent": train_percent,
    "selection_seed": seed,
    "selection_key": "sha256(seed|archive_filename)",
    "include_eval_splits": include_eval,
    "selected_filenames": [filename for filename, _ in selected_records],
}
with open(provenance_dst, "w", encoding="utf-8") as out:
    json.dump(provenance, out, indent=2, sort_keys=True)
    out.write("\n")

print(f"SA-V URL records: {len(records)}")
print(f"Train archive records: {len(train_records)}")
print(f"Selected archive records: {len(selected_records)}")
print(f"Clean URL list: {clean_dst}")
print(f"Selected URL list: {selected_dst}")
print(f"Selection provenance: {provenance_dst}")
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
  echo "TRAIN_PERCENT=${TRAIN_PERCENT}"
  echo "SELECTION_SEED=${SELECTION_SEED}"
  echo "INCLUDE_EVAL_SPLITS=${INCLUDE_EVAL_SPLITS}"
  echo "KEEP_ARCHIVES=${KEEP_ARCHIVES}"
  echo "CLEAN_URL_LIST=${CLEAN_URL_LIST}"
  echo "SELECTED_URL_LIST=${SELECTED_URL_LIST}"
  echo "PROVENANCE_FILE=${PROVENANCE_FILE}"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY_RUN=1; first selected archives:"
  if [[ "${SHOW_URLS}" -eq 1 ]]; then
    sed -n '1,20p' "${SELECTED_URL_LIST}"
  else
    cut -f1 "${SELECTED_URL_LIST}" | sed -n '1,20p'
  fi
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
done < "${SELECTED_URL_LIST}"

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
