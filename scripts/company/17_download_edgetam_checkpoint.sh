#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/group-volume/danny-dataset/sam2_distill/checkpoints/edgetam/edgetam.pt}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
HF_REPO_ID="${HF_REPO_ID:-facebook/EdgeTAM}"
HF_FILENAME="${HF_FILENAME:-edgetam.pt}"
FORCE="${FORCE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
      shift 2
      ;;
    --edgetam-root)
      EDGETAM_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  scripts/company/17_download_edgetam_checkpoint.sh [--out /path/to/edgetam.pt] [--edgetam-root /user-volume/repo/EdgeTAM] [--force]

Default output:
  /group-volume/danny-dataset/sam2_distill/checkpoints/edgetam/edgetam.pt

Download/copy priority:
  1. Copy EDGETAM_ROOT/checkpoints/edgetam.pt if it already exists.
  2. Download facebook/EdgeTAM edgetam.pt from Hugging Face.
  3. Download the same file through the direct Hugging Face resolve URL.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "${OUT}")"

if [[ -f "${OUT}" && "${FORCE}" != "1" ]]; then
  echo "exists: ${OUT}"
else
  LOCAL_SRC="${EDGETAM_ROOT}/checkpoints/edgetam.pt"
  if [[ -f "${LOCAL_SRC}" && "${FORCE}" != "1" ]]; then
    echo "copy: ${LOCAL_SRC} -> ${OUT}"
    cp -n "${LOCAL_SRC}" "${OUT}"
  else
    echo "download: hf://${HF_REPO_ID}/${HF_FILENAME}"
    if python - "${OUT}" "${HF_REPO_ID}" "${HF_FILENAME}" <<'PY'
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

dst = Path(sys.argv[1])
repo_id = sys.argv[2]
filename = sys.argv[3]
src = Path(hf_hub_download(repo_id=repo_id, filename=filename))
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(dst)
PY
    then
      :
    else
      echo "huggingface_hub download failed; trying direct URL" >&2
      URL="https://huggingface.co/${HF_REPO_ID}/resolve/main/${HF_FILENAME}"
      if command -v wget >/dev/null 2>&1; then
        wget -c "${URL}" -O "${OUT}"
      elif command -v curl >/dev/null 2>&1; then
        curl -L --fail --continue-at - "${URL}" -o "${OUT}"
      else
        echo "Need huggingface_hub, wget, or curl to download ${URL}" >&2
        exit 1
      fi
    fi
  fi
fi

python - "${OUT}" <<'PY'
import json
from pathlib import Path
import sys

import torch

path = Path(sys.argv[1])
ckpt = torch.load(path, map_location="cpu", weights_only=True)
if not isinstance(ckpt, dict) or "model" not in ckpt or not isinstance(ckpt["model"], dict):
    raise SystemExit(f"Invalid EdgeTAM checkpoint format: {path}")
summary = {
    "status": "pass",
    "checkpoint": str(path),
    "bytes": path.stat().st_size,
    "num_tensors": len(ckpt["model"]),
}
summary_path = path.with_suffix(".summary.json")
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

(
  cd "$(dirname "${OUT}")"
  sha256sum "$(basename "${OUT}")" > "$(basename "${OUT}").sha256"
)

echo "ready: ${OUT}"
echo "summary: ${OUT%.pt}.summary.json"
echo "sha256: ${OUT}.sha256"
