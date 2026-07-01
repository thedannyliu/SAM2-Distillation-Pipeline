#!/usr/bin/env bash
set -euo pipefail

OUT="/danny-dataset/sam2_distill/checkpoints"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUT}/sam2.1" "${OUT}/tinyvit"

download() {
  local url="$1"
  local dst="$2"
  if [[ -f "${dst}" ]]; then
    echo "exists: ${dst}"
    return
  fi
  echo "download: ${url}"
  wget -c "${url}" -O "${dst}"
}

download \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" \
  "${OUT}/sam2.1/sam2.1_hiera_base_plus.pt"

download \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
  "${OUT}/sam2.1/sam2.1_hiera_large.pt"

TINYVIT_DST="${OUT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
download \
  "https://huggingface.co/timm/tiny_vit_21m_512.dist_in22k_ft_in1k/resolve/main/model.safetensors" \
  "${TINYVIT_DST}" || {
    echo "TinyViT direct Hugging Face download failed; trying huggingface_hub." >&2
    python - "${TINYVIT_DST}" <<'PY'
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

dst = Path(sys.argv[1])
path = hf_hub_download(
    repo_id="timm/tiny_vit_21m_512.dist_in22k_ft_in1k",
    filename="model.safetensors",
)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(path, dst)
print(dst)
PY
  } || {
    echo "TinyViT Hugging Face download failed." >&2
    echo "Run 'huggingface-cli login' with a read token, or manually mirror model.safetensors to:" >&2
    echo "  ${TINYVIT_DST}" >&2
    exit 1
  }

(
  cd "${OUT}"
  sha256sum sam2.1/*.pt tinyvit/*.safetensors > SHA256SUMS.txt
)

echo "Weights ready under ${OUT}"
echo "Checksums: ${OUT}/SHA256SUMS.txt"
