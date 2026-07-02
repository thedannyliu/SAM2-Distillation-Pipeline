#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
THIRD_PARTY="${THIRD_PARTY:-${ROOT}/third_party}"
SAM2_REF="${SAM2_REF:-main}"
EDGETAM_REF="${EDGETAM_REF:-main}"

mkdir -p "${THIRD_PARTY}"

clone_or_update() {
  local url="$1"
  local dst="$2"
  local ref="$3"
  if [[ ! -d "${dst}/.git" ]]; then
    git clone "${url}" "${dst}"
  fi
  git -C "${dst}" fetch --tags origin
  git -C "${dst}" checkout "${ref}"
}

clone_or_update https://github.com/facebookresearch/sam2.git "${THIRD_PARTY}/sam2" "${SAM2_REF}"
clone_or_update https://github.com/facebookresearch/EdgeTAM.git "${THIRD_PARTY}/EdgeTAM" "${EDGETAM_REF}"

cat > "${THIRD_PARTY}/UPSTREAM_REVISIONS.txt" <<EOF
sam2 $(git -C "${THIRD_PARTY}/sam2" rev-parse HEAD)
EdgeTAM $(git -C "${THIRD_PARTY}/EdgeTAM" rev-parse HEAD)
EOF

echo "Upstreams ready under ${THIRD_PARTY}"
cat "${THIRD_PARTY}/UPSTREAM_REVISIONS.txt"

