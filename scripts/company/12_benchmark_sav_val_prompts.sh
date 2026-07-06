#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAV_VAL_ROOT="${SAV_VAL_ROOT:-${DATA_ROOT}/SA-V/sav_val}"
IMAGE_ROOT="${IMAGE_ROOT:-${SAV_VAL_ROOT}/JPEGImages_24fps}"
ANN_ROOT="${ANN_ROOT:-${SAV_VAL_ROOT}/Annotations_6fps}"

TINYVIT_RUN_NAME="${TINYVIT_RUN_NAME:-sav000_018_4gpu_tinyvit21m_b1_ieb8_ckpt0_w3_f15_wandb3}"
TINYVIT_RUN_DIR="${TINYVIT_RUN_DIR:-${SAM2D_ROOT}/runs/sav000_018_formal_image_encoder/${TINYVIT_RUN_NAME}}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${TINYVIT_RUN_DIR}/checkpoints/checkpoint_18.pt}"
TINYVIT_CONFIG="${TINYVIT_CONFIG:-${TINYVIT_RUN_DIR}/config_resolved.yaml}"

SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
SAM2L_CKPT="${SAM2L_CKPT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
SAM2L_CONFIG="${SAM2L_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

OUT_ROOT="${OUT_ROOT:-${SAM2D_ROOT}/runs/sav_val_prompt_benchmark/${TINYVIT_RUN_NAME}_vs_sam2p1l}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_OBJECTS="${MAX_OBJECTS:-2000}"
WARMUP_IMAGES="${WARMUP_IMAGES:-5}"
DEVICE="${DEVICE:-cuda}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/12_benchmark_sav_val_prompts.sh tinyvit
  scripts/company/12_benchmark_sav_val_prompts.sh sam2l
  scripts/company/12_benchmark_sav_val_prompts.sh all

Outputs:
  $OUT_ROOT/tinyvit21m/{box,point}/summary.json
  $OUT_ROOT/sam2p1l/{box,point}/summary.json
EOF
}

check_common_inputs() {
  local missing=0
  for path in "${IMAGE_ROOT}" "${ANN_ROOT}" "${SAM2_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
      echo "missing input: ${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
}

run_one() {
  local model_kind="$1"
  local name="$2"
  local checkpoint="$3"
  local config="$4"
  local prompt="$5"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ "${model_kind}" == "edgetam-trainer" && ! -f "${config}" ]]; then
    echo "missing TinyViT config: ${config}" >&2
    exit 1
  fi
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind "${model_kind}" \
    --prompt-kind "${prompt}" \
    --image-root "${IMAGE_ROOT}" \
    --ann-root "${ANN_ROOT}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --sam2-root "${SAM2_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-dir "${OUT_ROOT}/${name}/${prompt}" \
    --max-videos "${MAX_VIDEOS}" \
    --max-objects "${MAX_OBJECTS}" \
    --warmup-images "${WARMUP_IMAGES}" \
    --device "${DEVICE}"
}

run_tinyvit() {
  run_one edgetam-trainer tinyvit21m "${TINYVIT_CKPT}" "${TINYVIT_CONFIG}" box
  run_one edgetam-trainer tinyvit21m "${TINYVIT_CKPT}" "${TINYVIT_CONFIG}" point
}

run_sam2l() {
  run_one sam2 sam2p1l "${SAM2L_CKPT}" "${SAM2L_CONFIG}" box
  run_one sam2 sam2p1l "${SAM2L_CKPT}" "${SAM2L_CONFIG}" point
}

check_common_inputs
case "${1:-}" in
  tinyvit)
    run_tinyvit
    ;;
  sam2l)
    run_sam2l
    ;;
  all)
    run_tinyvit
    run_sam2l
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
