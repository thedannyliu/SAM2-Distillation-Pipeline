#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
RAW_SHARD_ROOT="${RAW_SHARD_ROOT:-/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030}"
PREP_ROOT="${PREP_ROOT:-${SAM2D_ROOT}/benchmarks/raw_sav030_prepared}"
OUT_ROOT="${OUT_ROOT:-${SAM2D_ROOT}/runs/raw_sav030_sam2p1l_benchmark}"

SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CKPT="${SAM2_CKPT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

MAX_VIDEOS="${MAX_VIDEOS:-2}"
MAX_OBJECTS_PER_VIDEO="${MAX_OBJECTS_PER_VIDEO:-2}"
MAX_IMAGE_OBJECTS="${MAX_IMAGE_OBJECTS:-200}"
SAVE_IMAGE_ARTIFACTS="${SAVE_IMAGE_ARTIFACTS:-10}"
VOS_OVERLAY_VIDEOS="${VOS_OVERLAY_VIDEOS:-2}"
VOS_OVERLAY_FRAMES="${VOS_OVERLAY_FRAMES:-240}"
VOS_TRACK_LATER="${VOS_TRACK_LATER:-1}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-4}"
DEVICE="${DEVICE:-cuda}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/14_benchmark_raw_sav_shard_sam2.sh prepare
  scripts/company/14_benchmark_raw_sav_shard_sam2.sh image
  scripts/company/14_benchmark_raw_sav_shard_sam2.sh vos
  scripts/company/14_benchmark_raw_sav_shard_sam2.sh artifacts
  scripts/company/14_benchmark_raw_sav_shard_sam2.sh all

Environment overrides:
  RAW_SHARD_ROOT=/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030
  PREP_ROOT=/group-volume/danny-dataset/sam2_distill/benchmarks/raw_sav030_prepared
  OUT_ROOT=/group-volume/danny-dataset/sam2_distill/runs/raw_sav030_sam2p1l_benchmark
  MAX_VIDEOS=2 MAX_OBJECTS_PER_VIDEO=2 MAX_IMAGE_OBJECTS=200
  VOS_TRACK_LATER=1 tracks objects whose first available GT mask is after frame 0.
EOF
}

check_inputs() {
  for path in "${RAW_SHARD_ROOT}" "${SAM2_ROOT}" "${SAM2_CKPT}"; do
    if [[ ! -e "${path}" ]]; then
      echo "missing input: ${path}" >&2
      exit 1
    fi
  done
}

prepare() {
  check_inputs
  python tools/data/prepare_sav_train_shard_benchmark.py \
    --shard-root "${RAW_SHARD_ROOT}" \
    --out-root "${PREP_ROOT}" \
    --max-videos "${MAX_VIDEOS}" \
    --max-objects-per-video "${MAX_OBJECTS_PER_VIDEO}"
}

run_image_one() {
  local prompt="$1"
  python tools/benchmark/benchmark_sav_prompt_masks.py \
    --model-kind sam2 \
    --prompt-kind "${prompt}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --ann-root "${PREP_ROOT}/Annotations_6fps" \
    --checkpoint "${SAM2_CKPT}" \
    --config "${SAM2_CONFIG}" \
    --sam2-root "${SAM2_ROOT}" \
    --out-dir "${OUT_ROOT}/image_${prompt}" \
    --max-objects "${MAX_IMAGE_OBJECTS}" \
    --save-artifacts "${SAVE_IMAGE_ARTIFACTS}" \
    --device "${DEVICE}"
}

image() {
  run_image_one box
  run_image_one point
}

vos() {
  local pred_root="${OUT_ROOT}/vos_pred"
  mkdir -p "${pred_root}"
  local start end elapsed
  local track_later_args=()
  if [[ "${VOS_TRACK_LATER}" == "1" ]]; then
    track_later_args+=(--track-object-appearing-later-in-video)
  fi
  start="$(date +%s)"
  python tools/eval/run_edgetam_vos_dataset.py \
    --edgetam-root "${SAM2_ROOT}" \
    --sam2-cfg "${SAM2_CONFIG}" \
    --checkpoint "${SAM2_CKPT}" \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --input-mask-root "${PREP_ROOT}/Annotations_6fps" \
    --out-dir "${pred_root}" \
    --video-list-file "${PREP_ROOT}/sav_train_benchmark.txt" \
    --per-obj-png-file \
    "${track_later_args[@]}" \
    --device "${DEVICE}"
  end="$(date +%s)"
  elapsed="$((end - start))"
  python tools/eval/run_sav_evaluator.py \
    --evaluator "${SAM2_ROOT}/sav_dataset/sav_evaluator.py" \
    --gt-root "${PREP_ROOT}/Annotations_6fps" \
    --pred-root "${pred_root}" \
    --out-json "${OUT_ROOT}/vos_eval_summary.json" \
    --num-processes "${NUM_EVAL_PROCESSES}" \
    --do-not-skip-first-and-last-frame
  python - "${OUT_ROOT}/vos_latency.json" "${elapsed}" "${pred_root}/summary.json" <<'PY'
import json, sys
out, elapsed, summary_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
summary = json.load(open(summary_path))
videos = summary.get("video_names", [])
payload = {
    "elapsed_sec": elapsed,
    "videos": len(videos),
    "sec_per_video": elapsed / max(len(videos), 1),
    "prediction_root": summary.get("prediction_root"),
}
open(out, "w").write(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
}

artifacts() {
  python tools/eval/make_vos_overlay_artifacts.py \
    --image-root "${PREP_ROOT}/JPEGImages_24fps" \
    --gt-root "${PREP_ROOT}/Annotations_6fps" \
    --pred-root "${OUT_ROOT}/vos_pred" \
    --video-list-file "${PREP_ROOT}/sav_train_benchmark.txt" \
    --out-dir "${OUT_ROOT}/vos_artifacts" \
    --max-videos "${VOS_OVERLAY_VIDEOS}" \
    --max-frames "${VOS_OVERLAY_FRAMES}"
}

case "${1:-}" in
  prepare) prepare ;;
  image) image ;;
  vos) vos ;;
  artifacts) artifacts ;;
  all)
    prepare
    image
    vos
    artifacts
    ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
