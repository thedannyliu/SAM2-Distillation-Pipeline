#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/danny-dataset}"
SAV_DATA_ROOT="${SAV_DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAV_SHARD_ROOT="${SAV_SHARD_ROOT:-${SAV_DATA_ROOT}/SA-V/sav_000}"
SAV_IMAGE_ROOT="${SAV_IMAGE_ROOT:-${SAV_SHARD_ROOT}/JPEGImages_24fps}"
SAV_ANN_ROOT="${SAV_ANN_ROOT:-${SAV_SHARD_ROOT}/annotations}"
SAV_VIDEO_ROOT="${SAV_VIDEO_ROOT:-${SAV_SHARD_ROOT}/videos}"
SAV_FILE_LIST="${SAV_FILE_LIST:-${SAV_SHARD_ROOT}/manifests/sav_train_filelist.txt}"
SAV_MAX_VIDEOS="${SAV_MAX_VIDEOS:-0}"
SAV_EXTRACT_FRAMES="${SAV_EXTRACT_FRAMES:-1}"
SAV_FRAME_SAMPLE_RATE="${SAV_FRAME_SAMPLE_RATE:-1}"
SAV_ANN_EVERY="${SAV_ANN_EVERY:-4}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
RUN_DIR="${RUN_DIR:-${SAM2D_ROOT}/runs/sav000_tinyvit_image_encoder_1h100}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FRAMES="${NUM_FRAMES:-8}"
MAX_OBJECTS="${MAX_OBJECTS:-3}"
RESOLUTION="${RESOLUTION:-1024}"
IMAGE_ENCODER_BATCH="${IMAGE_ENCODER_BATCH:-1}"
IMAGE_ENCODER_CKPT="${IMAGE_ENCODER_CKPT:-1}"
TARGET_STEPS="${TARGET_STEPS:-1000}"
SEED="${SEED:-250107256}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh prepare
  scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh warmup
  scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh finetune
  scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh estimate
  scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh all

Default flow:
  - Uses /group-volume/danny-dataset/SA-V/sav_000
  - Ensures JPEG frames and SA-V manual JSON annotations are available
  - Phase 1 trains image_encoder.neck only with TinyViT trunk frozen
  - Phase 2 resumes and trains the full image_encoder only
  - Memory, prompt, mask, and other video components remain frozen in both phases
EOF
}

find_existing_dir() {
  for path in "$@"; do
    if [[ -d "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

autodetect_layout() {
  if [[ ! -d "${SAV_IMAGE_ROOT}" ]]; then
    SAV_IMAGE_ROOT="$(find_existing_dir \
      "${SAV_SHARD_ROOT}/JPEGImages_24fps" \
      "${SAV_SHARD_ROOT}/train/JPEGImages_24fps" \
      "${SAV_SHARD_ROOT}/JPEGImages" \
      "${SAV_SHARD_ROOT}/frames" \
      2>/dev/null || echo "${SAV_IMAGE_ROOT}")"
  fi
  if [[ ! -d "${SAV_ANN_ROOT}" ]]; then
    SAV_ANN_ROOT="$(find_existing_dir \
      "${SAV_SHARD_ROOT}/annotations" \
      "${SAV_SHARD_ROOT}/train/annotations" \
      "${SAV_SHARD_ROOT}/Annotations" \
      2>/dev/null || echo "${SAV_ANN_ROOT}")"
  fi
  if [[ ! -d "${SAV_VIDEO_ROOT}" ]]; then
    SAV_VIDEO_ROOT="$(find_existing_dir \
      "${SAV_SHARD_ROOT}/videos" \
      "${SAV_SHARD_ROOT}/train/videos" \
      "${SAV_SHARD_ROOT}" \
      2>/dev/null || echo "${SAV_VIDEO_ROOT}")"
  fi
}

prepare_frames() {
  autodetect_layout
  if find "${SAV_IMAGE_ROOT}" -mindepth 2 -name '*.jpg' -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  if [[ "${SAV_EXTRACT_FRAMES}" -ne 1 ]]; then
    echo "Missing JPEG frames under ${SAV_IMAGE_ROOT}. Set SAV_EXTRACT_FRAMES=1 or provide SAV_IMAGE_ROOT." >&2
    exit 2
  fi
  echo "extract SA-V frames from ${SAV_VIDEO_ROOT} to ${SAV_IMAGE_ROOT}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  python "${REPO_ROOT}/tools/data/extract_sav_frames_local.py" \
    --video-root "${SAV_VIDEO_ROOT}" \
    --output-root "${SAV_IMAGE_ROOT}" \
    --sample-rate "${SAV_FRAME_SAMPLE_RATE}" \
    --max-videos "${SAV_MAX_VIDEOS}"
}

prepare_filelist() {
  autodetect_layout
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    cat <<EOF
DRY_RUN prepare:
  shard      ${SAV_SHARD_ROOT}
  image_root ${SAV_IMAGE_ROOT}
  ann_root   ${SAV_ANN_ROOT}
  video_root ${SAV_VIDEO_ROOT}
  file_list  ${SAV_FILE_LIST}
EOF
    return 0
  fi
  mkdir -p "$(dirname "${SAV_FILE_LIST}")"
  python - "${SAV_IMAGE_ROOT}" "${SAV_ANN_ROOT}" "${SAV_FILE_LIST}" "${SAV_MAX_VIDEOS}" <<'PY'
import sys
from pathlib import Path

image_root = Path(sys.argv[1])
ann_root = Path(sys.argv[2])
out = Path(sys.argv[3])
max_videos = int(sys.argv[4])
if not image_root.exists():
    raise SystemExit(f"missing image root: {image_root}")
if not ann_root.exists():
    raise SystemExit(f"missing annotation root: {ann_root}")
videos = []
for video_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
    if not any(video_dir.glob("*.jpg")):
        continue
    if not (ann_root / f"{video_dir.name}_manual.json").exists():
        continue
    videos.append(video_dir.name)
    if max_videos > 0 and len(videos) >= max_videos:
        break
if not videos:
    raise SystemExit(f"no videos with JPEG frames and *_manual.json under {image_root} / {ann_root}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(f"{video}\n" for video in videos), encoding="utf-8")
print(f"videos={len(videos)}")
print(f"file_list={out}")
PY
}

prepare_weights() {
  if [[ -s "${TINYVIT_CKPT}" ]]; then
    echo "exists ${TINYVIT_CKPT}"
    return 0
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN weights: scripts/company/01_download_weights.sh --out ${CHECKPOINT_ROOT}"
    return 0
  fi
  bash "${REPO_ROOT}/scripts/company/01_download_weights.sh" --out "${CHECKPOINT_ROOT}"
}

prepare() {
  prepare_frames
  prepare_filelist
  prepare_weights
}

run_phase() {
  local phase="$1"
  local max_epochs="$2"
  local trainable_mode="$3"
  local activation_args=()
  if [[ "${IMAGE_ENCODER_CKPT}" -eq 1 ]]; then
    activation_args+=(--image-encoder-activation-checkpoint)
  fi
  local cmd=(
    python "${REPO_ROOT}/tools/train/run_edgetam_trainer_smoke.py"
    --config "${REPO_ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml"
    --sam2-training-root "${SAM2_TRAINING_ROOT}"
    --edgetam-root "${EDGETAM_ROOT}"
    --out-dir "${RUN_DIR}"
    --max-epochs "${max_epochs}"
    --num-workers "${NUM_WORKERS}"
    --num-frames "${NUM_FRAMES}"
    --max-num-objects "${MAX_OBJECTS}"
    --batch-size "${BATCH_SIZE}"
    --resolution "${RESOLUTION}"
    --dataset-mode sav-json
    --sav-image-root "${SAV_IMAGE_ROOT}"
    --sav-ann-root "${SAV_ANN_ROOT}"
    --sav-file-list "${SAV_FILE_LIST}"
    --sav-ann-every "${SAV_ANN_EVERY}"
    --tinyvit-checkpoint "${TINYVIT_CKPT}"
    --image-encoder-forward-batch-size "${IMAGE_ENCODER_BATCH}"
    --trainable-module-mode "${trainable_mode}"
    --lambda-img 0
    --lambda-mem 0
    --seed "${SEED}"
    "${activation_args[@]}"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY_RUN %s command:\n' "${phase}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "${RUN_DIR}"
  local runtime_json="${RUN_DIR}/runtime_${phase}.json"
  local start end rc
  start="$(date +%s)"
  printf '{"phase": "%s", "start_unix": %s, "command": ' "${phase}" "${start}" > "${runtime_json}"
  python - "${cmd[@]}" >> "${runtime_json}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
  printf '}\n' >> "${runtime_json}"
  set +e
  "${cmd[@]}"
  rc=$?
  set -e
  end="$(date +%s)"
  python - "${runtime_json}" "${end}" "${rc}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data["end_unix"] = int(sys.argv[2])
data["elapsed_sec"] = data["end_unix"] - int(data["start_unix"])
data["return_code"] = int(sys.argv[3])
path.write_text(json.dumps(data, indent=2) + "\n")
PY
  cp "${RUN_DIR}/summary.json" "${RUN_DIR}/summary_${phase}.json"
  if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
  fi
}

warmup() {
  prepare
  run_phase "warmup" "${WARMUP_EPOCHS}" "image_neck_only"
}

finetune() {
  prepare
  local total_epochs=$((WARMUP_EPOCHS + FINETUNE_EPOCHS))
  run_phase "finetune" "${total_epochs}" "image_encoder_only"
}

estimate() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN estimate: would read ${RUN_DIR}/runtime_{warmup,finetune}.json"
    echo "DRY_RUN estimate: would write ${RUN_DIR}/runtime_estimate.json"
    return 0
  fi
  python - "${RUN_DIR}" "${TARGET_STEPS}" <<'PY'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
target_steps = int(sys.argv[2])
out = {}
for phase in ("warmup", "finetune"):
    runtime = run_dir / f"runtime_{phase}.json"
    summary = run_dir / f"summary_{phase}.json"
    if not runtime.exists() or not summary.exists():
        continue
    r = json.loads(runtime.read_text())
    s = json.loads(summary.read_text())
    before = s.get("checkpoint_before") or {"steps": {}}
    after = s.get("checkpoint_after") or {"steps": {}}
    before_steps = max([0, *[int(v) for v in before.get("steps", {}).values()]])
    after_steps = max([0, *[int(v) for v in after.get("steps", {}).values()]])
    observed_steps = max(1, after_steps - before_steps)
    sec_per_step = max(1, int(r["elapsed_sec"])) / observed_steps
    out[phase] = {
        "observed_steps": observed_steps,
        "elapsed_sec": int(r["elapsed_sec"]),
        "sec_per_step": sec_per_step,
        "steps_per_hour": 3600.0 / sec_per_step,
        "estimated_target_hours": target_steps * sec_per_step / 3600.0,
        "trainable_summary": s.get("trainable_summary_after"),
    }
(run_dir / "runtime_estimate.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  warmup)
    warmup
    ;;
  finetune)
    finetune
    ;;
  estimate)
    estimate
    ;;
  all)
    warmup
    finetune
    estimate
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
