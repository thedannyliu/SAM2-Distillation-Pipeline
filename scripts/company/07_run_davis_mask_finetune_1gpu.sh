#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
DAVIS_ROOT="${DAVIS_ROOT:-${DATA_ROOT}/DAVIS/2017}"
DAVIS_URL="${DAVIS_URL:-https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip}"
DAVIS_REFERER="${DAVIS_REFERER:-https://davischallenge.org/davis2017/code.html}"
DAVIS_USER_AGENT="${DAVIS_USER_AGENT:-Mozilla/5.0}"
DAVIS_ZIP="${DAVIS_ZIP:-${DAVIS_ROOT}/raw/DAVIS-2017-trainval-480p.zip}"
DAVIS_MAX_FRAMES="${DAVIS_MAX_FRAMES:-500}"
DAVIS_SUBSET_ROOT="${DAVIS_SUBSET_ROOT:-${DAVIS_ROOT}/trainval_480p_subset_${DAVIS_MAX_FRAMES}}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
RUN_DIR="${RUN_DIR:-${SAM2D_ROOT}/runs/davis_tinyvit_mask_finetune_1gpu}"

SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"

MAX_EPOCHS="${MAX_EPOCHS:-1}"
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
  scripts/company/07_run_davis_mask_finetune_1gpu.sh prepare
  scripts/company/07_run_davis_mask_finetune_1gpu.sh train
  scripts/company/07_run_davis_mask_finetune_1gpu.sh estimate
  scripts/company/07_run_davis_mask_finetune_1gpu.sh all

Purpose:
  Fast company-side fallback when SA-V is slow or blocked. Downloads DAVIS 2017
  trainval 480p, extracts a <=500-frame subset, and runs one-GPU EdgeTAM
  TinyViT-21M mask finetune through the upstream SAM2 Trainer path.

Important defaults:
  DATA_ROOT=/group-volume/danny-dataset
  DAVIS_ROOT=$DATA_ROOT/DAVIS/2017
  SAM2D_ROOT=$DATA_ROOT/sam2_distill
  TINYVIT_CKPT=$SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
  RUN_DIR=$SAM2D_ROOT/runs/davis_tinyvit_mask_finetune_1gpu
  SAM2_TRAINING_ROOT=/user-volume/repo/facebookresearch-sam2
  EDGETAM_ROOT=/user-volume/repo/EdgeTAM

Useful overrides:
  DAVIS_MAX_FRAMES=500
  DAVIS_URL=https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
  DAVIS_REFERER=https://davischallenge.org/davis2017/code.html
  MAX_EPOCHS=1
  BATCH_SIZE=1
  NUM_FRAMES=8
  MAX_OBJECTS=3
  RESOLUTION=1024
  IMAGE_ENCODER_BATCH=1
  IMAGE_ENCODER_CKPT=1
  TARGET_STEPS=1000
  DRY_RUN=1
EOF
}

download_file() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ -s "${dst}" ]]; then
    echo "exists ${dst}"
    return 0
  fi
  echo "download ${dst}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c --auto-file-renaming=false --allow-overwrite=true \
      --user-agent="${DAVIS_USER_AGENT}" \
      --referer="${DAVIS_REFERER}" \
      -d "$(dirname "${dst}")" -o "$(basename "${dst}")" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c --user-agent="${DAVIS_USER_AGENT}" --referer="${DAVIS_REFERER}" -O "${dst}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --continue-at - \
      -A "${DAVIS_USER_AGENT}" \
      -e "${DAVIS_REFERER}" \
      -o "${dst}" "${url}"
  else
    python - "${url}" "${dst}" "${DAVIS_USER_AGENT}" "${DAVIS_REFERER}" <<'PY'
import sys
import urllib.request

url, dst, user_agent, referer = sys.argv[1:5]
request = urllib.request.Request(
    url,
    headers={
        "User-Agent": user_agent,
        "Referer": referer,
    },
)
with urllib.request.urlopen(request, timeout=120) as response, open(dst, "wb") as out:
    out.write(response.read())
PY
  fi
}

prepare() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    cat <<EOF
DRY_RUN prepare:
  download ${DAVIS_URL}
  zip      ${DAVIS_ZIP}
  subset   ${DAVIS_SUBSET_ROOT}
EOF
    return 0
  fi

  if [[ ! -f "${DAVIS_SUBSET_ROOT}/val.txt" ]]; then
    download_file "${DAVIS_URL}" "${DAVIS_ZIP}"
    python "${REPO_ROOT}/tools/data/extract_davis_zip_smoke_subset.py" \
      --zip "${DAVIS_ZIP}" \
      --out-root "${DAVIS_SUBSET_ROOT}" \
      --max-frames "${DAVIS_MAX_FRAMES}"
  else
    echo "exists ${DAVIS_SUBSET_ROOT}/val.txt"
  fi

  if [[ "${KEEP_ARCHIVES}" -eq 0 && -f "${DAVIS_ZIP}" ]]; then
    rm -f "${DAVIS_ZIP}"
    rmdir "$(dirname "${DAVIS_ZIP}")" 2>/dev/null || true
  fi
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

train() {
  prepare
  prepare_weights

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
    --max-epochs "${MAX_EPOCHS}"
    --num-workers "${NUM_WORKERS}"
    --num-frames "${NUM_FRAMES}"
    --max-num-objects "${MAX_OBJECTS}"
    --batch-size "${BATCH_SIZE}"
    --resolution "${RESOLUTION}"
    --dataset-mode vos
    --vos-image-root "${DAVIS_SUBSET_ROOT}/JPEGImages"
    --vos-gt-root "${DAVIS_SUBSET_ROOT}/Annotations"
    --vos-file-list "${DAVIS_SUBSET_ROOT}/val.txt"
    --tinyvit-checkpoint "${TINYVIT_CKPT}"
    --image-encoder-forward-batch-size "${IMAGE_ENCODER_BATCH}"
    --lambda-img 0
    --lambda-mem 0
    --seed "${SEED}"
    "${activation_args[@]}"
  )

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY_RUN train command:\n'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "${RUN_DIR}"

  local start end rc
  start="$(date +%s)"
  printf '{"start_unix": %s, "command": ' "${start}" > "${RUN_DIR}/runtime.json"
  python - "${cmd[@]}" >> "${RUN_DIR}/runtime.json" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
  printf '}\n' >> "${RUN_DIR}/runtime.json"

  set +e
  "${cmd[@]}"
  rc=$?
  set -e
  end="$(date +%s)"
  python - "${RUN_DIR}/runtime.json" "${end}" "${rc}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
end = int(sys.argv[2])
rc = int(sys.argv[3])
data = json.loads(path.read_text())
data["end_unix"] = end
data["elapsed_sec"] = end - int(data["start_unix"])
data["return_code"] = rc
path.write_text(json.dumps(data, indent=2) + "\n")
PY
  if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
  fi
  estimate
}

estimate() {
  python - "${RUN_DIR}" "${DAVIS_SUBSET_ROOT}/val.txt" "${TARGET_STEPS}" "${BATCH_SIZE}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
file_list = Path(sys.argv[2])
target_steps = int(sys.argv[3])
batch_size = int(sys.argv[4])
summary_path = run_dir / "summary.json"
runtime_path = run_dir / "runtime.json"
if not summary_path.exists() or not runtime_path.exists():
    raise SystemExit(f"Missing {summary_path} or {runtime_path}; run train first.")

summary = json.loads(summary_path.read_text())
runtime = json.loads(runtime_path.read_text())
before = summary.get("checkpoint_before") or {"steps": {}}
after = summary.get("checkpoint_after") or {"steps": {}}
before_steps = max([0, *[int(v) for v in before.get("steps", {}).values()]])
after_steps = max([0, *[int(v) for v in after.get("steps", {}).values()]])
observed_steps = max(1, after_steps - before_steps)
elapsed = max(1, int(runtime["elapsed_sec"]))
sec_per_step = elapsed / observed_steps
videos = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
steps_per_epoch_hint = max(1, math.ceil(len(videos) / max(1, batch_size)))
estimate = {
    "run_dir": str(run_dir),
    "observed_steps": observed_steps,
    "elapsed_sec": elapsed,
    "sec_per_step": sec_per_step,
    "steps_per_hour": 3600.0 / sec_per_step,
    "target_steps": target_steps,
    "estimated_target_hours": target_steps * sec_per_step / 3600.0,
    "video_count": len(videos),
    "batch_size": batch_size,
    "steps_per_epoch_hint": steps_per_epoch_hint,
    "estimated_epoch_hours_hint": steps_per_epoch_hint * sec_per_step / 3600.0,
}
(run_dir / "runtime_estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
print(json.dumps(estimate, indent=2))
PY
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  train)
    train
    ;;
  estimate)
    estimate
    ;;
  all)
    train
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
