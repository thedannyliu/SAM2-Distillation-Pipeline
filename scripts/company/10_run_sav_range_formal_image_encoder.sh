#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-${DATA_ROOT}/SA-V}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-18}"
printf -v START_SHARD_PAD "%03d" "${START_SHARD}"
printf -v END_SHARD_PAD "%03d" "${END_SHARD}"
RANGE_NAME="${RANGE_NAME:-sav${START_SHARD_PAD}_${END_SHARD_PAD}}"
COMBINED_ROOT="${COMBINED_ROOT:-${SAV_ROOT}/${RANGE_NAME}_formal}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"

RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/${RANGE_NAME}_formal_image_encoder}"
RUN_NAME="${RUN_NAME:-}"
GPUS="${GPUS:-}"
NPROC="${NPROC:-1}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FRAMES="${NUM_FRAMES:-8}"
MAX_OBJECTS="${MAX_OBJECTS:-3}"
RESOLUTION="${RESOLUTION:-1024}"
IMAGE_ENCODER_BATCH="${IMAGE_ENCODER_BATCH:-8}"
IMAGE_ENCODER_CKPT="${IMAGE_ENCODER_CKPT:-0}"
FREEZE_BATCHNORM="${FREEZE_BATCHNORM:-0}"
SAV_ANN_EVERY="${SAV_ANN_EVERY:-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-15}"
CHECKPOINT_SAVE_FREQ="${CHECKPOINT_SAVE_FREQ:-1}"
SEED="${SEED:-250107256}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"
EXTRACT_MISSING_FRAMES="${EXTRACT_MISSING_FRAMES:-0}"
MOVE_FRAMES_TO_COMBINED="${MOVE_FRAMES_TO_COMBINED:-0}"
SAV_FRAME_SAMPLE_RATE="${SAV_FRAME_SAMPLE_RATE:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_NAME="${WANDB_NAME:-${RANGE_NAME}_formal_image_encoder}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_REQUIRE_LOGIN="${WANDB_REQUIRE_LOGIN:-1}"
WANDB_LIVE_LOGGER="${WANDB_LIVE_LOGGER:-1}"
WANDB_SAVE_SUMMARY_FILES="${WANDB_SAVE_SUMMARY_FILES:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/10_run_sav_range_formal_image_encoder.sh prepare
  scripts/company/10_run_sav_range_formal_image_encoder.sh 1gpu
  scripts/company/10_run_sav_range_formal_image_encoder.sh 4gpu
  scripts/company/10_run_sav_range_formal_image_encoder.sh 1gpu-finetune
  scripts/company/10_run_sav_range_formal_image_encoder.sh 4gpu-finetune

Formal flow:
  - Prepare SA-V shard range into a combined root.
  - Warmup trains image_encoder.neck only.
  - Finetune resumes and trains the full image_encoder only.
  - Prompt, mask, memory, and other non-image components stay frozen.

Defaults:
  START_SHARD=0 END_SHARD=18
  WARMUP_EPOCHS=3 FINETUNE_EPOCHS=15
  BATCH_SIZE=1 IMAGE_ENCODER_BATCH=8 IMAGE_ENCODER_CKPT=0
  FREEZE_BATCHNORM=1 keeps BatchNorm layers in eval mode and freezes BN affine params.
  CHECKPOINT_SAVE_FREQ=1 saves every checkpoint interval supported by SAM2 Trainer.
  TensorBoard writes under RUN_DIR/tensorboard.
  W&B companion logging uses WANDB_PROJECT, WANDB_NAME, and optional WANDB_RUN_ID.
EOF
}

prepare() {
  local args=()
  if [[ "${EXTRACT_MISSING_FRAMES}" -eq 1 ]]; then
    args+=(--extract-missing-frames)
  fi
  if [[ "${MOVE_FRAMES_TO_COMBINED}" -eq 1 ]]; then
    args+=(--move-frames-to-out-root)
  fi
  python "${REPO_ROOT}/tools/data/prepare_sav_shard_range.py" \
    --sav-root "${SAV_ROOT}" \
    --start-shard "${START_SHARD}" \
    --end-shard "${END_SHARD}" \
    --out-root "${COMBINED_ROOT}" \
    --frame-sample-rate "${SAV_FRAME_SAMPLE_RATE}" \
    "${args[@]}"
}

require_prepared() {
  local missing=0
  for path in \
    "${COMBINED_ROOT}/manifests/sav_train_filelist.txt" \
    "${COMBINED_ROOT}/JPEGImages_24fps" \
    "${COMBINED_ROOT}/annotations"
  do
    if [[ ! -e "${path}" ]]; then
      echo "missing prepared SA-V path: ${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    echo "Run scripts/company/10_run_sav_range_formal_image_encoder.sh prepare first." >&2
    exit 1
  fi
}

run_dir() {
  local mode="$1"
  if [[ -n "${RUN_NAME}" ]]; then
    echo "${RUN_ROOT}/${RUN_NAME}"
  else
    echo "${RUN_ROOT}/${mode}_b${BATCH_SIZE}_ieb${IMAGE_ENCODER_BATCH}_ckpt${IMAGE_ENCODER_CKPT}_w${WARMUP_EPOCHS}_f${FINETUNE_EPOCHS}"
  fi
}

count_videos() {
  wc -l < "${COMBINED_ROOT}/manifests/sav_train_filelist.txt" | tr -d ' '
}

start_gpu_monitor() {
  local out_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  (
    echo "timestamp,index,utilization.gpu,memory.used,memory.total,power.draw"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits || true
      sleep "${GPU_SAMPLE_INTERVAL}"
    done
  ) > "${out_csv}" &
  echo "$!"
}

stop_gpu_monitor() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

start_wandb_log_monitor() {
  local phase="$1"
  local out_dir="$2"
  local log_file="$3"
  if [[ "${NO_WANDB}" -eq 1 || "${WANDB_LIVE_LOGGER}" -ne 1 ]]; then
    return 0
  fi
  python "${REPO_ROOT}/tools/monitor/log_sam2_train_log_to_wandb.py" \
    --log-file "${log_file}" \
    --out-dir "${out_dir}" \
    --project "${WANDB_PROJECT}" \
    --name "${WANDB_NAME}" \
    --run-id "${WANDB_RUN_ID}" \
    --phase "${phase}" \
    > "${out_dir}/wandb_live_${phase}.log" 2>&1 &
  echo "$!"
}

print_preflight() {
  local mode="$1"
  local out_dir="$2"
  local videos global_batch steps_per_epoch total_epochs
  videos="$(count_videos)"
  global_batch=$((BATCH_SIZE * NPROC))
  steps_per_epoch=$(((videos + global_batch - 1) / global_batch))
  total_epochs=$((WARMUP_EPOCHS + FINETUNE_EPOCHS))
  python - "${mode}" "${videos}" "${NPROC}" "${BATCH_SIZE}" "${global_batch}" "${steps_per_epoch}" "${WARMUP_EPOCHS}" "${FINETUNE_EPOCHS}" "${NUM_FRAMES}" "${IMAGE_ENCODER_BATCH}" "${IMAGE_ENCODER_CKPT}" "${CHECKPOINT_SAVE_FREQ}" "${FREEZE_BATCHNORM}" <<'PY' | tee "${out_dir}/preflight.json"
import json
import sys

mode = sys.argv[1]
keys = [
    "videos", "nproc", "per_gpu_batch_size", "global_batch_size",
    "steps_per_epoch", "warmup_epochs", "finetune_epochs", "num_frames",
    "image_encoder_batch", "image_encoder_ckpt", "checkpoint_save_freq",
    "freeze_batchnorm",
]
vals = {key: int(value) for key, value in zip(keys, sys.argv[2:])}
vals["mode"] = mode
vals["warmup_steps_estimate"] = vals["steps_per_epoch"] * vals["warmup_epochs"]
vals["finetune_steps_estimate"] = vals["steps_per_epoch"] * vals["finetune_epochs"]
vals["total_steps_estimate"] = vals["steps_per_epoch"] * (vals["warmup_epochs"] + vals["finetune_epochs"])
vals["trainable_schedule"] = [
    {"phase": "warmup", "trainable": "image_encoder.neck"},
    {"phase": "finetune", "trainable": "image_encoder"},
]
vals["tensorboard_dir"] = None
print(json.dumps(vals, indent=2))
PY
}

write_run_metadata() {
  local out_dir="$1"
  python - "${out_dir}" "${CONFIG}" "${TINYVIT_CKPT}" "${WANDB_PROJECT}" "${WANDB_NAME}" "${WANDB_RUN_ID}" "${NO_WANDB}" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
metadata = {
    "out_dir": str(out_dir),
    "config": sys.argv[2],
    "tinyvit_checkpoint": sys.argv[3],
    "tensorboard_dir": str(out_dir / "tensorboard"),
    "checkpoint_dir": str(out_dir / "checkpoints"),
    "wandb_dir": str(out_dir / "wandb"),
    "wandb_project": sys.argv[4],
    "wandb_name": sys.argv[5],
    "wandb_run_id": sys.argv[6] or None,
    "wandb_disabled": sys.argv[7] == "1",
}
(out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
PY
}

check_wandb_ready() {
  if [[ "${NO_WANDB}" -eq 1 || "${DRY_RUN}" -eq 1 || "${WANDB_REQUIRE_LOGIN}" -eq 0 ]]; then
    return 0
  fi
  case "${WANDB_MODE:-online}" in
    offline|disabled|dryrun)
      return 0
      ;;
  esac
  python - <<'PY'
import netrc
import os
import sys

if os.environ.get("WANDB_API_KEY"):
    raise SystemExit(0)

try:
    auth = netrc.netrc().authenticators("api.wandb.ai")
except Exception:
    auth = None

if not auth or not auth[2]:
    print("W&B requested but no API key was found in WANDB_API_KEY or ~/.netrc.", file=sys.stderr)
    print("Run `wandb login --relogin`, or set WANDB_MODE=offline to sync later, or set NO_WANDB=1.", file=sys.stderr)
    raise SystemExit(1)
PY
}

ensure_wandb_run_id() {
  local out_dir="${1:-}"
  if [[ "${NO_WANDB}" -eq 1 || -n "${WANDB_RUN_ID}" ]]; then
    return 0
  fi
  WANDB_RUN_ID="$(python - "${out_dir}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1]) if sys.argv[1] else None
if out_dir is not None:
    for name in ("wandb_run.json", "run_metadata.json"):
        path = out_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        run_id = data.get("run_id") or data.get("wandb_run_id")
        if run_id:
            print(run_id)
            raise SystemExit(0)

try:
    import wandb
    print(wandb.util.generate_id())
except Exception:
    import secrets
    print(secrets.token_hex(8))
PY
)"
  export WANDB_RUN_ID
}

phase_command() {
  local out_dir="$1"
  local max_epochs="$2"
  local trainable_mode="$3"
  local activation_args=()
  if [[ "${IMAGE_ENCODER_CKPT}" -eq 1 ]]; then
    activation_args+=(--image-encoder-activation-checkpoint)
  fi
  local train_args=(
    --config "${CONFIG}"
    --sam2-training-root "${SAM2_TRAINING_ROOT}"
    --edgetam-root "${EDGETAM_ROOT}"
    --out-dir "${out_dir}"
    --max-epochs "${max_epochs}"
    --num-workers "${NUM_WORKERS}"
    --num-frames "${NUM_FRAMES}"
    --max-num-objects "${MAX_OBJECTS}"
    --batch-size "${BATCH_SIZE}"
    --resolution "${RESOLUTION}"
    --dataset-mode sav-json
    --sav-image-root "${COMBINED_ROOT}/JPEGImages_24fps"
    --sav-ann-root "${COMBINED_ROOT}/annotations"
    --sav-file-list "${COMBINED_ROOT}/manifests/sav_train_filelist.txt"
    --sav-ann-every "${SAV_ANN_EVERY}"
    --tinyvit-checkpoint "${TINYVIT_CKPT}"
    --image-encoder-forward-batch-size "${IMAGE_ENCODER_BATCH}"
    --trainable-module-mode "${trainable_mode}"
    --lambda-img 0
    --lambda-mem 0
    --checkpoint-save-freq "${CHECKPOINT_SAVE_FREQ}"
    --seed "${SEED}"
    "${activation_args[@]}"
  )
  if [[ "${FREEZE_BATCHNORM}" -eq 1 ]]; then
    train_args+=(--freeze-batchnorm)
  fi
  if [[ "${NO_WANDB}" -eq 1 || "${WANDB_LIVE_LOGGER}" -eq 1 ]]; then
    train_args+=(--no-wandb)
  else
    train_args+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-name "${WANDB_NAME}"
      --wandb-run-id "${WANDB_RUN_ID}"
      --wandb-phase "${trainable_mode}"
    )
  fi
  if [[ "${NPROC}" -gt 1 ]]; then
    printf '%q ' torchrun --standalone --nproc_per_node="${NPROC}" "${REPO_ROOT}/tools/train/run_edgetam_trainer_smoke.py" "${train_args[@]}"
  else
    printf '%q ' python "${REPO_ROOT}/tools/train/run_edgetam_trainer_smoke.py" "${train_args[@]}"
  fi
}

run_phase() {
  local phase="$1"
  local out_dir="$2"
  local max_epochs="$3"
  local trainable_mode="$4"
  local cmd_str runtime_json monitor_pid wandb_monitor_pid train_log start end rc
  cmd_str="$(phase_command "${out_dir}" "${max_epochs}" "${trainable_mode}")"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY_RUN %s command:\nCUDA_VISIBLE_DEVICES=%q %s\n' "${phase}" "${GPUS}" "${cmd_str}" | tee -a "${out_dir}/commands.txt"
    return 0
  fi
  runtime_json="${out_dir}/runtime_${phase}.json"
  train_log="${out_dir}/train_${phase}.log"
  : > "${train_log}"
  monitor_pid="$(start_gpu_monitor "${out_dir}/gpu_usage_${phase}.csv" || true)"
  wandb_monitor_pid="$(start_wandb_log_monitor "${phase}" "${out_dir}" "${train_log}" || true)"
  start="$(date +%s)"
  python - "${phase}" "${start}" "${cmd_str}" > "${runtime_json}" <<'PY'
import json
import sys
print(json.dumps({"phase": sys.argv[1], "start_unix": int(sys.argv[2]), "command": sys.argv[3]}, indent=2))
PY
  set +e
  CUDA_VISIBLE_DEVICES="${GPUS}" bash -lc "${cmd_str}" 2>&1 | tee -a "${train_log}"
  rc=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor "${monitor_pid}"
  stop_gpu_monitor "${wandb_monitor_pid}"
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
  if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
  fi
  cp "${out_dir}/summary.json" "${out_dir}/summary_${phase}.json"
}

summarize_formal() {
  local out_dir="$1"
  python - "${out_dir}" "${NPROC}" "${BATCH_SIZE}" "${NUM_FRAMES}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
nproc = int(sys.argv[2])
batch_size = int(sys.argv[3])
num_frames = int(sys.argv[4])
preflight = json.loads((out_dir / "preflight.json").read_text())
phases = {}
for phase in ("warmup", "finetune"):
    runtime_path = out_dir / f"runtime_{phase}.json"
    summary_path = out_dir / f"summary_{phase}.json"
    if not runtime_path.exists() or not summary_path.exists():
        continue
    runtime = json.loads(runtime_path.read_text())
    summary = json.loads(summary_path.read_text())
    before = summary.get("checkpoint_before") or {"steps": {}}
    after = summary.get("checkpoint_after") or {"steps": {}}
    before_steps = max([0, *[int(v) for v in before.get("steps", {}).values()]])
    after_steps = max([0, *[int(v) for v in after.get("steps", {}).values()]])
    observed_steps = max(1, after_steps - before_steps)
    sec_per_step = max(1, int(runtime["elapsed_sec"])) / observed_steps
    phases[phase] = {
        "observed_steps": observed_steps,
        "elapsed_sec": int(runtime["elapsed_sec"]),
        "sec_per_step": sec_per_step,
        "steps_per_hour": 3600.0 / sec_per_step,
        "clips_per_hour": 3600.0 * nproc * batch_size / sec_per_step,
        "frames_per_hour": 3600.0 * nproc * batch_size * num_frames / sec_per_step,
        "trainable_summary": summary.get("trainable_summary_after"),
        "checkpoint_before": before,
        "checkpoint_after": after,
    }
out = {
    "result": "pass",
    "out_dir": str(out_dir),
    "preflight": preflight,
    "phases": phases,
}
(out_dir / "formal_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
}

log_wandb_summary() {
  local out_dir="$1"
  if [[ "${NO_WANDB}" -eq 1 || "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  python - "${out_dir}" "${WANDB_PROJECT}" "${WANDB_NAME}" "${WANDB_RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
project = sys.argv[2]
name = sys.argv[3]
run_id = sys.argv[4] or None

try:
    import wandb
except ImportError as exc:
    raise SystemExit("wandb is not installed. Run setup or set NO_WANDB=1.") from exc

summary = json.loads((out_dir / "formal_summary.json").read_text())
metadata = json.loads((out_dir / "run_metadata.json").read_text())
run = wandb.init(
    project=project,
    name=name,
    id=run_id,
    resume="allow" if run_id else None,
    dir=str(out_dir / "wandb"),
    config={
        "preflight": summary.get("preflight", {}),
        "tensorboard_dir": metadata.get("tensorboard_dir"),
        "checkpoint_dir": metadata.get("checkpoint_dir"),
        "wandb_dir": metadata.get("wandb_dir"),
    },
)
(out_dir / "wandb_run.json").write_text(json.dumps({"run_id": run.id, "project": project, "name": name}) + "\n")
metrics = {}
for phase, values in summary.get("phases", {}).items():
    for key in ("observed_steps", "elapsed_sec", "sec_per_step", "steps_per_hour", "clips_per_hour", "frames_per_hour"):
        if key in values:
            metrics[f"{phase}/{key}"] = values[key]
wandb.log(metrics)
if os.environ.get("WANDB_SAVE_SUMMARY_FILES", "0") == "1":
    for file_name in ("formal_summary.json", "preflight.json", "run_metadata.json"):
        path = out_dir / file_name
        if path.exists():
            wandb.save(str(path), base_path=str(out_dir))
run.finish()
PY
}

run_formal() {
  local mode="$1"
  local out_dir total_epochs
  if [[ -z "${WANDB_PROJECT}" ]]; then
    WANDB_PROJECT="sam2-distill-edgetam-formal-${mode}"
  fi
  check_wandb_ready
  out_dir="$(run_dir "${mode}")"
  ensure_wandb_run_id "${out_dir}"
  export WANDB_DIR="${out_dir}/wandb"
  prepare
  mkdir -p "${out_dir}"
  print_preflight "${mode}" "${out_dir}"
  write_run_metadata "${out_dir}"
  total_epochs=$((WARMUP_EPOCHS + FINETUNE_EPOCHS))
  run_phase "warmup" "${out_dir}" "${WARMUP_EPOCHS}" "image_neck_only"
  run_phase "finetune" "${out_dir}" "${total_epochs}" "image_encoder_only"
  if [[ "${DRY_RUN}" -ne 1 ]]; then
    summarize_formal "${out_dir}"
    log_wandb_summary "${out_dir}"
  fi
}

run_formal_finetune_only() {
  local mode="$1"
  local out_dir total_epochs checkpoint_path
  if [[ -z "${WANDB_PROJECT}" ]]; then
    WANDB_PROJECT="sam2-distill-edgetam-formal-${mode}"
  fi
  check_wandb_ready
  out_dir="$(run_dir "${mode}")"
  checkpoint_path="${out_dir}/checkpoints/checkpoint.pt"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "missing warmup checkpoint for finetune resume: ${checkpoint_path}" >&2
    exit 1
  fi
  ensure_wandb_run_id "${out_dir}"
  export WANDB_DIR="${out_dir}/wandb"
  require_prepared
  mkdir -p "${out_dir}"
  print_preflight "${mode}" "${out_dir}"
  write_run_metadata "${out_dir}"
  total_epochs=$((WARMUP_EPOCHS + FINETUNE_EPOCHS))
  run_phase "finetune" "${out_dir}" "${total_epochs}" "image_encoder_only"
  if [[ "${DRY_RUN}" -ne 1 ]]; then
    summarize_formal "${out_dir}"
    log_wandb_summary "${out_dir}"
  fi
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  1gpu)
    GPUS="${GPUS:-0}"
    NPROC=1
    run_formal "1gpu"
    ;;
  4gpu)
    GPUS="${GPUS:-0,1,2,3}"
    NPROC=4
    run_formal "4gpu"
    ;;
  1gpu-finetune)
    GPUS="${GPUS:-0}"
    NPROC=1
    run_formal_finetune_only "1gpu"
    ;;
  4gpu-finetune)
    GPUS="${GPUS:-0,1,2,3}"
    NPROC=4
    run_formal_finetune_only "4gpu"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
