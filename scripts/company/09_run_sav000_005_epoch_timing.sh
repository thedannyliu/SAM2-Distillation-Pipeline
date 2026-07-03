#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-${DATA_ROOT}/SA-V}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-5}"
COMBINED_ROOT="${COMBINED_ROOT:-${SAV_ROOT}/sav_000_005_epoch_timing}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAM2D_ROOT}/checkpoints}"
TINYVIT_CKPT="${TINYVIT_CKPT:-${CHECKPOINT_ROOT}/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"

RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sav000_005_epoch_timing}"
RUN_NAME="${RUN_NAME:-}"
GPUS="${GPUS:-}"
NPROC="${NPROC:-1}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FRAMES="${NUM_FRAMES:-8}"
MAX_OBJECTS="${MAX_OBJECTS:-3}"
RESOLUTION="${RESOLUTION:-1024}"
IMAGE_ENCODER_BATCH="${IMAGE_ENCODER_BATCH:-16}"
IMAGE_ENCODER_CKPT="${IMAGE_ENCODER_CKPT:-0}"
SAV_ANN_EVERY="${SAV_ANN_EVERY:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TARGET_STEPS="${TARGET_STEPS:-1000}"
BASELINE_SEC_PER_STEP="${BASELINE_SEC_PER_STEP:-2.35}"
SEED="${SEED:-250107256}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-5}"
EXTRACT_MISSING_FRAMES="${EXTRACT_MISSING_FRAMES:-0}"
SAV_FRAME_SAMPLE_RATE="${SAV_FRAME_SAMPLE_RATE:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/09_run_sav000_005_epoch_timing.sh prepare
  scripts/company/09_run_sav000_005_epoch_timing.sh 1gpu
  scripts/company/09_run_sav000_005_epoch_timing.sh 4gpu
  scripts/company/09_run_sav000_005_epoch_timing.sh summarize

Defaults:
  - Data: /group-volume/danny-dataset/SA-V/sav_000 ... sav_005
  - Aggressive timing: BATCH_SIZE=4, IMAGE_ENCODER_BATCH=16, IMAGE_ENCODER_CKPT=0
  - One epoch, full image_encoder trainable, non-image modules frozen
  - 1gpu uses GPUS=0, NPROC=1
  - 4gpu uses GPUS=0,1,2,3, NPROC=4
EOF
}

prepare() {
  local extract_args=()
  if [[ "${EXTRACT_MISSING_FRAMES}" -eq 1 ]]; then
    extract_args+=(--extract-missing-frames)
  fi
  python "${REPO_ROOT}/tools/data/prepare_sav_shard_range.py" \
    --sav-root "${SAV_ROOT}" \
    --start-shard "${START_SHARD}" \
    --end-shard "${END_SHARD}" \
    --out-root "${COMBINED_ROOT}" \
    --frame-sample-rate "${SAV_FRAME_SAMPLE_RATE}" \
    "${extract_args[@]}"
}

count_videos() {
  wc -l < "${COMBINED_ROOT}/manifests/sav_train_filelist.txt" | tr -d ' '
}

run_dir() {
  local mode="$1"
  if [[ -n "${RUN_NAME}" ]]; then
    echo "${RUN_ROOT}/${RUN_NAME}"
  else
    echo "${RUN_ROOT}/${mode}_b${BATCH_SIZE}_ieb${IMAGE_ENCODER_BATCH}_ckpt${IMAGE_ENCODER_CKPT}"
  fi
}

start_gpu_monitor() {
  local out_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skip GPU monitor" >&2
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

print_preflight() {
  local mode="$1"
  local videos="$2"
  local global_batch=$((BATCH_SIZE * NPROC))
  local steps=$(((videos + global_batch - 1) / global_batch))
  python - "${mode}" "${videos}" "${global_batch}" "${steps}" "${BASELINE_SEC_PER_STEP}" "${NPROC}" "${BATCH_SIZE}" "${NUM_FRAMES}" <<'PY'
import json
import sys
mode, videos, global_batch, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
baseline = float(sys.argv[5])
nproc, batch_size, num_frames = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
scale = max(1, nproc)
rough_sec_per_step = baseline
rough_hours = steps * rough_sec_per_step / 3600.0
print(json.dumps({
    "mode": mode,
    "videos": videos,
    "nproc": nproc,
    "per_gpu_batch_size": batch_size,
    "global_batch_size": global_batch,
    "num_frames": num_frames,
    "estimated_steps_per_epoch": steps,
    "rough_single_h100_sec_per_step_baseline": baseline,
    "rough_epoch_hours_before_multigpu_scaling": rough_hours,
    "note": "Actual runtime is measured below; this preflight ETA is only a baseline sanity check.",
}, indent=2))
PY
}

summarize_run() {
  local out_dir="$1"
  python - "${out_dir}" "${TARGET_STEPS}" "${NPROC}" "${BATCH_SIZE}" "${NUM_FRAMES}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
target_steps = int(sys.argv[2])
nproc = int(sys.argv[3])
batch_size = int(sys.argv[4])
num_frames = int(sys.argv[5])
runtime = json.loads((out_dir / "runtime_epoch.json").read_text())
summary = json.loads((out_dir / "summary.json").read_text())
before = summary.get("checkpoint_before") or {"steps": {}}
after = summary.get("checkpoint_after") or {"steps": {}}
before_steps = max([0, *[int(v) for v in before.get("steps", {}).values()]])
after_steps = max([0, *[int(v) for v in after.get("steps", {}).values()]])
observed_steps = max(1, after_steps - before_steps)
sec_per_step = max(1, int(runtime["elapsed_sec"])) / observed_steps
global_batch = nproc * batch_size

gpu = {}
csv_path = out_dir / "gpu_usage.csv"
if csv_path.exists():
    rows_by_gpu = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = str(row["index"]).strip()
            rows_by_gpu.setdefault(idx, []).append(row)
    for idx, rows in rows_by_gpu.items():
        util = [float(r["utilization.gpu"]) for r in rows if r.get("utilization.gpu", "").strip()]
        mem_used = [float(r["memory.used"]) for r in rows if r.get("memory.used", "").strip()]
        mem_total = [float(r["memory.total"]) for r in rows if r.get("memory.total", "").strip()]
        gpu[idx] = {
            "samples": len(rows),
            "avg_utilization_gpu_pct": sum(util) / len(util) if util else None,
            "max_memory_used_mb": max(mem_used) if mem_used else None,
            "memory_total_mb": max(mem_total) if mem_total else None,
        }

out = {
    "result": "pass",
    "out_dir": str(out_dir),
    "nproc": nproc,
    "per_gpu_batch_size": batch_size,
    "global_batch_size": global_batch,
    "num_frames": num_frames,
    "observed_steps": observed_steps,
    "elapsed_sec": int(runtime["elapsed_sec"]),
    "sec_per_step": sec_per_step,
    "steps_per_hour": 3600.0 / sec_per_step,
    "clips_per_hour": 3600.0 * global_batch / sec_per_step,
    "frames_per_hour": 3600.0 * global_batch * num_frames / sec_per_step,
    "estimated_target_hours": target_steps * sec_per_step / 3600.0,
    "trainable_summary": summary.get("trainable_summary_after"),
    "checkpoint_before": before,
    "checkpoint_after": after,
    "gpu_usage": gpu,
}
(out_dir / "epoch_timing_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
}

run_epoch() {
  local mode="$1"
  local out_dir
  out_dir="$(run_dir "${mode}")"
  prepare
  local videos
  videos="$(count_videos)"
  mkdir -p "${out_dir}"
  print_preflight "${mode}" "${videos}" | tee "${out_dir}/preflight.json"

  local activation_args=()
  if [[ "${IMAGE_ENCODER_CKPT}" -eq 1 ]]; then
    activation_args+=(--image-encoder-activation-checkpoint)
  fi
  local train_args=(
    --config "${REPO_ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml"
    --sam2-training-root "${SAM2_TRAINING_ROOT}"
    --edgetam-root "${EDGETAM_ROOT}"
    --out-dir "${out_dir}"
    --max-epochs "${MAX_EPOCHS}"
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
    --trainable-module-mode image_encoder_only
    --lambda-img 0
    --lambda-mem 0
    --seed "${SEED}"
    "${activation_args[@]}"
  )

  local launch_cmd=()
  if [[ "${NPROC}" -gt 1 ]]; then
    launch_cmd=(torchrun --standalone --nproc_per_node="${NPROC}" "${REPO_ROOT}/tools/train/run_edgetam_trainer_smoke.py" "${train_args[@]}")
  else
    launch_cmd=(python "${REPO_ROOT}/tools/train/run_edgetam_trainer_smoke.py" "${train_args[@]}")
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY_RUN command:\n' | tee "${out_dir}/command.txt"
    printf ' %q' CUDA_VISIBLE_DEVICES="${GPUS}" "${launch_cmd[@]}" | tee -a "${out_dir}/command.txt"
    printf '\n' | tee -a "${out_dir}/command.txt"
    return 0
  fi

  local monitor_pid=""
  monitor_pid="$(start_gpu_monitor "${out_dir}/gpu_usage.csv" || true)"
  local start end rc
  start="$(date +%s)"
  printf '{"phase": "epoch", "start_unix": %s, "command": ' "${start}" > "${out_dir}/runtime_epoch.json"
  python - "${launch_cmd[@]}" >> "${out_dir}/runtime_epoch.json" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
  printf '}\n' >> "${out_dir}/runtime_epoch.json"
  set +e
  CUDA_VISIBLE_DEVICES="${GPUS}" "${launch_cmd[@]}" 2>&1 | tee "${out_dir}/train.log"
  rc=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor "${monitor_pid}"
  end="$(date +%s)"
  python - "${out_dir}/runtime_epoch.json" "${end}" "${rc}" <<'PY'
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
  summarize_run "${out_dir}"
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  1gpu)
    GPUS="${GPUS:-0}"
    NPROC=1
    run_epoch "1gpu"
    ;;
  4gpu)
    GPUS="${GPUS:-0,1,2,3}"
    NPROC=4
    run_epoch "4gpu"
    ;;
  summarize)
    summarize_run "$(run_dir "${RUN_NAME:-1gpu}")"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
