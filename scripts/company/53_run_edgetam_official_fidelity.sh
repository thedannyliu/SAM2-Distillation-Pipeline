#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-all}"
case "${ACTION}" in
  describe|gate|all) ;;
  *)
    echo "Usage: $0 {describe|gate|all}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "[ERROR] Official EdgeTAM fidelity requires exactly four GPUs" >&2
  return 2 2>/dev/null || exit 2
fi

SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
if [[ -z "${SAV_ROOT:-}" ]]; then
  for candidate in \
    /group-volume/danny-dataset/SA-V \
    /mnt/data/danny-dataset/SA-V \
    /danny-dataset/SA-V; do
    if [[ -f "${candidate}/sav_val/sav_val.txt" && \
          -f "${candidate}/sav_test/sav_test.txt" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
EDGETAM_REQUIRED_COMMIT="${EDGETAM_REQUIRED_COMMIT:-7711e012a30a2402c4eaab637bdb00a521302c91}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/edgetam/edgetam.pt}"
if [[ -z "${EDGETAM_CONFIG:-}" ]]; then
  for candidate in \
    "${EDGETAM_ROOT}/sam2/configs/edgetam.yaml" \
    "${EDGETAM_ROOT}/configs/edgetam.yaml"; do
    if [[ -f "${candidate}" ]]; then
      EDGETAM_CONFIG="${candidate}"
      break
    fi
  done
fi
EDGETAM_CONFIG="${EDGETAM_CONFIG:-${EDGETAM_ROOT}/sam2/configs/edgetam.yaml}"
EXPERIMENT_NAME="${EDGETAM_FIDELITY_EXPERIMENT:-E0_official_upstream}"
GATE_SEED="${EDGETAM_FIDELITY_GATE_SEED:-edgetam-memory-gate-v2}"
RUN_DIR="${RUN_DIR:-${SAM2D_ROOT}/runs/edgetam_fidelity_v3/${EXPERIMENT_NAME}}"
GATE_COUNT="${EDGETAM_FIDELITY_GATE_VIDEOS:-32}"
GATE_MIN_JF="${EDGETAM_FIDELITY_MIN_JF:-55}"
WANDB_PROJECT="${WANDB_PROJECT:-edgetam-fidelity-v3}"
WANDB_MODE="${WANDB_MODE:-online}"
SKIP_DONE="${SKIP_DONE:-1}"

describe() {
  echo "Experiment: ${EXPERIMENT_NAME}"
  echo "Purpose: validate the unmodified released EdgeTAM model and evaluator"
  echo "Checkpoint: ${EDGETAM_CHECKPOINT}"
  echo "Config: ${EDGETAM_CONFIG}"
  echo "Gate: fixed ${GATE_COUNT}-video SA-V val; seed ${GATE_SEED}; J&F >= ${GATE_MIN_JF}"
  echo "Passing path: gate -> full SA-V val -> full SA-V test -> W&B"
  echo "Run dir: ${RUN_DIR}"
}

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

ensure_checkpoint() {
  local lock_file="${EDGETAM_CHECKPOINT}.download.lock"
  local status=0
  mkdir -p "$(dirname "${EDGETAM_CHECKPOINT}")" || return 1
  exec 8>"${lock_file}" || return 1
  flock 8 || return 1
  if [[ -f "${EDGETAM_CHECKPOINT}" ]]; then
    flock -u 8
    return 0
  fi
  OUT="${EDGETAM_CHECKPOINT}" EDGETAM_ROOT="${EDGETAM_ROOT}" \
    scripts/company/17_download_edgetam_checkpoint.sh
  status=$?
  flock -u 8
  return "${status}"
}

validate_inputs() {
  local path
  for path in \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt" \
    "${SAM2_ROOT}/sav_dataset/sav_evaluator.py" \
    "${EDGETAM_ROOT}/sam2/modeling/perceiver.py" \
    "${EDGETAM_CONFIG}" \
    "${EDGETAM_CHECKPOINT}"; do
    require_path "${path}" || return 1
  done
  git -C "${EDGETAM_ROOT}" merge-base --is-ancestor \
    "${EDGETAM_REQUIRED_COMMIT}" HEAD || {
      echo "[ERROR] EdgeTAM checkout must contain ${EDGETAM_REQUIRED_COMMIT}" >&2
      return 1
    }
}

ensure_wandb_run() {
  [[ "${WANDB_MODE}" == "online" ]] || return 0
  WANDB_PROJECT="${WANDB_PROJECT}" \
  RUN_DIR="${RUN_DIR}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  GATE_SEED="${GATE_SEED}" \
  GATE_COUNT="${GATE_COUNT}" \
    python - <<'PY'
import json
import os
from pathlib import Path

import wandb

run_dir = Path(os.environ["RUN_DIR"])
experiment = os.environ["EXPERIMENT_NAME"]
run_file = run_dir / "wandb/wandb_run.json"
run_file.parent.mkdir(parents=True, exist_ok=True)
run_id = None
if run_file.is_file():
    run_id = json.loads(run_file.read_text(encoding="utf-8"))["run_id"]
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    name=experiment,
    id=run_id,
    resume="must" if run_id else None,
    dir=str(run_file.parent),
    config={
        "experiment": experiment,
        "model": "official EdgeTAM",
        "training": False,
        "selection_split": "sav_val",
        "gate_seed": os.environ["GATE_SEED"],
        "gate_videos": int(os.environ["GATE_COUNT"]),
    },
)
run_file.write_text(
    json.dumps(
        {
            "run_id": run.id,
            "url": run.url,
            "entity": run.entity,
            "project": os.environ["WANDB_PROJECT"],
            "name": experiment,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
run.finish()
print(f"W&B evaluation run: {run.url}", flush=True)
PY
}

benchmark() {
  local split="$1" benchmark_root="$2" video_list="$3"
  local aggregate_csv="${RUN_DIR}/all_metrics.csv"
  MODEL_FAMILY=edgetam \
  STAGE1_CHECKPOINT="${EDGETAM_CHECKPOINT}" \
  EDGETAM_ROOT="${EDGETAM_ROOT}" \
  EDGETAM_CONFIG="${EDGETAM_CONFIG}" \
  SAM2_ROOT="${SAM2_ROOT}" \
  EXPERIMENT="${EXPERIMENT_NAME}" \
  RUN_DIR="${RUN_DIR}" \
  BENCH_ROOT="${benchmark_root}" \
  AGGREGATE_CSV="${aggregate_csv}" \
  SAV_ROOT="${SAV_ROOT}" \
  SAV_SPLIT="${split}" \
  VIDEO_LIST_FILE="${video_list}" \
  MAX_VIDEOS=0 \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  SKIP_DONE="${SKIP_DONE}" \
  CLEAN_PREDICTIONS=1 \
    scripts/company/25_benchmark_stage1_sav_test.sh
}

check_gate() {
  python - \
    "${RUN_DIR}/sav_val_gate${GATE_COUNT}_box_benchmark/metrics.csv" \
    "${GATE_MIN_JF}" \
    "${RUN_DIR}/gate_status.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
minimum = float(sys.argv[2])
out_path = Path(sys.argv[3])
with metrics_path.open(encoding="utf-8", newline="") as handle:
    rows = {row["mode"]: row for row in csv.DictReader(handle)}
jf = float(rows["video_tracking"]["J&F"])
payload = {
    "status": "pass" if jf >= minimum else "fail",
    "J&F": jf,
    "minimum_J&F": minimum,
    "metrics_path": str(metrics_path),
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if payload["status"] == "pass" else 1)
PY
}

log_metrics() {
  [[ "${WANDB_MODE}" == "online" ]] || return 0
  local -a args=()
  if [[ -f "${RUN_DIR}/sav_val_gate${GATE_COUNT}_box_benchmark/metrics.csv" ]]; then
    args+=(
      --metrics
      "sav_val_gate${GATE_COUNT}=${RUN_DIR}/sav_val_gate${GATE_COUNT}_box_benchmark/metrics.csv"
    )
  fi
  if [[ -f "${RUN_DIR}/sav_val_box_benchmark/metrics.csv" ]]; then
    args+=(
      --metrics
      "sav_val=${RUN_DIR}/sav_val_box_benchmark/metrics.csv"
    )
  fi
  if [[ -f "${RUN_DIR}/sav_test_box_benchmark/metrics.csv" ]]; then
    args+=(
      --metrics
      "sav_test=${RUN_DIR}/sav_test_box_benchmark/metrics.csv"
    )
  fi
  env -u WANDB_RUN_ID python tools/train/log_task_eval_to_wandb.py \
    --run-file "${RUN_DIR}/wandb/wandb_run.json" \
    "${args[@]}"
}

if [[ "${ACTION}" == "describe" ]]; then
  describe
  return 0 2>/dev/null || exit 0
fi

describe
ensure_checkpoint || return 1 2>/dev/null || exit 1
validate_inputs || return 1 2>/dev/null || exit 1
mkdir -p "${RUN_DIR}/checkpoints"
ln -sfn "${EDGETAM_CHECKPOINT}" "${RUN_DIR}/checkpoints/last.pt"
ln -sfn last.pt "${RUN_DIR}/checkpoints/best.pt"
ln -sfn best.pt "${RUN_DIR}/checkpoints/checkpoint.pt"
cp "${EDGETAM_CONFIG}" "${RUN_DIR}/resolved_config.yaml" || \
  return 1 2>/dev/null || exit 1
ensure_wandb_run || return 1 2>/dev/null || exit 1

python tools/experiments/sample_video_gate.py \
  --input "${SAV_ROOT}/sav_val/sav_val.txt" \
  --output "${RUN_DIR}/gate_sav_val_${GATE_COUNT}.txt" \
  --count "${GATE_COUNT}" \
  --seed "${GATE_SEED}" || return 1 2>/dev/null || exit 1
benchmark \
  sav_val \
  "${RUN_DIR}/sav_val_gate${GATE_COUNT}_box_benchmark" \
  "${RUN_DIR}/gate_sav_val_${GATE_COUNT}.txt" || \
  return 1 2>/dev/null || exit 1
check_gate || {
  log_metrics
  echo "[STOP] Official EdgeTAM failed the fidelity gate; do not run hybrids." >&2
  return 1 2>/dev/null || exit 1
}

if [[ "${ACTION}" == "gate" ]]; then
  log_metrics
  return 0 2>/dev/null || exit 0
fi

benchmark \
  sav_val \
  "${RUN_DIR}/sav_val_box_benchmark" \
  "${SAV_ROOT}/sav_val/sav_val.txt" || return 1 2>/dev/null || exit 1
benchmark \
  sav_test \
  "${RUN_DIR}/sav_test_box_benchmark" \
  "${SAV_ROOT}/sav_test/sav_test.txt" || return 1 2>/dev/null || exit 1
log_metrics || return 1 2>/dev/null || exit 1
python - "${RUN_DIR}/training_status.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "complete",
            "training": False,
            "note": "Official checkpoint evaluation-only fidelity baseline",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
touch "${RUN_DIR}/.pipeline_complete"

echo "Official EdgeTAM fidelity status: 0"
echo "Gate: ${RUN_DIR}/gate_status.json"
echo "Val: ${RUN_DIR}/sav_val_box_benchmark/metrics.csv"
echo "Test: ${RUN_DIR}/sav_test_box_benchmark/metrics.csv"
return 0 2>/dev/null || exit 0
