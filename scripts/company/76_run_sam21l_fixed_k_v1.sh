#!/usr/bin/env bash

main() {
  local action="${1:-describe}"
  local target="${2:-A20}"
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
  cd "${repo_root}" || return 1

  local sam2d_root="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
  local sav_root="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
  local sam2_root="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
  local manifest="${MANIFEST:-${sam2d_root}/manifests/sav_train_6fps_full.parquet}"
  local checkpoint="${SAM2_CHECKPOINT:-${sam2d_root}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
  local model_config="${SAM2_MODEL_CONFIG:-${sam2_root}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
  local train_config="${repo_root}/configs/sam2_task/sam21l_two_clock_reuse_v1.yaml"
  local run_root="${RUN_ROOT:-${sam2d_root}/runs/sam21l_fixed_k_v1}"
  local old_run_root="${OLD_RUN_ROOT:-${sam2d_root}/runs/sam21l_two_clock_reuse_v1}"
  local log_root="${LOG_ROOT:-/user-volume/log/sam21l_fixed_k_v1}"
  local wandb_project="${WANDB_PROJECT:-sam2-fixed-k-v1}"
  local wandb_mode="${WANDB_MODE:-online}"
  local gpus="${GPUS:-0,1,2,3}"
  local gpu_count
  local git_sha
  IFS=, read -r -a gpu_array <<< "${gpus}"
  gpu_count="${#gpu_array[@]}"
  git_sha="$(git rev-parse --short=12 HEAD)" || return 1

  fixed_interval() {
    case "$1" in
      A1) echo 1 ;;
      A4) echo 4 ;;
      A8) echo 8 ;;
      A12) echo 12 ;;
      A16) echo 16 ;;
      A20) echo 20 ;;
      D4) echo 4 ;;
      D8) echo 8 ;;
      D12) echo 12 ;;
      D16) echo 16 ;;
      D20) echo 20 ;;
      *) echo "[ERROR] unsupported fixed-K target: $1" >&2; return 2 ;;
    esac
  }

  experiment_name() {
    case "$1" in
      A*) echo A ;;
      D*) echo D ;;
      *) echo "[ERROR] unsupported experiment target: $1" >&2; return 2 ;;
    esac
  }

  clip_length() {
    local interval="$1"
    if [[ "${interval}" -lt 8 ]]; then echo 8; else echo $((interval + 1)); fi
  }

  require_path() {
    [[ -e "$1" ]] || { echo "[ERROR] Missing required path: $1" >&2; return 1; }
  }

  describe() {
    echo "SAM2.1-L fixed-K v1: A1/A4/A8/A12/A16/A20 and D4/D8/D12/D16/D20"
    echo "Each formal training target uses one 4xH100 node for one SA-V epoch."
    echo "Each eval-fixed target uses one 4xH100 node for its phase-neutral eval30."
    echo "Selection-only snapshots: 10%, 25%, 50%; resumable checkpoint: epoch 1."
    echo "The eval-current action uses exactly one H100 and runs all old models sequentially."
    echo "Run root: ${run_root}"
    echo "W&B: ${wandb_project} (${wandb_mode})"
  }

  audit() {
    local status=0 path
    for path in \
      "${manifest}" "${sav_root}/sav_train" \
      "${sav_root}/sav_val/JPEGImages_24fps" \
      "${sav_root}/sav_val/Annotations_6fps" \
      "${sav_root}/sav_val/sav_val.txt" \
      "${checkpoint}" "${model_config}" \
      "${sam2_root}/training/model/sam2.py" \
      "${sam2_root}/sav_dataset/sav_evaluator.py"; do
      require_path "${path}" || status=1
    done
    if [[ "${status}" -eq 0 ]]; then
      SAV_ROOT="${sav_root}" SAM2D_ROOT="${sam2d_root}" \
        scripts/company/65_prepare_full_sav_memory_data.sh audit || status="$?"
    fi
    echo "Fixed-K input audit status: ${status}"
    return "${status}"
  }

  record_launch() {
    local run_dir="$1" scope="$2" experiment="$3" interval="$4" frames="$5" max_videos="$6"
    python - "${run_dir}" "${scope}" "${target}" "${interval}" "${frames}" \
      "${max_videos}" "${manifest}" "${checkpoint}" "${train_config}" "${gpus}" \
      "${wandb_project}" "${wandb_mode}" "${experiment}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

run, scope, target, interval, frames, max_videos, manifest, checkpoint, config, gpus, wandb_project, wandb_mode, experiment = sys.argv[1:]
run = Path(run)
payload = {
    "schema_version": 1,
    "suite": "sam21l_fixed_k_v1",
    "scope": scope,
    "target": target,
    "experiment": experiment,
    "fixed_refresh_interval": int(interval),
    "num_frames": int(frames),
    "max_feature_age": 19,
    "max_videos": int(max_videos),
    "epochs": 1,
    "gpus": gpus,
    "gpu_type": subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    ).stdout.strip().splitlines(),
    "seed": 250107256,
    "wandb_project": wandb_project,
    "wandb_mode": "disabled" if scope == "smoke" else wandb_mode,
    "run_directory": str(run),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "inputs": {},
}
for name, value in {"manifest": manifest, "initializer": checkpoint, "config": config}.items():
    path = Path(value)
    payload["inputs"][name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
path = run / "launch_manifest.json"
run.mkdir(parents=True, exist_ok=True)
if path.is_file() and json.loads(path.read_text()) != payload:
    raise SystemExit(f"launch manifest conflicts with existing run: {path}")
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  }

  train_target() {
    local scope="${1:-formal}" experiment interval frames run_dir max_videos freeze_steps mode fractions
    experiment="$(experiment_name "${target}")" || return $?
    interval="$(fixed_interval "${target}")" || return $?
    frames="$(clip_length "${interval}")"
    if [[ "${gpu_count}" -ne 4 ]]; then
      echo "[ERROR] training requires exactly four GPUs; got ${gpus}" >&2
      return 2
    fi
    if [[ "${scope}" == "smoke" ]]; then
      run_dir="${run_root}/smoke/${git_sha}/${target}"
      max_videos="${SMOKE_MAX_VIDEOS:-8}"
      freeze_steps=1
      mode=disabled
      fractions=""
    else
      run_dir="${run_root}/${target}"
      max_videos="${TASK_MAX_VIDEOS:-0}"
      freeze_steps="${TASK_ENCODER_FREEZE_STEPS:-1000}"
      mode="${wandb_mode}"
      fractions="0.10,0.25,0.50"
    fi
    record_launch "${run_dir}" "${scope}" "${experiment}" "${interval}" "${frames}" "${max_videos}" || return $?
    mkdir -p "${run_dir}" "${log_root}/${scope}"
    echo "===== ${scope} ${target}: fixed K=${interval}, T=${frames} ====="
    CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
    SAM2_TRAINING_ROOT="${sam2_root}" \
    TASK_TWO_CLOCK_V1=1 \
    TASK_TWO_CLOCK_EXPERIMENT="${experiment}" \
    TASK_TWO_CLOCK_FIXED_INTERVAL="${interval}" \
    TASK_TWO_CLOCK_MAX_AGE=19 \
    TASK_EXPERIMENT_SUITE=sam21l_fixed_k_v1 \
    TASK_STAGE_NAME="${target}" \
    TASK_MANIFEST="${manifest}" \
    SAV_ROOT="${sav_root}" \
    TASK_TEACHER_MODEL_CONFIG="${model_config}" \
    TASK_TEACHER_CHECKPOINT="${checkpoint}" \
    TASK_RUN_DIR="${run_dir}" \
    TASK_EPOCHS=1 \
    TASK_NUM_FRAMES="${frames}" \
    TASK_TRAIN_BATCH_SIZE=1 \
    TASK_MAX_NUM_OBJECTS=3 \
    TASK_MAX_VIDEOS="${max_videos}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
    TASK_SEED="${TASK_SEED:-250107256}" \
    TASK_ENCODER_FREEZE_STEPS="${freeze_steps}" \
    TASK_ENCODER_FORWARD_BATCH_SIZE="${TASK_ENCODER_FORWARD_BATCH_SIZE:-1}" \
    TASK_LR_WARMUP_FRACTION="${TASK_LR_WARMUP_FRACTION:-0.05}" \
    TASK_LR_WARMUP_START_FACTOR=0.1 \
    TASK_ENCODER_LR="${TASK_ENCODER_LR:-1e-6}" \
    TASK_ENCODER_LR_END="${TASK_ENCODER_LR_END:-1e-7}" \
    TASK_HEAD_LR="${TASK_HEAD_LR:-5e-6}" \
    TASK_HEAD_LR_END="${TASK_HEAD_LR_END:-5e-7}" \
    TASK_TEMPORAL_LR="${TASK_TEMPORAL_LR:-5e-5}" \
    TASK_TEMPORAL_LR_END="${TASK_TEMPORAL_LR_END:-5e-6}" \
    TASK_WEIGHT_DECAY=0.1 \
    TASK_LOG_EVERY="${TASK_LOG_EVERY:-30}" \
    TASK_PRINT_EVERY="${TASK_PRINT_EVERY:-300}" \
    TASK_CAPACITY_PROBE="$([[ "${scope}" == "smoke" ]] && echo 1 || echo 0)" \
    TASK_CAPACITY_WARMUP_STEPS=0 \
    TASK_GRADIENT_DIAGNOSTICS=1 \
    TASK_FRACTION_CHECKPOINTS="${fractions}" \
    WANDB_MODE="${mode}" \
      torchrun --standalone --nproc_per_node=4 \
        tools/train/run_sam2_task_training.py \
        --config "${train_config}" \
        --wandb-project "${wandb_project}" \
        --wandb-name "${target}_${scope}_seed${TASK_SEED:-250107256}" \
        --wandb-dir "${run_dir}/wandb" 2>&1 | tee -a "${log_root}/${scope}/${target}.log"
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -eq 0 && -f "${run_dir}/checkpoints/checkpoint.pt" ]]; then
      cp "${run_dir}/checkpoints/checkpoint.pt" "${run_dir}/checkpoints/last.pt"
    fi
    echo "${scope} status (${target}): ${status}"
    return "${status}"
  }

  eval_current() {
    if [[ "${gpu_count}" -ne 1 ]]; then
      echo "[ERROR] eval-current requires exactly one GPU; got ${gpus}" >&2
      return 2
    fi
    mkdir -p "${run_root}/eval30_current" "${log_root}/eval"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
      python tools/eval/run_two_clock_eval30.py \
        --repo-root "${repo_root}" \
        --sam2-root "${sam2_root}" \
        --sav-root "${sav_root}" \
        --old-run-root "${old_run_root}" \
        --official-checkpoint "${checkpoint}" \
        --out-root "${run_root}/eval30_current" \
        --count 30 \
        --seed 250107256 \
        --eval-processes "${EVAL_PROCESSES:-16}" 2>&1 | \
          tee -a "${log_root}/eval/eval30_current.log"
    local status="${PIPESTATUS[0]}"
    echo "Sequential eval30 status: ${status}"
    return "${status}"
  }

  eval_fixed() {
    local interval output status
    interval="$(fixed_interval "${target}")" || return $?
    if [[ "${gpu_count}" -ne 4 ]]; then
      echo "[ERROR] eval-fixed requires exactly four GPUs; got ${gpus}" >&2
      return 2
    fi
    require_path "${run_root}/A1/checkpoints/last.pt" || return $?
    if [[ "${target}" == D* ]]; then
      require_path "${run_root}/A${interval}/checkpoints/last.pt" || return $?
    fi
    require_path "${run_root}/${target}/checkpoints/last.pt" || return $?
    output="${run_root}/eval30_fixed/${target}"
    mkdir -p "${output}" "${log_root}/eval"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
      python tools/eval/run_fixed_k_eval30.py \
        --repo-root "${repo_root}" \
        --sam2-root "${sam2_root}" \
        --sav-root "${sav_root}" \
        --run-root "${run_root}" \
        --official-checkpoint "${checkpoint}" \
        --out-root "${output}" \
        --target "${target}" \
        --world-size "${gpu_count}" \
        --count 30 \
        --seed 250107256 \
        --eval-processes "${EVAL_PROCESSES:-16}" 2>&1 | \
          tee -a "${log_root}/eval/eval30_fixed_${target}.log"
    status="${PIPESTATUS[0]}"
    echo "Fixed-K eval30 status (${target}): ${status}"
    return "${status}"
  }

  status() {
    local item run_dir
    for item in A1 A4 A8 A12 A16 A20 D4 D8 D12 D16 D20; do
      run_dir="${run_root}/${item}"
      echo "===== ${item} ====="
      if [[ -f "${run_dir}/training_status.json" ]]; then
        cat "${run_dir}/training_status.json"
      else
        echo "training: pending"
      fi
      find "${run_dir}/checkpoints" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort
    done
    if [[ -f "${run_root}/eval30_current/report/report.md" ]]; then
      cat "${run_root}/eval30_current/report/report.md"
    else
      echo "eval30 report: pending"
    fi
  }

  case "${action}" in
    describe) describe ;;
    audit) audit ;;
    smoke) train_target smoke ;;
    train) train_target formal ;;
    eval-current) eval_current ;;
    eval-fixed) eval_fixed ;;
    status) status ;;
    *)
      echo "Usage: $0 {describe|audit|smoke|train|eval-current|eval-fixed|status} [A1|A4|A8|A12|A16|A20|D4|D8|D12|D16|D20]" >&2
      return 2
      ;;
  esac
}

main "$@"
