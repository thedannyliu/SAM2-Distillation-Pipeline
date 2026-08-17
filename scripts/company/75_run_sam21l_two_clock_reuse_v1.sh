#!/usr/bin/env bash

main() {
  local action="${1:-describe}"
  local target="${2:-E}"
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
  local verification_contract="${repo_root}/configs/verification/sam21l_two_clock_reuse_v1.json"
  local verification_tool="${repo_root}/tools/train/verify_two_clock_experiment.py"
  local run_root="${RUN_ROOT:-${sam2d_root}/runs/sam21l_two_clock_reuse_v1}"
  local verification_dir="${run_root}/verification"
  local log_root="${LOG_ROOT:-/user-volume/log/sam21l_two_clock_reuse_v1}"
  local wandb_project="${WANDB_PROJECT:-sam2-two-clock-reuse-v1}"
  local wandb_mode="${WANDB_MODE:-online}"
  local gpus="${GPUS:-0,1,2,3}"
  local gpu_count
  IFS=, read -r -a gpu_array <<< "${gpus}"
  gpu_count="${#gpu_array[@]}"
  local git_sha
  git_sha="$(git rev-parse --short=12 HEAD)" || return 1

  is_train_target() {
    case "$1" in
      O2|A|B|C|C-r|D|E) return 0 ;;
      *) return 1 ;;
    esac
  }

  require_path() {
    if [[ ! -e "$1" ]]; then
      echo "[ERROR] Missing required path: $1" >&2
      return 1
    fi
  }

  smoke_run_dir() {
    echo "${verification_dir}/smoke/${git_sha}/$1"
  }

  verify_local() {
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      local \
      --evidence-dir "${verification_dir}"
  }

  record_launch() {
    local experiment="$1" scope="$2" run_dir="$3" max_videos="$4"
    local epochs="$5" freeze_steps="$6" warmup="$7"
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      record-launch \
      --evidence-dir "${verification_dir}" \
      --run-dir "${run_dir}" \
      --scope "${scope}" \
      --target "${experiment}" \
      --gpus "${gpus}" \
      --seed "${TASK_SEED:-250107256}" \
      --wandb-project "${wandb_project}" \
      --wandb-mode "$([[ "${scope}" == "smoke" ]] && echo disabled || echo "${wandb_mode}")" \
      --setting "epochs=${epochs}" \
      --setting "max_videos=${max_videos}" \
      --setting "num_frames=8" \
      --setting "per_gpu_batch=1" \
      --setting "max_objects=3" \
      --setting "video_ids_file=${TASK_VIDEO_IDS_FILE:-}" \
      --setting "encoder_freeze_steps=${freeze_steps}" \
      --setting "encoder_forward_batch_size=${TASK_ENCODER_FORWARD_BATCH_SIZE:-1}" \
      --setting "lr_warmup_fraction=${warmup}" \
      --setting "encoder_lr=${TASK_ENCODER_LR:-1e-6}" \
      --setting "encoder_lr_end=${TASK_ENCODER_LR_END:-1e-7}" \
      --setting "head_lr=${TASK_HEAD_LR:-5e-6}" \
      --setting "head_lr_end=${TASK_HEAD_LR_END:-5e-7}" \
      --setting "temporal_lr=${TASK_TEMPORAL_LR:-5e-5}" \
      --setting "temporal_lr_end=${TASK_TEMPORAL_LR_END:-5e-6}" \
      --manifest "${manifest}" \
      --sam2-root "${sam2_root}" \
      --sam2-config "${model_config}" \
      --checkpoint "${checkpoint}"
  }

  check_launch() {
    local experiment="$1" run_dir="$2"
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      check-launch \
      --evidence-dir "${verification_dir}" \
      --run-dir "${run_dir}" \
      --target "${experiment}" \
      --manifest "${manifest}" \
      --sam2-root "${sam2_root}" \
      --sam2-config "${model_config}" \
      --checkpoint "${checkpoint}"
  }

  check_eval() {
    local experiment="$1" run_dir="$2"
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      check-eval \
      --run-dir "${run_dir}" \
      --target "${experiment}" \
      --manifest "${manifest}" \
      --sam2-root "${sam2_root}" \
      --sam2-config "${model_config}" \
      --checkpoint "${checkpoint}"
  }

  verify_remote() {
    local smoke_dir
    smoke_dir="$(smoke_run_dir E)" || return 1
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      remote \
      --evidence-dir "${verification_dir}" \
      --smoke-dir "${smoke_dir}" \
      --input-audit "${run_root}/audit/input_audit.json" \
      --manifest "${manifest}" \
      --sam2-root "${sam2_root}" \
      --sam2-config "${model_config}" \
      --checkpoint "${checkpoint}"
  }

  audit_existing() {
    local experiment="$1"
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      audit-existing \
      --run-dir "${run_root}/${experiment}" \
      --target "${experiment}" \
      --manifest "${manifest}" \
      --sam2-root "${sam2_root}" \
      --sam2-config "${model_config}" \
      --checkpoint "${checkpoint}"
  }

  postrun_audit() {
    local experiment="$1"
    python "${verification_tool}" \
      --repo-root "${repo_root}" \
      --contract "${verification_contract}" \
      postrun \
      --run-dir "${run_root}/${experiment}" \
      --target "${experiment}" \
      --manifest "${manifest}" \
      --sam2-root "${sam2_root}" \
      --sam2-config "${model_config}" \
      --checkpoint "${checkpoint}"
  }

  describe() {
    echo "SAM2.1-L two-clock stale-observation reuse v1"
    echo "Wave 1 nodes: O2, A, B, C, D, E; each node uses four H100s"
    echo "Training: full 50,337-video SA-V train, raw 24 FPS T8, five epochs"
    echo "Selection: phase-balanced full SA-V val R4 J&F across checkpoint_1..checkpoint_5"
    echo "Controls: official O0 R1 and frozen O1 R2--R6"
    echo "Run root: ${run_root}"
    echo "W&B: ${wandb_project} (${wandb_mode})"
    echo "GPUs: ${gpus}"
  }

  audit() {
    local status=0 path
    if [[ "${gpu_count}" -ne 4 ]]; then
      echo "[ERROR] Exactly four GPUs are required; got ${gpus}" >&2
      status=1
    fi
    for path in \
      "${manifest}" \
      "${sav_root}/sav_train" \
      "${sav_root}/sav_val/JPEGImages_24fps" \
      "${sav_root}/sav_val/Annotations_6fps" \
      "${sav_root}/sav_val/sav_val.txt" \
      "${checkpoint}" \
      "${model_config}" \
      "${sam2_root}/training/model/sam2.py" \
      "${sam2_root}/sav_dataset/sav_evaluator.py"; do
      require_path "${path}" || status=1
    done
    mkdir -p "${run_root}/audit"
    if [[ "${status}" -eq 0 ]]; then
      exec 9>"${run_root}/audit/.full_data_preparation.lock" || return 1
      flock 9 || return 1
      if [[ ! -f "${run_root}/audit/.full_data_preparation_passed" ]]; then
        SAV_ROOT="${sav_root}" SAM2D_ROOT="${sam2d_root}" \
          scripts/company/65_prepare_full_sav_memory_data.sh audit || status="$?"
        if [[ "${status}" -eq 0 ]]; then
          touch "${run_root}/audit/.full_data_preparation_passed"
        fi
      fi
      flock -u 9
    fi
    if [[ "${status}" -eq 0 ]]; then
      PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
        python tools/train/audit_two_clock_inputs.py \
          --manifest "${manifest}" \
          --sav-root "${sav_root}" \
          --sam2-root "${sam2_root}" \
          --sam2-config "${model_config}" \
          --checkpoint "${checkpoint}" \
          --sample-videos "${AUDIT_SAMPLE_VIDEOS:-32}" \
          --decode-videos "${AUDIT_DECODE_VIDEOS:-2}" \
          --out-json "${run_root}/audit/input_audit.json" || status="$?"
    fi
    echo "Two-clock input audit status: ${status}"
    return "${status}"
  }

  train_target() {
    local experiment="$1"
    local scope="${2:-formal}"
    local run_dir max_videos epochs mode freeze_steps warmup log_file status
    if ! is_train_target "${experiment}"; then
      echo "[ERROR] Train target must be O2/A/B/C/C-r/D/E: ${experiment}" >&2
      return 2
    fi
    if [[ "${scope}" == "smoke" ]]; then
      run_dir="$(smoke_run_dir "${experiment}")" || return 1
      max_videos="${SMOKE_MAX_VIDEOS:-8}"
      epochs=1
      mode=disabled
      freeze_steps=1
      warmup=0
    else
      run_dir="${run_root}/${experiment}"
      max_videos="${TASK_MAX_VIDEOS:-0}"
      epochs="${TASK_EPOCHS:-5}"
      mode="${wandb_mode}"
      freeze_steps="${TASK_ENCODER_FREEZE_STEPS:-1000}"
      warmup="${TASK_LR_WARMUP_FRACTION:-0.05}"
    fi
    record_launch \
      "${experiment}" "${scope}" "${run_dir}" "${max_videos}" \
      "${epochs}" "${freeze_steps}" "${warmup}" || return $?
    mkdir -p "${run_dir}" "${log_root}/${scope}"
    log_file="${log_root}/${scope}/${experiment}.log"
    echo "===== ${scope} training: ${experiment} ====="
    echo "Run directory: ${run_dir}"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
    SAM2_TRAINING_ROOT="${sam2_root}" \
    TASK_TWO_CLOCK_V1=1 \
    TASK_TWO_CLOCK_EXPERIMENT="${experiment}" \
    TASK_TWO_CLOCK_MAX_AGE=5 \
    TASK_EXPERIMENT_SUITE=sam21l_two_clock_reuse_v1 \
    TASK_STAGE_NAME="${experiment}" \
    TASK_MANIFEST="${manifest}" \
    SAV_ROOT="${sav_root}" \
    TASK_TEACHER_MODEL_CONFIG="${model_config}" \
    TASK_TEACHER_CHECKPOINT="${checkpoint}" \
    TASK_RUN_DIR="${run_dir}" \
    TASK_EPOCHS="${epochs}" \
    TASK_NUM_FRAMES=8 \
    TASK_TRAIN_BATCH_SIZE=1 \
    TASK_MAX_NUM_OBJECTS=3 \
    TASK_MAX_VIDEOS="${max_videos}" \
    TASK_VIDEO_IDS_FILE="${TASK_VIDEO_IDS_FILE:-}" \
    TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
    TASK_SEED="${TASK_SEED:-250107256}" \
    TASK_ENCODER_FREEZE_STEPS="${freeze_steps}" \
    TASK_ENCODER_FORWARD_BATCH_SIZE="${TASK_ENCODER_FORWARD_BATCH_SIZE:-1}" \
    TASK_LR_WARMUP_FRACTION="${warmup}" \
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
    TASK_GRADIENT_DIAGNOSTICS="${TASK_GRADIENT_DIAGNOSTICS:-1}" \
    TASK_TWO_CLOCK_FLIGHT_RECORDER="$([[ "${scope}" == "smoke" ]] && echo 1 || echo 0)" \
    WANDB_MODE="${mode}" \
      torchrun --standalone --nproc_per_node="${gpu_count}" \
        tools/train/run_sam2_task_training.py \
        --config "${train_config}" \
        --wandb-project "${wandb_project}" \
        --wandb-name "${experiment}_${scope}_seed${TASK_SEED:-250107256}" \
        --wandb-dir "${run_dir}/wandb" 2>&1 | tee -a "${log_file}"
    status="${PIPESTATUS[0]}"
    if [[ "${status}" -eq 0 && -f "${run_dir}/checkpoints/checkpoint.pt" ]]; then
      cp "${run_dir}/checkpoints/checkpoint.pt" "${run_dir}/checkpoints/last.pt"
    fi
    echo "${scope} training status (${experiment}): ${status}"
    return "${status}"
  }

  verify_smoke() {
    local experiment="$1" smoke_dir
    smoke_dir="$(smoke_run_dir "$1")" || return 1
    python - "${smoke_dir}" "${gpu_count}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
world = int(sys.argv[2])
required = [
    run / "checkpoints" / "checkpoint.pt",
    run / "checkpoints" / "last.pt",
    run / "resolved_config.yaml",
    run / "training_status.json",
    run / "gradient_diagnostics.json",
    run / "optimizer_audit.json",
]
required.extend(run / f"capacity_rank{rank}.json" for rank in range(world))
required.extend(run / f"mechanism_gradients_rank{rank}.json" for rank in range(world))
required.extend(run / f"flight_recorder_rank{rank}.jsonl" for rank in range(world))
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"smoke artifacts missing: {missing}")
training = json.loads((run / "training_status.json").read_text(encoding="utf-8"))
gradients = json.loads((run / "gradient_diagnostics.json").read_text(encoding="utf-8"))
if training.get("status") != "complete":
    raise SystemExit(f"smoke training did not complete: {training}")
if gradients.get("status") != "pass" or gradients.get("nonfinite_steps") != 0:
    raise SystemExit(f"smoke gradients failed: {gradients}")
capacities = [
    json.loads((run / f"capacity_rank{rank}.json").read_text(encoding="utf-8"))
    for rank in range(world)
]
if any(row.get("per_gpu_batch") != 1 for row in capacities):
    raise SystemExit(f"smoke batch contract failed: {capacities}")
print(json.dumps({"status": "pass", "training": training, "gradients": gradients, "capacity": capacities}, indent=2))
PY
  }

  evaluate_checkpoint() {
    local experiment="$1" checkpoint_path="$2" resolved_config="$3"
    local interval="$4" output="$5" status log_file
    if [[ -f "${output}/sav_eval.json" && -f "${output}/age_metrics.json" && "${SKIP_DONE:-1}" == "1" ]]; then
      if python - "${output}/sav_eval.json" "${output}/age_metrics.json" "${interval}" <<'PY'
import json
import sys
sav = json.load(open(sys.argv[1], encoding="utf-8"))
age = json.load(open(sys.argv[2], encoding="utf-8"))
valid = (
    sav.get("status") == "pass"
    and age.get("status") == "pass"
    and age.get("protocol") == "balanced_phase_v1"
    and age.get("refresh_phase_mode") == "balanced"
    and age.get("refresh_interval") == int(sys.argv[3])
    and all(row.get("count", 0) > 0 for row in age.get("by_age", []))
)
raise SystemExit(0 if valid else 1)
PY
      then
        echo "Skip completed evaluation: ${output}"
        return 0
      fi
    fi
    mkdir -p "${output}/pred" "${log_root}/eval"
    log_file="${log_root}/eval/${experiment}_$(basename "${output}").log"
    echo "===== val ${experiment} R${interval}: ${checkpoint_path} ====="
    CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHONPATH="${repo_root}:${sam2_root}:${PYTHONPATH:-}" \
      torchrun --standalone --nproc_per_node="${gpu_count}" \
        tools/eval/run_edgetam_vos_dataset.py \
        --model-kind two-clock \
        --sam2-root "${sam2_root}" \
        --sam2-cfg "${model_config}" \
        --checkpoint "${checkpoint_path}" \
        --resolved-config "${resolved_config}" \
        --experiment "${experiment}" \
        --refresh-interval "${interval}" \
        --refresh-phase-mode balanced \
        --refresh-phase-seed 250107256 \
        --image-root "${sav_root}/sav_val/JPEGImages_24fps" \
        --input-mask-root "${sav_root}/sav_val/Annotations_6fps" \
        --video-list-file "${sav_root}/sav_val/sav_val.txt" \
        --out-dir "${output}/pred" \
        --per-obj-png-file \
        --track-object-appearing-later-in-video \
        --device cuda 2>&1 | tee -a "${log_file}"
    status="${PIPESTATUS[0]}"
    if [[ "${status}" -ne 0 ]]; then
      return "${status}"
    fi
    python tools/eval/merge_vos_rank_summaries.py \
      --run-dir "${output}/pred" || return $?
    python tools/eval/run_sav_evaluator.py \
      --evaluator "${sam2_root}/sav_dataset/sav_evaluator.py" \
      --gt-root "${sav_root}/sav_val/Annotations_6fps" \
      --pred-root "${output}/pred" \
      --out-json "${output}/sav_eval.json" \
      --num-processes "${EVAL_PROCESSES:-16}" \
      --strict || return $?
    python tools/eval/evaluate_two_clock_age.py \
      --sam2-root "${sam2_root}" \
      --gt-root "${sav_root}/sav_val/Annotations_6fps" \
      --pred-root "${output}/pred" \
      --refresh-interval "${interval}" \
      --refresh-phase-mode balanced \
      --refresh-phase-seed 250107256 \
      --video-list-file "${sav_root}/sav_val/sav_val.txt" \
      --out-json "${output}/age_metrics.json" \
      --out-csv "${output}/age_metrics.csv"
  }

  select_checkpoint() {
    local experiment="$1" run_dir checkpoint_path output epoch
    run_dir="${run_root}/${experiment}"
    check_eval "${experiment}" "${run_dir}" || return $?
    require_path "${run_dir}/resolved_config.yaml" || return 1
    for epoch in 1 2 3 4 5; do
      checkpoint_path="${run_dir}/checkpoints/checkpoint_${epoch}.pt"
      require_path "${checkpoint_path}" || return 1
      output="${run_dir}/val/epoch_${epoch}/R4"
      evaluate_checkpoint \
        "${experiment}" "${checkpoint_path}" "${run_dir}/resolved_config.yaml" 4 "${output}" || return $?
    done
    python - "${run_dir}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = []
for epoch in range(1, 6):
    path = run / "val" / f"epoch_{epoch}" / "R4" / "sav_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"epoch": epoch, **payload["metrics"], "metrics": str(path)})
if any("J&F" not in row for row in rows):
    raise SystemExit(f"missing J&F in epoch rows: {rows}")
best = max(rows, key=lambda row: (row["J&F"], -row["epoch"]))
source = run / "checkpoints" / f"checkpoint_{best['epoch']}.pt"
destination = run / "checkpoints" / "best.pt"
shutil.copy2(source, destination)
result = {
    "selection_metric": "full_sav_val_balanced_phase_R4_J&F",
    "selected_epoch": best["epoch"],
    "selected_checkpoint": str(destination),
    "rows": rows,
}
(run / "best_selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
  }

  curves() {
    local experiment="$1" run_dir interval
    run_dir="${run_root}/${experiment}"
    check_eval "${experiment}" "${run_dir}" || return $?
    require_path "${run_dir}/checkpoints/best.pt" || return 1
    for interval in 1 2 3 4 5 6; do
      evaluate_checkpoint \
        "${experiment}" \
        "${run_dir}/checkpoints/best.pt" \
        "${run_dir}/resolved_config.yaml" \
        "${interval}" \
        "${run_dir}/val/selected/R${interval}" || return $?
    done
  }

  controls() {
    local interval
    evaluate_checkpoint \
      O0 "${checkpoint}" "${model_config}" 1 "${run_root}/controls/O0/R1" || return $?
    for interval in 2 3 4 5 6; do
      evaluate_checkpoint \
        O1 "${checkpoint}" "${model_config}" "${interval}" \
        "${run_root}/controls/O1/R${interval}" || return $?
    done
  }

  status() {
    local experiment run_dir
    for experiment in O2 A B C D E; do
      run_dir="${run_root}/${experiment}"
      echo "===== ${experiment} ====="
      if [[ -f "${run_dir}/training_status.json" ]]; then
        cat "${run_dir}/training_status.json"
      else
        echo "training: pending"
      fi
      if [[ -f "${run_dir}/best_selection.json" ]]; then
        cat "${run_dir}/best_selection.json"
      else
        echo "selection: pending"
      fi
    done
  }

  case "${action}" in
    describe) describe ;;
    audit) audit ;;
    verify-local) verify_local ;;
    verify-remote) audit && verify_remote ;;
    audit-existing) audit_existing "${target}" ;;
    postrun) postrun_audit "${target}" ;;
    smoke)
      if [[ "${target}" != "E" ]]; then
        echo "[ERROR] The registered remote verification smoke target is E" >&2
        return 2
      fi
      verify_local && audit && train_target "${target}" smoke && \
        verify_smoke "${target}" && verify_remote
      ;;
    train) train_target "${target}" formal ;;
    select) select_checkpoint "${target}" ;;
    curves) curves "${target}" ;;
    val)
      select_checkpoint "${target}" && curves "${target}" && \
        postrun_audit "${target}"
      ;;
    controls) controls ;;
    run)
      audit && train_target "${target}" formal && select_checkpoint "${target}" && \
        curves "${target}" && postrun_audit "${target}"
      ;;
    status) status ;;
    *)
      echo "Usage: $0 {describe|audit|verify-local|smoke|verify-remote|audit-existing|postrun|train|select|curves|val|controls|run|status} [O2|A|B|C|C-r|D|E]" >&2
      return 2
      ;;
  esac
}

main "$@"
