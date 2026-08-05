#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
RUNNER="scripts/company/49_run_edgetam_memory_ablation.sh"
GPUS="${GPUS:-0,1,2,3}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"

SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/edgetam_tv21_sam21l_v1/formal}"
BEST_TV21_RUN="${BEST_TV21_RUN:-${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/tv21/main}"
BEST_TV21_CHECKPOINT="${BEST_TV21_CHECKPOINT:-${BEST_TV21_RUN}/checkpoints/best.pt}"
BEST_TV21_VAL_METRICS="${BEST_TV21_VAL_METRICS:-${BEST_TV21_RUN}/sav_val_box_benchmark/metrics.csv}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/edgetam/edgetam.pt}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-edgetam-tv21-sam21l-v1}"
WANDB_MODE="${WANDB_MODE:-online}"

IMAGE=ETV1_image_reanchor_1ep
T4=ETV2_t4_bootstrap_2ep
T8=ETV3_t8_joint_3ep
T16=ETV4_t16_refine_1ep

stage_run_dir() {
  echo "${RUN_ROOT}/$1/main"
}

stage_metrics() {
  echo "$(stage_run_dir "$1")/sav_val_box_benchmark/metrics.csv"
}

describe() {
  echo "TV21 EdgeTAM + online SAM2.1-L formal run"
  echo "GPUs: ${GPUS} (${#GPU_ARRAY[@]} ranks)"
  echo "Data: ${MANIFEST}"
  echo "Image: batch/GPU 32, global 128, T1, 1 pass, full val"
  echo "T4: batch/GPU 6, global 24, 2 passes, full val"
  echo "T8: batch/GPU 4, global 16, 3 passes, full val"
  echo "T16: batch/GPU 2, global 8, 1 pass, full val then full test"
  echo "Gradient accumulation: disabled"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}; mode ${WANDB_MODE}"
  local stage
  for stage in "${IMAGE}" "${T4}" "${T8}" "${T16}"; do
    "${RUNNER}" describe "${stage}" || return $?
  done
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  fi
}

audit() {
  local path status=0
  if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "[ERROR] Exactly four GPUs are required: ${GPUS}" >&2
    status=1
  fi
  for path in \
    "${MANIFEST}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt" \
    "${BEST_TV21_CHECKPOINT}" \
    "${BEST_TV21_VAL_METRICS}" \
    "${SOURCE_STAGE1_CHECKPOINT}" \
    "${TINYVIT_CHECKPOINT}" \
    "${SAM2_CHECKPOINT}" \
    "${EDGETAM_CHECKPOINT}" \
    "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
    "${EDGETAM_ROOT}/sam2/modeling/perceiver.py"; do
    require_path "${path}" || status=1
  done
  if [[ "${status}" -eq 0 ]]; then
    python tools/train/audit_sam2_task_inputs.py \
      --manifest "${MANIFEST}" \
      --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
      --sav-root "${SAV_ROOT}" \
      --sample-videos "${AUDIT_SAMPLE_VIDEOS:-1000}" \
      --compact || status="$?"
  fi
  echo "Formal run audit status: ${status}"
  return "${status}"
}

select_t4_base() {
  local gate
  gate="$(stage_run_dir "${IMAGE}")/gate_status.json"
  if gate_passed "${gate}"; then
    echo "$(stage_run_dir "${IMAGE}")/checkpoints/last.pt"
  else
    echo "${BEST_TV21_CHECKPOINT}"
  fi
}

gate_passed() {
  local gate="$1"
  [[ -f "${gate}" ]] || return 1
  python - "${gate}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "pass" else 1)
PY
}

require_stage_gate() {
  local stage="$1" gate
  gate="$(stage_run_dir "${stage}")/gate_status.json"
  if ! gate_passed "${gate}"; then
    echo "[ERROR] ${stage} must pass full-validation gate first: ${gate}" >&2
    return 1
  fi
}

run_stage() {
  local stage="$1" eval_mode="$2" t4_base eval_gpus
  t4_base="${ETV_T4_BASE_CHECKPOINT:-$(select_t4_base)}"
  eval_gpus="${GPUS}"
  echo "===== ${stage} | eval=${eval_mode} ====="
  SAM2D_ROOT="${SAM2D_ROOT}" \
  SAV_ROOT="${SAV_ROOT}" \
  MANIFEST="${MANIFEST}" \
  GPUS="${GPUS}" \
  FULL_EVAL_GPUS="${eval_gpus}" \
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
  EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE:-1}" \
  EDGETAM_SKIP_INPUT_AUDIT=1 \
  EDGETAM_EVAL_MODE="${eval_mode}" \
  ETV_T4_BASE_CHECKPOINT="${t4_base}" \
  BEST_TV21_CHECKPOINT="${BEST_TV21_CHECKPOINT}" \
  SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
  TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
  SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
  EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT}" \
  SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT}" \
  EDGETAM_ROOT="${EDGETAM_ROOT}" \
  VOS_EXECUTION_MODE=legacy \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
  TASK_EPOCHS_OVERRIDE="${TASK_EPOCHS_OVERRIDE:-}" \
  TASK_MAX_VIDEOS_OVERRIDE="${TASK_MAX_VIDEOS_OVERRIDE:-}" \
    "${RUNNER}" run "${stage}"
}

gate_stage() {
  local stage="$1" reference="$2" min_jf="$3" max_jf_drop="$4"
  local run_dir metrics out
  run_dir="$(stage_run_dir "${stage}")"
  metrics="$(stage_metrics "${stage}")"
  out="${run_dir}/gate_status.json"
  python tools/experiments/check_sav_memory_gate.py \
    --metrics "${metrics}" \
    --reference-metrics "${reference}" \
    --out-json "${out}" \
    --min-jf "${min_jf}" \
    --max-jf-drop "${max_jf_drop}" \
    --max-miou-drop 0.005 \
    --max-ap-drop 0.005
}

run_image() {
  run_stage "${IMAGE}" val || return $?
  if gate_stage "${IMAGE}" "${BEST_TV21_VAL_METRICS}" 0 100; then
    echo "Image re-anchor gate: PASS"
  else
    echo "[FALLBACK] Image re-anchor failed; T4 will use selected TV21." >&2
  fi
}

run_t4() {
  run_stage "${T4}" val || return $?
  gate_stage "${T4}" "${BEST_TV21_VAL_METRICS}" 55 100
}

run_t8() {
  require_stage_gate "${T4}" || return $?
  run_stage "${T8}" val || return $?
  gate_stage "${T8}" "${BEST_TV21_VAL_METRICS}" 60 100
}

run_t16() {
  require_stage_gate "${T8}" || return $?
  run_stage "${T16}" val || return $?
  gate_stage "${T16}" "$(stage_metrics "${T8}")" 60 0.3
}

run_final_test() {
  require_stage_gate "${T16}" || return $?
  run_stage "${T16}" full
}

smoke() {
  local formal_root="${RUN_ROOT}"
  local smoke_root="${formal_root}/smoke"
  local smoke_image="${smoke_root}/${IMAGE}/main/checkpoints/last.pt"
  echo "===== Bounded four-stage forward/backward smoke ====="
  RUN_ROOT="${smoke_root}" \
  WANDB_MODE=disabled \
  TASK_NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}" \
  TASK_EPOCHS_OVERRIDE=1 \
  TASK_MAX_VIDEOS_OVERRIDE=128 \
    run_stage "${IMAGE}" none || return $?
  RUN_ROOT="${smoke_root}" \
  WANDB_MODE=disabled \
  ETV_T4_BASE_CHECKPOINT="${smoke_image}" \
  TASK_NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}" \
  TASK_EPOCHS_OVERRIDE=1 \
  TASK_MAX_VIDEOS_OVERRIDE=24 \
    run_stage "${T4}" none || return $?
  RUN_ROOT="${smoke_root}" \
  WANDB_MODE=disabled \
  TASK_NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}" \
  TASK_EPOCHS_OVERRIDE=1 \
  TASK_MAX_VIDEOS_OVERRIDE=16 \
    run_stage "${T8}" none || return $?
  RUN_ROOT="${smoke_root}" \
  WANDB_MODE=disabled \
  TASK_NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}" \
  TASK_EPOCHS_OVERRIDE=1 \
  TASK_MAX_VIDEOS_OVERRIDE=8 \
    run_stage "${T16}" none || return $?
  echo "Four-stage smoke: PASS"
}

run_all() {
  describe || return $?
  audit || return $?
  run_image || return $?
  run_t4 || return $?
  run_t8 || return $?
  run_t16 || return $?
  run_final_test
}

status() {
  local stage metrics checkpoint
  for stage in "${IMAGE}" "${T4}" "${T8}" "${T16}"; do
    echo "===== ${stage} ====="
    checkpoint="$(stage_run_dir "${stage}")/checkpoints/last.pt"
    metrics="$(stage_metrics "${stage}")"
    if [[ -f "${checkpoint}" ]]; then
      python - "${checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
steps = checkpoint.get("steps", {})
if isinstance(steps, dict):
    steps = steps.get("train", steps)
print(f"checkpoint={sys.argv[1]}")
print(f"epoch={checkpoint.get('epoch', '-')} train_steps={steps}")
PY
    else
      echo "checkpoint: pending"
    fi
    if [[ -f "${metrics}" ]]; then
      column -s, -t < "${metrics}"
    else
      echo "validation: pending"
    fi
  done
  metrics="$(stage_run_dir "${T16}")/sav_test_box_benchmark/metrics.csv"
  echo "===== final sav_test ====="
  if [[ -f "${metrics}" ]]; then
    column -s, -t < "${metrics}"
  else
    echo "test: pending"
  fi
}

STATUS=0
case "${ACTION}" in
  describe)
    describe || STATUS="$?"
    ;;
  audit)
    audit || STATUS="$?"
    ;;
  smoke)
    audit && smoke || STATUS="$?"
    ;;
  image)
    run_image || STATUS="$?"
    ;;
  t4)
    run_t4 || STATUS="$?"
    ;;
  t8)
    run_t8 || STATUS="$?"
    ;;
  t16)
    run_t16 || STATUS="$?"
    ;;
  test)
    run_final_test || STATUS="$?"
    ;;
  all)
    run_all || STATUS="$?"
    ;;
  status)
    status || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|audit|smoke|image|t4|t8|t16|test|all|status}" >&2
    STATUS=2
    ;;
esac

echo "TV21 EdgeTAM formal run status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
