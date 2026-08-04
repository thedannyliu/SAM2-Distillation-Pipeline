#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
RUNNER="scripts/company/49_run_edgetam_memory_ablation.sh"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/tinyvit21_edgetam_memory_v1}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-tinyvit21-edgetam-memory-v1}"
GPUS="${GPUS:-0,1,2,3}"

STAGE1=EM1_t4_official_temporal_2ep
STAGE2=EM2_t8_joint_edgetam_5ep
STAGE3=EM3_t16_memory_refine_2ep

describe() {
  echo "TinyViT-21M EdgeTAM memory feasibility v1"
  echo "Data: full SA-V train manifest"
  echo "GPU: one node, four H100s (${GPUS})"
  echo "Stage 1: ${STAGE1} | freeze image encoder; adapt official EdgeTAM temporal stack"
  echo "Stage 2: ${STAGE2} | low-LR TinyViT + EdgeTAM memory joint adaptation"
  echo "Stage 3: ${STAGE3} | freeze encoder; T16 temporal refinement"
  echo "Evaluation: full SA-V val, then full SA-V test"
  echo "Manifest: ${MANIFEST}"
  echo "Run root: ${RUN_ROOT}"
}

audit() {
  local path
  for path in \
    "${MANIFEST}" \
    "${SOURCE_STAGE1_CHECKPOINT}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt"; do
    if [[ ! -e "${path}" ]]; then
      echo "[ERROR] Missing required path: ${path}" >&2
      return 1
    fi
  done
  python tools/train/audit_sam2_task_inputs.py \
    --manifest "${MANIFEST}" \
    --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
    --sav-root "${SAV_ROOT}" \
    --sample-videos "${AUDIT_SAMPLE_VIDEOS:-1000}" \
    --compact
}

run_stage() {
  local variant="$1" eval_mode="$2"
  echo "===== ${variant} | eval=${eval_mode} ====="
  SAM2D_ROOT="${SAM2D_ROOT}" \
  SAV_ROOT="${SAV_ROOT}" \
  MANIFEST="${MANIFEST}" \
  GPUS="${GPUS}" \
  FULL_EVAL_GPUS="${GPUS}" \
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
  EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE:-1}" \
  EDGETAM_SKIP_INPUT_AUDIT=1 \
  EDGETAM_EVAL_MODE="${eval_mode}" \
  SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
    "${RUNNER}" run "${variant}"
}

train() {
  run_stage "${STAGE1}" none || return $?
  run_stage "${STAGE2}" none || return $?
  run_stage "${STAGE3}" none
}

evaluate() {
  run_stage "${STAGE3}" full
}

run_all() {
  audit || return $?
  run_stage "${STAGE1}" none || return $?
  run_stage "${STAGE2}" none || return $?
  run_stage "${STAGE3}" full
}

status() {
  local variant checkpoint
  for variant in "${STAGE1}" "${STAGE2}" "${STAGE3}"; do
    checkpoint="${RUN_ROOT}/${variant}/main/checkpoints/last.pt"
    echo "===== ${variant} ====="
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
  done
  for split in sav_val sav_test; do
    local metrics="${RUN_ROOT}/${STAGE3}/main/${split}_box_benchmark/metrics.csv"
    echo "===== ${split} ====="
    if [[ -f "${metrics}" ]]; then
      column -s, -t < "${metrics}"
    else
      echo "metrics: pending"
    fi
  done
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  audit)
    audit || STATUS="$?"
    ;;
  train)
    train || STATUS="$?"
    ;;
  evaluate)
    evaluate || STATUS="$?"
    ;;
  all)
    run_all || STATUS="$?"
    ;;
  status)
    status || STATUS="$?"
    ;;
  summarize)
    EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
    EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
      "${RUNNER}" summarize || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|audit|train|evaluate|all|status|summarize}" >&2
    STATUS=2
    ;;
esac

echo "TinyViT-21M EdgeTAM memory v1 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
