#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
RUNNER="scripts/company/49_run_edgetam_memory_ablation.sh"
GPUS="${GPUS:-0,1,2,3}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-tv-multiplex-v1}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${SAM2D_ROOT:-}" ]]; then
  for candidate in \
    /danny-dataset/sam2_distill \
    /group-volume/danny-dataset/sam2_distill \
    /mnt/data/danny-dataset/sam2_distill; do
    if [[ -f "${candidate}/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt" ]]; then
      SAM2D_ROOT="${candidate}"
      break
    fi
  done
fi
SAM2D_ROOT="${SAM2D_ROOT:-/danny-dataset/sam2_distill}"

if [[ -z "${SAV_ROOT:-}" ]]; then
  for candidate in \
    /danny-dataset/SA-V \
    /group-volume/danny-dataset/SA-V \
    /mnt/data/danny-dataset/SA-V; do
    if [[ -f "${candidate}/sav_val/sav_val.txt" && \
          -f "${candidate}/sav_test/sav_test.txt" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/danny-dataset/SA-V}"

if [[ -z "${MANIFEST:-}" ]]; then
  for candidate in \
    "${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet" \
    /danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet \
    /group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet \
    /mnt/data/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet; do
    if [[ -f "${candidate}" ]]; then
      MANIFEST="${candidate}"
      break
    fi
  done
fi
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"

if [[ -z "${MX5_SLOT8_CHECKPOINT:-}" ]]; then
  for candidate in \
    "${SAM2D_ROOT}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/checkpoints/last.pt" \
    /danny-dataset/sam2_distill/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/checkpoints/last.pt \
    /group-volume/danny-dataset/sam2_distill/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/checkpoints/last.pt; do
    if [[ -f "${candidate}" ]]; then
      MX5_SLOT8_CHECKPOINT="${candidate}"
      break
    fi
  done
fi
MX5_SLOT8_CHECKPOINT="${MX5_SLOT8_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main/checkpoints/last.pt}"

RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_tv_multiplex_v1}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
STAGE1=SMX1_slot8_t4_bootstrap_2ep
STAGE2=SMX2_slot8_t8_fullsav_8ep
STAGE3=SMX3_slot8_t16_refine_2ep

describe() {
  echo "SAM2-TV multiplex v1"
  echo "Method: SAM 3.1-style 8 mask + 8 condition channels, private slot pointers, one attention/decoder call per bucket"
  echo "Stage 1: ${STAGE1} | T4 bootstrap from quality-preserving MX5"
  echo "Stage 2: ${STAGE2} | T8 full-SA-V training"
  echo "Stage 3: ${STAGE3} | T16 temporal refinement"
  echo "Evaluation: full sav_val -> full sav_test -> isolated N=1,2,4,8 latency"
  echo "Manifest: ${MANIFEST}"
  echo "Base checkpoint: ${MX5_SLOT8_CHECKPOINT}"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}; mode ${WANDB_MODE}"
}

audit() {
  local path
  for path in \
    "${MANIFEST}" \
    "${MX5_SLOT8_CHECKPOINT}" \
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
  echo "===== ${variant} | ${eval_mode} ====="
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
  MX5_SLOT8_CHECKPOINT="${MX5_SLOT8_CHECKPOINT}" \
  VOS_EXECUTION_MODE=legacy \
  SAM2_TV_COMPILE="${SAM2_TV_COMPILE:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
    "${RUNNER}" run "${variant}"
}

ensure_latency_cohort() {
  local cohort_dir="${RUN_ROOT}/cohorts/val_dense8"
  LATENCY_COHORT="${cohort_dir}/cohort.txt"
  mkdir -p "${cohort_dir}"
  if [[ ! -s "${LATENCY_COHORT}" ]]; then
    python tools/data/audit_vos_object_density.py \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${SAV_ROOT}/sav_val/sav_val.txt" \
      --out-dir "${cohort_dir}" \
      --min-shared-objects 8 \
      --max-cohort-videos "${LATENCY_MAX_VIDEOS:-16}" \
      --seed 310107256 || return 1
  fi
}

latency() {
  local run_dir="${RUN_ROOT}/${STAGE3}/main"
  local suffix="${LATENCY_OUTPUT_SUFFIX:-}"
  local out_dir="${run_dir}/multiobject_latency/point_n1-2-4-8${suffix}"
  ensure_latency_cohort || return 1
  if [[ "${SKIP_DONE:-1}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skip completed latency: ${out_dir}"
    column -s, -t < "${out_dir}/aggregate.csv"
    return 0
  fi
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${LATENCY_GPU:-${GPUS%%,*}}" \
  SAM2_TV_COMPILE="${SAM2_TV_COMPILE:-0}" \
  PYTHONPATH="${REPO_ROOT}:${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}:${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}:${PYTHONPATH:-}" \
    python tools/benchmark/benchmark_sam2_multiobject_scaling.py \
      --model-kind edgetam-trainer \
      --prompt-kind point \
      --sam2-root "${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}" \
      --sam2-cfg "${run_dir}/resolved_config.yaml" \
      --checkpoint "${run_dir}/checkpoints/last.pt" \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${LATENCY_COHORT}" \
      --out-dir "${out_dir}" \
      --object-counts 1,2,4,8 \
      --max-videos "${LATENCY_MAX_VIDEOS:-16}" \
      --repetitions "${LATENCY_REPETITIONS:-3}" \
      --warmup-videos 1 \
      --execution-mode legacy \
      --seed 310107256 \
      --device cuda \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${STAGE3}_latency${suffix}" \
      --wandb-mode "${WANDB_MODE}"
}

latency_compiled() {
  SAM2_TV_COMPILE=1 LATENCY_OUTPUT_SUFFIX=_compiled latency
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
  describe
  audit || return $?
  run_stage "${STAGE1}" none || return $?
  run_stage "${STAGE2}" none || return $?
  run_stage "${STAGE3}" full || return $?
  latency
}

status() {
  local variant checkpoint split metrics latency_csv
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
    metrics="${RUN_ROOT}/${STAGE3}/main/${split}_box_benchmark/metrics.csv"
    echo "===== ${split} ====="
    if [[ -f "${metrics}" ]]; then
      column -s, -t < "${metrics}"
    else
      echo "metrics: pending"
    fi
  done
  latency_csv="${RUN_ROOT}/${STAGE3}/main/multiobject_latency/point_n1-2-4-8/aggregate.csv"
  echo "===== MULTIOBJECT LATENCY ====="
  if [[ -f "${latency_csv}" ]]; then
    column -s, -t < "${latency_csv}"
  else
    echo "latency: pending"
  fi
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
  latency)
    latency || STATUS="$?"
    ;;
  latency-compiled)
    latency_compiled || STATUS="$?"
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
    echo "Usage: $0 {describe|audit|train|evaluate|latency|latency-compiled|all|status|summarize}" >&2
    STATUS=2
    ;;
esac

echo "SAM2-TV multiplex v1 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
