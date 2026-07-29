#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
VARIANT="${2:-}"
VARIANTS=(
  MO0_mem4_task_dense8_5ep
  MO1_mem2_task_dense8_5ep
  MO2_mem2_logits_dense8_5ep
  MO3_mem2_memlogits_dense8_5ep
)

is_variant() {
  local candidate="$1" item
  for item in "${VARIANTS[@]}"; do
    [[ "${candidate}" == "${item}" ]] && return 0
  done
  return 1
}

if [[ -z "${MODEL_SOURCE_ROOT:-}" ]]; then
  for candidate in \
    /danny-dataset/sam2_distill \
    /group-volume/danny-dataset/sam2_distill \
    /mnt/data/danny-dataset/sam2_distill; do
    if [[ -f "${candidate}/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt" && \
          -f "${candidate}/runs/tinyvit_max_jf_v1/tv21/main/resolved_config.yaml" && \
          -f "${candidate}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt" && \
          -f "${candidate}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors" && \
          -f "${candidate}/manifests/sav_stage1_vbal16_6fps_group_runtime.parquet" ]]; then
      MODEL_SOURCE_ROOT="${candidate}"
      break
    fi
  done
fi
MODEL_SOURCE_ROOT="${MODEL_SOURCE_ROOT:-/danny-dataset/sam2_distill}"

if [[ -z "${RUN_STORAGE_ROOT:-}" ]]; then
  if [[ -d /danny-dataset ]]; then
    RUN_STORAGE_ROOT=/danny-dataset/sam2_distill
  else
    RUN_STORAGE_ROOT="${MODEL_SOURCE_ROOT}"
  fi
fi

if [[ -z "${SAV_ROOT:-}" ]]; then
  for candidate in \
    /danny-dataset/SA-V \
    /group-volume/danny-dataset/SA-V \
    /mnt/data/danny-dataset/SA-V; do
    if [[ -f "${candidate}/sav_val/sav_val.txt" && \
          -f "${candidate}/sav_test/sav_test.txt" && \
          -d "${candidate}/sav_train" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/danny-dataset/SA-V}"

GPUS="${GPUS:-0,1,2,3}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "[ERROR] Each multi-object lane requires exactly four GPUs: ${GPUS}" >&2
  return 2 2>/dev/null || exit 2
fi

SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
RUN_ROOT="${RUN_ROOT:-${RUN_STORAGE_ROOT}/runs/sam2_multiobject_training_v1}"
SUMMARY_CSV="${SUMMARY_CSV:-${RUN_ROOT}/summary.csv}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-multiobject-training-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
LATENCY_GPU="${LATENCY_GPU:-${GPU_ARRAY[0]}}"
LATENCY_MAX_VIDEOS="${LATENCY_MAX_VIDEOS:-16}"
LATENCY_REPETITIONS="${LATENCY_REPETITIONS:-2}"
LATENCY_COHORT_ROOT="${RUN_ROOT}/cohorts/val_dense8"
LATENCY_COHORT="${LATENCY_COHORT_ROOT}/cohort.txt"
BEST_TV21_RUN="${MODEL_SOURCE_ROOT}/runs/tinyvit_max_jf_v1/tv21/main"

export GPUS
export FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
export SAM2D_ROOT="${MODEL_SOURCE_ROOT}"
export SAV_ROOT
export SAM2_TRAINING_ROOT
export EDGETAM_ROOT
export EDGETAM_MEMORY_ROOT="${RUN_ROOT}"
export EDGETAM_MEMORY_SUMMARY_CSV="${SUMMARY_CSV}"
export BEST_TV21_RUN
export BEST_TV21_CHECKPOINT="${BEST_TV21_RUN}/checkpoints/best.pt"
export BEST_TV21_CONFIG="${BEST_TV21_RUN}/resolved_config.yaml"
export WANDB_PROJECT
export WANDB_MODE
export EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE:-1}"

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

describe() {
  echo "SAM2 dense multi-object training v1"
  echo "Base/teacher: ${BEST_TV21_CHECKPOINT} (selected by full SA-V val J&F)"
  echo "Shared task: T4, up to 8 objects, frozen encoder/BN, dense SA-V stratum"
  echo "Node 1: MO0 | standard4 | task-only control | 5 epochs"
  echo "Node 2: MO1 | standard2 | task-only speed intervention | 5 epochs"
  echo "Node 3: MO2 | standard2 | task + mask-logit KD | 5 epochs"
  echo "Node 4: MO3 | standard2 | task + mask-logit + memory KD | 5 epochs"
  echo "Each node: train -> full SA-V val -> full SA-V test -> N=1/2/4/8 point latency"
  echo "Training/eval GPUs: ${GPUS}; latency GPU: ${LATENCY_GPU} in isolation"
  echo "Expected wall time: roughly 8-12 hours per node; storage/data throughput can shift this"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}; mode ${WANDB_MODE}"
}

validate_inputs() {
  local path
  for path in \
    "${BEST_TV21_CHECKPOINT}" \
    "${BEST_TV21_CONFIG}" \
    "${MODEL_SOURCE_ROOT}/manifests/sav_stage1_vbal16_6fps_group_runtime.parquet" \
    "${MODEL_SOURCE_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt" \
    "${MODEL_SOURCE_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors" \
    "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
    "${EDGETAM_ROOT}/sam2/modeling/perceiver.py" \
    "${SAV_ROOT}/sav_train" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
    "${SAV_ROOT}/sav_val/Annotations_6fps" \
    "${SAV_ROOT}/sav_test/sav_test.txt"; do
    require_path "${path}" || return 1
  done
}

ensure_latency_cohort() {
  mkdir -p "${LATENCY_COHORT_ROOT}"
  exec 8>"${LATENCY_COHORT_ROOT}/.lock" || return 1
  flock 8 || return 1
  if [[ ! -s "${LATENCY_COHORT}" ]]; then
    python tools/data/audit_vos_object_density.py \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${SAV_ROOT}/sav_val/sav_val.txt" \
      --out-dir "${LATENCY_COHORT_ROOT}" \
      --min-shared-objects 8 \
      --max-cohort-videos "${LATENCY_MAX_VIDEOS}" \
      --seed 310107256 || return 1
  fi
  flock -u 8
}

run_latency() {
  local run_dir="${RUN_ROOT}/${VARIANT}/main"
  local out_dir="${run_dir}/multiobject_latency/point_n1-2-4-8"
  ensure_latency_cohort || return 1
  if [[ "${SKIP_DONE:-1}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skip completed multi-object latency benchmark: ${VARIANT}"
    cat "${out_dir}/aggregate.csv"
    return 0
  fi
  require_path "${run_dir}/checkpoints/last.pt" || return 1
  require_path "${run_dir}/resolved_config.yaml" || return 1
  mkdir -p "${out_dir}"
  echo "===== N=1/2/4/8 shared-session latency: ${VARIANT} ====="
  CUDA_VISIBLE_DEVICES="${LATENCY_GPU}" \
  PYTHONPATH="${REPO_ROOT}:${EDGETAM_ROOT}:${SAM2_TRAINING_ROOT}:${PYTHONPATH:-}" \
    python tools/benchmark/benchmark_sam2_multiobject_scaling.py \
      --model-kind edgetam-trainer \
      --prompt-kind point \
      --sam2-root "${EDGETAM_ROOT}" \
      --sam2-cfg "${run_dir}/resolved_config.yaml" \
      --checkpoint "${run_dir}/checkpoints/last.pt" \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${LATENCY_COHORT}" \
      --out-dir "${out_dir}" \
      --object-counts 1,2,4,8 \
      --max-videos "${LATENCY_MAX_VIDEOS}" \
      --repetitions "${LATENCY_REPETITIONS}" \
      --warmup-videos 1 \
      --seed 310107256 \
      --device cuda \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${VARIANT}_latency" \
      --wandb-mode "${WANDB_MODE}"
}

refresh_summary() {
  python tools/train/summarize_mask_finetune_ablations.py scan \
    --root "${RUN_ROOT}" \
    --central-csv "${SUMMARY_CSV}"
}

run_variant() {
  validate_inputs || return 1
  scripts/company/49_run_edgetam_memory_ablation.sh run "${VARIANT}" || return 1
  run_latency || return 1
  refresh_summary || return 1
  touch "${RUN_ROOT}/${VARIANT}/.pipeline_complete"
  echo "Completed: ${VARIANT}"
  echo "Summary: ${SUMMARY_CSV}"
  cat "${RUN_ROOT}/${VARIANT}/summary.csv"
}

describe
case "${ACTION}" in
  describe)
    ;;
  run)
    if ! is_variant "${VARIANT}"; then
      echo "[ERROR] Usage: $0 run {${VARIANTS[*]}}" >&2
      return 2 2>/dev/null || exit 2
    fi
    run_variant
    ;;
  summarize)
    refresh_summary
    ;;
  *)
    echo "Usage: $0 {describe|run VARIANT|summarize}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

STATUS="$?"
echo "Multi-object training status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY_CSV}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
