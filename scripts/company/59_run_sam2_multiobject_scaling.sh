#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-all}"
case "${ACTION}" in
  describe|audit|tv21|sam21l|all) ;;
  *)
    echo "Usage: $0 {describe|audit|tv21|sam21l|all}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

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
          -d "${candidate}/sav_val/Annotations_6fps" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/danny-dataset/SA-V}"

SPLIT="${SPLIT:-sav_val}"
SPLIT_ROOT="${SAV_ROOT}/${SPLIT}"
IMAGE_ROOT="${IMAGE_ROOT:-${SPLIT_ROOT}/JPEGImages_24fps}"
ANN_ROOT="${ANN_ROOT:-${SPLIT_ROOT}/Annotations_6fps}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${SPLIT_ROOT}/${SPLIT}.txt}"

SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
TV21_CHECKPOINT="${TV21_CHECKPOINT:-${SAM2D_ROOT}/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
TINYVIT_MODEL_NAME="${TINYVIT_MODEL_NAME:-tiny_vit_21m_512.dist_in22k_ft_in1k}"

if [[ -z "${RUN_STORAGE_ROOT:-}" ]]; then
  if [[ -d /danny-dataset ]]; then
    RUN_STORAGE_ROOT=/danny-dataset/sam2_distill
  else
    RUN_STORAGE_ROOT="${SAM2D_ROOT}"
  fi
fi
RUN_ROOT="${RUN_ROOT:-${RUN_STORAGE_ROOT}/runs/sam2_multiobject_scaling_v1}"
LOG_ROOT="${LOG_ROOT:-/user-volume/sam2_multiobject_scaling_logs}"
OBJECT_COUNTS="${OBJECT_COUNTS:-1,2,4,8,16}"
MAX_OBJECTS=0
for object_count in ${OBJECT_COUNTS//,/ }; do
  if [[ "${object_count}" -gt "${MAX_OBJECTS}" ]]; then
    MAX_OBJECTS="${object_count}"
  fi
done
MAX_VIDEOS="${MAX_VIDEOS:-8}"
REPETITIONS="${REPETITIONS:-2}"
WARMUP_VIDEOS="${WARMUP_VIDEOS:-1}"
PROMPT_KIND="${PROMPT_KIND:-point}"
SEED="${SEED:-310107256}"
GPU="${GPU:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-multiobject-scaling-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
SKIP_DONE="${SKIP_DONE:-1}"
EXECUTION_MODE="${EXECUTION_MODE:-legacy}"
BUCKET_SIZE="${BUCKET_SIZE:-4}"
VERIFY_BUCKET_FRAMES="${VERIFY_BUCKET_FRAMES:-4}"
AUDIT_ROOT="${AUDIT_ROOT:-${RUN_ROOT}/density_audit_n${MAX_OBJECTS}}"
COHORT_FILE="${COHORT_FILE:-${AUDIT_ROOT}/cohort.txt}"

case "${PROMPT_KIND}" in
  point|box) ;;
  *)
    echo "[ERROR] PROMPT_KIND must be point or box: ${PROMPT_KIND}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
case "${EXECUTION_MODE}" in
  legacy|bucket) ;;
  *)
    echo "[ERROR] EXECUTION_MODE must be legacy or bucket: ${EXECUTION_MODE}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
if [[ "${BUCKET_SIZE}" -lt 1 ]]; then
  echo "[ERROR] BUCKET_SIZE must be positive: ${BUCKET_SIZE}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ "${VERIFY_BUCKET_FRAMES}" -lt 0 ]]; then
  echo "[ERROR] VERIFY_BUCKET_FRAMES cannot be negative: ${VERIFY_BUCKET_FRAMES}" >&2
  return 2 2>/dev/null || exit 2
fi

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

wandb_preflight() {
  [[ "${WANDB_MODE}" != "online" ]] && return 0
  python - <<'PY'
import wandb

viewer = wandb.Api(timeout=30).viewer
identity = (
    getattr(viewer, "username", None)
    or getattr(viewer, "email", None)
    or str(viewer)
)
if not identity:
    raise SystemExit("W&B viewer identity is empty")
print(f"W&B online preflight: PASS | {identity}", flush=True)
PY
}

describe_lane() {
  echo "SAM2 multi-object shared-session scaling v1"
  echo "Question: how quickly does FPS fall as one session grows from ${OBJECT_COUNTS} objects?"
  echo "Dataset: ${SPLIT_ROOT}"
  echo "Cohort: ${MAX_VIDEOS} videos with >= ${MAX_OBJECTS} non-empty masks on one shared frame"
  echo "Measurement: ${REPETITIONS} repetitions; ${WARMUP_VIDEOS} warmup video(s)"
  echo "Prompt: ${PROMPT_KIND}"
  echo "Execution: ${EXECUTION_MODE}; bucket capacity ${BUCKET_SIZE}"
  echo "Bucket correctness frames: ${VERIFY_BUCKET_FRAMES}"
  echo "GPU: visible device ${GPU}; one isolated GPU is intentional for latency validity"
  echo "TinyViT-21M task checkpoint: ${TV21_CHECKPOINT}"
  echo "SAM2.1-L comparison checkpoint: ${SAM2_CHECKPOINT}"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}; mode ${WANDB_MODE}"
  echo "Actions: audit -> tv21 -> sam21l"
}

run_audit() {
  local log
  log="${LOG_ROOT}/audit_$(date -u +%Y%m%d_%H%M%S).log"
  mkdir -p "${AUDIT_ROOT}" "${LOG_ROOT}"
  require_path "${IMAGE_ROOT}" || return 1
  require_path "${ANN_ROOT}" || return 1
  require_path "${VIDEO_LIST_FILE}" || return 1
  python tools/data/audit_vos_object_density.py \
    --image-root "${IMAGE_ROOT}" \
    --ann-root "${ANN_ROOT}" \
    --video-list-file "${VIDEO_LIST_FILE}" \
    --out-dir "${AUDIT_ROOT}" \
    --min-shared-objects "${MAX_OBJECTS}" \
    --max-cohort-videos "${MAX_VIDEOS}" \
    --seed "${SEED}" 2>&1 | tee "${log}"
  local status="${PIPESTATUS[0]}"
  echo "Density audit status: ${status}"
  echo "Cohort: ${COHORT_FILE}"
  echo "Log: ${log}"
  return "${status}"
}

ensure_cohort() {
  if [[ ! -s "${COHORT_FILE}" ]]; then
    run_audit || return 1
  fi
}

run_benchmark() {
  local model="$1"
  local count_tag="${OBJECT_COUNTS//,/-}"
  local execution_tag=""
  if [[ "${EXECUTION_MODE}" == "bucket" ]]; then
    execution_tag="_bucket${BUCKET_SIZE}"
  fi
  local out_dir="${RUN_ROOT}/${model}/${PROMPT_KIND}_n${count_tag}${execution_tag}"
  local log
  log="${LOG_ROOT}/${model}_${PROMPT_KIND}_$(date -u +%Y%m%d_%H%M%S).log"
  local -a model_args
  ensure_cohort || return 1
  require_path "${SAM2_ROOT}" || return 1
  require_path "${SAM2_CHECKPOINT}" || return 1
  if [[ "${SKIP_DONE}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skip completed benchmark: ${model}"
    cat "${out_dir}/aggregate.csv"
    return 0
  fi
  if [[ "${model}" == "tv21_best" ]]; then
    require_path "${TV21_CHECKPOINT}" || return 1
    require_path "${TINYVIT_CHECKPOINT}" || return 1
    model_args=(
      --model-kind stage1-student
      --checkpoint "${TV21_CHECKPOINT}"
      --sam2-checkpoint "${SAM2_CHECKPOINT}"
      --student-checkpoint "${TINYVIT_CHECKPOINT}"
      --student-family tinyvit
      --student-model-name "${TINYVIT_MODEL_NAME}"
    )
  elif [[ "${model}" == "sam21l" ]]; then
    model_args=(
      --model-kind sam2
      --checkpoint "${SAM2_CHECKPOINT}"
    )
  else
    echo "[ERROR] Unknown benchmark model: ${model}" >&2
    return 2
  fi

  mkdir -p "${out_dir}" "${LOG_ROOT}"
  wandb_preflight || return 1
  echo "===== Benchmark ${model} on GPU ${GPU} ====="
  CUDA_VISIBLE_DEVICES="${GPU}" \
    python tools/benchmark/benchmark_sam2_multiobject_scaling.py \
      "${model_args[@]}" \
      --prompt-kind "${PROMPT_KIND}" \
      --sam2-root "${SAM2_ROOT}" \
      --sam2-cfg "${SAM2_CONFIG}" \
      --image-root "${IMAGE_ROOT}" \
      --ann-root "${ANN_ROOT}" \
      --video-list-file "${COHORT_FILE}" \
      --out-dir "${out_dir}" \
      --object-counts "${OBJECT_COUNTS}" \
      --max-videos "${MAX_VIDEOS}" \
      --repetitions "${REPETITIONS}" \
      --warmup-videos "${WARMUP_VIDEOS}" \
      --seed "${SEED}" \
      --device cuda \
      --execution-mode "${EXECUTION_MODE}" \
      --bucket-size "${BUCKET_SIZE}" \
      --verify-bucket-frames "${VERIFY_BUCKET_FRAMES}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${model}_${SPLIT}_${PROMPT_KIND}_${EXECUTION_MODE}${BUCKET_SIZE}" \
      --wandb-mode "${WANDB_MODE}" 2>&1 | tee "${log}"
  local status="${PIPESTATUS[0]}"
  echo "${model} benchmark status: ${status}"
  echo "Aggregate: ${out_dir}/aggregate.csv"
  echo "Summary: ${out_dir}/summary.json"
  echo "Log: ${log}"
  return "${status}"
}

describe_lane
case "${ACTION}" in
  describe)
    ;;
  audit)
    run_audit
    ;;
  tv21)
    run_benchmark tv21_best
    ;;
  sam21l)
    run_benchmark sam21l
    ;;
  all)
    run_audit &&
      run_benchmark tv21_best &&
      run_benchmark sam21l
    ;;
esac
