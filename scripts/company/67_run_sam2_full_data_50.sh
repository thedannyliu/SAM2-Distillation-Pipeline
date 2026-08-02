#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
TARGET="${2:-}"
FULL_DATA_TOOL="tools/experiments/sam2_full_data_50.py"
mapfile -t VARIANTS < <(python "${FULL_DATA_TOOL}" list)

SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
FULL_SAV_MANIFEST="${FULL_SAV_MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/sam2_full_data_50_v1}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-full-data-50-v1}"
WANDB_MODE="${WANDB_MODE:-online}"
GPUS="${GPUS:-0,1,2,3}"
IFS=, read -r -a GPU_ARRAY <<< "${GPUS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "[ERROR] This suite requires four GPUs per node: ${GPUS}" >&2
  return 2 2>/dev/null || exit 2
fi

SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
MX5_RUN="${SAM2D_ROOT}/runs/sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep/main"
REFERENCE_LATENCY_DIR="${REFERENCE_LATENCY_DIR:-${SAM2D_ROOT}/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_ROOT}/comparison}"
GATE_VIDEOS="${GATE_VIDEOS:-64}"
LATENCY_GPU="${LATENCY_GPU:-${GPU_ARRAY[0]}}"
LATENCY_REPETITIONS="${LATENCY_REPETITIONS:-3}"
LATENCY_MAX_VIDEOS="${LATENCY_MAX_VIDEOS:-16}"
LATENCY_VERIFY_FRAMES="${LATENCY_VERIFY_FRAMES:-128}"
SKIP_DONE="${SKIP_DONE:-1}"
VARIANT_CSV="$(IFS=,; echo "${VARIANTS[*]}")"
LATENCY_COHORT="${RUN_ROOT}/cohorts/val_dense8/cohort.txt"

is_variant() {
  local candidate="$1" item
  for item in "${VARIANTS[@]}"; do
    [[ "${candidate}" == "${item}" ]] && return 0
  done
  return 1
}

load_variant_env() {
  local rendered key value
  rendered="$(python "${FULL_DATA_TOOL}" env "$1")" || return 1
  while IFS=$'\t' read -r key value; do
    case "${key}" in
      FD_*|TASK_*)
        printf -v "${key}" '%s' "${value}"
        export "${key?}"
        ;;
      *)
        echo "[ERROR] Unsupported full-data setting: ${key}" >&2
        return 2
        ;;
    esac
  done <<< "${rendered}"
}

describe() {
  echo "SAM2 full-data 50-run suite v1"
  echo "Budget: ten independent nodes, four H100s each, five sequential long runs per node"
  echo "Training: 5-8 epochs per variant; no one-epoch screening shortcuts"
  echo "Data: ${FULL_SAV_MANIFEST}"
  echo "All-video lanes: nodes 1, 2, 10, plus the FD11 data control"
  echo "Multiplex lanes compare full-manifest all/dense4/dense8 cohorts"
  echo "Evaluation: fixed ${GATE_VIDEOS}-video val gate plus 3-repeat N=1/2/4/8 latency"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}; mode ${WANDB_MODE}"
  local node
  for node in $(seq 1 10); do
    echo "Node ${node}: $(python "${FULL_DATA_TOOL}" queue "${node}")"
  done
}

prepare_repeated_cohort() {
  local name="$1"
  local source="${SAM2D_ROOT}/cohorts/sav_train_6fps_full/${name}_unique.txt"
  local output="${RUN_ROOT}/cohorts/${name}_train_ids.txt"
  if [[ -s "${output}" ]]; then
    echo "Training cohort ready: ${output}"
    return 0
  fi
  if [[ ! -s "${source}" ]]; then
    echo "[WARN] Unique ${name} cohort is missing; the first training node will rebuild it from the manifest: ${source}" >&2
    return 0
  fi
  python - "${source}" "${output}" "${MO_TRAIN_SAMPLES:-50337}" <<'PY'
import random
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:3])
target = int(sys.argv[3])
values = [
    line.strip()
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if not values:
    raise SystemExit(f"empty source cohort: {source}")
rng = random.Random(250107256)
repeated = []
while len(repeated) < target:
    cycle = list(values)
    rng.shuffle(cycle)
    repeated.extend(cycle)
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(
    "".join(f"{video_id}\n" for video_id in repeated[:target]),
    encoding="utf-8",
)
temporary.replace(output)
print(
    f"Prepared {output}: unique={len(values)}, samples={target}, seed=250107256"
)
PY
}

audit_inputs_once() {
  local marker="${RUN_ROOT}/.full_data_input_audit_passed"
  local lock="${RUN_ROOT}/.full_data_input_audit.lock"
  mkdir -p "${RUN_ROOT}"
  exec 7>"${lock}" || return 1
  flock 7 || return 1
  prepare_repeated_cohort dense4 || return 1
  prepare_repeated_cohort dense8 || return 1
  if [[ -f "${marker}" && "${marker}" -nt "${FULL_SAV_MANIFEST}" ]]; then
    echo "Full-data input audit already passed: ${marker}"
    flock -u 7
    return 0
  fi
  python "${FULL_DATA_TOOL}" validate || return 1
  python tools/train/audit_sam2_task_inputs.py \
    --manifest "${FULL_SAV_MANIFEST}" \
    --stage1-checkpoint "${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt" \
    --sav-root "${SAV_ROOT}" \
    --sample-videos "${AUDIT_SAMPLE_VIDEOS:-1000}" \
    --compact || return 1
  touch "${marker}"
  flock -u 7
  echo "Full-data input audit: PASS"
}

ensure_latency_cohort() {
  local cohort_dir="${RUN_ROOT}/cohorts/val_dense8"
  mkdir -p "${cohort_dir}"
  exec 8>"${cohort_dir}/.lock" || return 1
  flock 8 || return 1
  if [[ ! -s "${LATENCY_COHORT}" ]]; then
    python tools/data/audit_vos_object_density.py \
      --image-root "${SAV_ROOT}/sav_val/JPEGImages_24fps" \
      --ann-root "${SAV_ROOT}/sav_val/Annotations_6fps" \
      --video-list-file "${SAV_ROOT}/sav_val/sav_val.txt" \
      --out-dir "${cohort_dir}" \
      --min-shared-objects 8 \
      --max-cohort-videos "${LATENCY_MAX_VIDEOS}" \
      --seed 310107256 || return 1
  fi
  flock -u 8
}

run_latency() {
  local variant="$1" run_dir out_dir
  local -a mode_args
  load_variant_env "${variant}" || return 1
  run_dir="${RUN_ROOT}/${variant}/main"
  out_dir="${run_dir}/multiobject_latency/point_n1-2-4-8"
  if [[ "${SKIP_DONE}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skip completed latency: ${variant}"
    cat "${out_dir}/aggregate.csv"
    return 0
  fi
  ensure_latency_cohort || return 1
  if [[ "${FD_LATENCY_MODE}" == "bucket" ]]; then
    mode_args=(
      --execution-mode bucket
      --bucket-size "${TASK_OBJECT_SLOT_COUNT}"
      --bucket-min-objects "${TASK_OBJECT_SLOT_MIN_OBJECTS}"
      --verify-bucket-frames "${LATENCY_VERIFY_FRAMES}"
    )
  else
    mode_args=(--execution-mode legacy)
  fi
  mkdir -p "${out_dir}"
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
      "${mode_args[@]}" \
      --seed 310107256 \
      --device cuda \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-name "${variant}_latency" \
      --wandb-mode "${WANDB_MODE}"
}

run_variant() {
  local variant="$1" variant_dir="${RUN_ROOT}/$1" status
  if [[ "${SKIP_DONE}" == "1" && -f "${variant_dir}/.screen_complete" ]]; then
    echo "Skip completed full-data experiment: ${variant}"
    return 0
  fi
  load_variant_env "${variant}" || return 1
  audit_inputs_once || return 1
  echo "===== Long training: ${variant} ====="
  echo "Question: $(python "${FULL_DATA_TOOL}" describe "${variant}" | sed -n '2p')"
  echo "Epochs/T/cohort: ${TASK_EPOCHS}/${TASK_NUM_FRAMES}/${FD_DATA_COHORT}"
  if [[ "${FD_LATENCY_MODE}" == "bucket" ]]; then
    export VOS_EXECUTION_MODE=bucket
    export VOS_BUCKET_SIZE="${TASK_OBJECT_SLOT_COUNT}"
    export VOS_BUCKET_MIN_OBJECTS="${TASK_OBJECT_SLOT_MIN_OBJECTS}"
  else
    export VOS_EXECUTION_MODE=legacy
    unset VOS_BUCKET_SIZE VOS_BUCKET_MIN_OBJECTS
  fi
  SAM2D_ROOT="${SAM2D_ROOT}" \
  SAV_ROOT="${SAV_ROOT}" \
  MANIFEST="${FULL_SAV_MANIFEST}" \
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  GPUS="${GPUS}" \
  FULL_EVAL_GPUS="${GPUS}" \
  EDGETAM_SKIP_INPUT_AUDIT=1 \
  EDGETAM_EVAL_MODE=gate \
  EDGETAM_GATE_ENFORCE=0 \
  EDGETAM_GATE_MAX_VIDEOS="${GATE_VIDEOS}" \
  EDGETAM_GATE_MIN_JF=0 \
  EDGETAM_GATE_MIN_JF_RETENTION=0.95 \
  EDGETAM_GATE_MAX_JF_DROP=100 \
  EDGETAM_GATE_MAX_IMAGE_DROP=0.02 \
  EDGETAM_GATE_REFERENCE_CHECKPOINT="${MX5_RUN}/checkpoints/last.pt" \
  EDGETAM_GATE_REFERENCE_CONFIG="${MX5_RUN}/resolved_config.yaml" \
  EDGETAM_GATE_REFERENCE_TAG=mx5 \
  EDGETAM_GATE_REFERENCE_BUCKET_SIZE=8 \
  EDGETAM_GATE_REFERENCE_BUCKET_MIN_OBJECTS=4 \
  EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE}" \
    scripts/company/49_run_edgetam_memory_ablation.sh run "${variant}" || return 1
  run_latency "${variant}" || return 1
  touch "${variant_dir}/.screen_complete"
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
    scripts/company/49_run_edgetam_memory_ablation.sh summarize || status="$?"
  status="${status:-0}"
  echo "Completed ${variant}; status ${status}"
  return "${status}"
}

run_queue() {
  local node="$1" variant status queue
  local -a failures=()
  if [[ ! "${node}" =~ ^([1-9]|10)$ ]]; then
    echo "[ERROR] Queue number must be 1 through 10" >&2
    return 2
  fi
  queue="$(python "${FULL_DATA_TOOL}" queue "${node}")" || return 1
  audit_inputs_once || return 1
  for variant in ${queue}; do
    echo "===== Node ${node}: ${variant} ====="
    run_variant "${variant}"
    status="$?"
    echo "${variant} status: ${status}"
    if [[ "${status}" -ne 0 ]]; then
      failures+=("${variant}:${status}")
      echo "[CONTINUE] Recorded failure for ${variant}; starting the next experiment." >&2
    fi
  done
  if [[ "${#failures[@]}" -gt 0 ]]; then
    echo "===== Node ${node} failures =====" >&2
    printf '%s\n' "${failures[@]}" >&2
    echo "Rerun queue ${node} to retry failed variants; completed variants will be skipped." >&2
    return 1
  fi
  echo "Node ${node}: all five experiments completed successfully."
}

summarize() {
  python tools/benchmark/summarize_sam2_multiplex_screen.py \
    --run-root "${RUN_ROOT}" \
    --reference-latency-dir "${REFERENCE_LATENCY_DIR}" \
    --out-dir "${RESULTS_DIR}" \
    --variants "${VARIANT_CSV}" \
    --gate-videos "${GATE_VIDEOS}" \
    --min-quality-retention 0.95 \
    --min-learned-mask-iou 0.95 \
    --max-promotions 10
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  audit)
    audit_inputs_once || STATUS="$?"
    ;;
  run)
    if ! is_variant "${TARGET}"; then
      echo "[ERROR] Unknown full-data variant: ${TARGET}" >&2
      STATUS=2
    else
      run_variant "${TARGET}" || STATUS="$?"
    fi
    ;;
  queue)
    run_queue "${TARGET}" || STATUS="$?"
    ;;
  summarize)
    summarize || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|audit|run VARIANT|queue NODE|summarize}" >&2
    STATUS=2
    ;;
esac

echo "SAM2 full-data 50 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "Results: ${RESULTS_DIR}/screen_results.md"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
