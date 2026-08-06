#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
SAM2_ROOT="${SAM2_ROOT:-/user-volume/repo/facebookresearch-sam2}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/eventsam2_coast_v2}"
COHORT_ROOT="${COHORT_ROOT:-${SAM2D_ROOT}/cohorts/eventsam2_sav_v1}"
SCREEN_DATA="${SCREEN_DATA:-${RUN_ROOT}/data/selection_screen32}"
SCREEN_ROOT="${RUN_ROOT}/selection_screen32"
VAL_ROOT="${RUN_ROOT}/sav_val"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_EVAL_PROCESSES="${NUM_EVAL_PROCESSES:-16}"
MAX_SCREEN_OBJECTS="${MAX_SCREEN_OBJECTS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-eventsam2-sav}"
WANDB_MODE="${WANDB_MODE:-online}"
COAST_WORKERS="${COAST_WORKERS:-16}"

describe() {
  echo "EventSAM2 A0/A1: bottleneck-skipping oracle-reset COAST screen"
  echo "Train roles: ${COHORT_ROOT}"
  echo "Selection screen: ${SCREEN_ROOT}"
  echo "Full SA-V validation: ${VAL_ROOT}"
  echo "COAST: 256px DIS optical flow + transient mask warp; no SAM2 encoder/read/decoder/write"
  echo "REFRESH: frozen SAM2.1-L full-trajectory mask reset"
  echo "Object contract: standard multi-object inference over frame-0 prompted objects"
  echo "This is an optimistic action-headroom screen, not state-consistent deployment."
  echo "sav_test is intentionally not read by this script."
}

check_common() {
  for path in "${SAV_ROOT}" "${MANIFEST}" "${SAM2_ROOT}" "${SAM2_CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
      echo "missing input: ${path}" >&2
      return 1
    fi
  done
}

audit() {
  check_common || return $?
  SAV_ROOT="${SAV_ROOT}" \
  SAM2D_ROOT="${SAM2D_ROOT}" \
  MANIFEST="${MANIFEST}" \
    scripts/company/65_prepare_full_sav_memory_data.sh audit
}

split() {
  check_common || return $?
  python tools/data/split_sav_eventsam2.py \
    --manifest "${MANIFEST}" \
    --sav-root "${SAV_ROOT}" \
    --out-dir "${COHORT_ROOT}" \
    --seed eventsam2_sav_v1 \
    --selection-screen-videos 32
}

prepare_screen() {
  check_common || return $?
  if [[ ! -f "${COHORT_ROOT}/selection_screen.txt" ]]; then
    split || return $?
  fi
  python tools/data/prepare_sav_train_shard_benchmark.py \
    --shard-root "${SAV_ROOT}/sav_train" \
    --out-root "${SCREEN_DATA}" \
    --video-list-file "${COHORT_ROOT}/selection_screen.txt" \
    --max-videos 0 \
    --max-objects-per-video "${MAX_SCREEN_OBJECTS}" \
    --require-first-frame-mask \
    --ann-every 4 \
    --frame-sample-rate 1
}

run_full() {
  local dataset_root="$1"
  local video_list="$2"
  local out_root="$3"
  mkdir -p "${out_root}/full_pred"
  torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    tools/eval/run_edgetam_vos_dataset.py \
    --edgetam-root "${SAM2_ROOT}" \
    --sam2-cfg "${SAM2_CONFIG}" \
    --checkpoint "${SAM2_CHECKPOINT}" \
    --image-root "${dataset_root}/JPEGImages_24fps" \
    --input-mask-root "${dataset_root}/Annotations_6fps" \
    --out-dir "${out_root}/full_pred" \
    --video-list-file "${video_list}" \
    --per-obj-png-file \
    --device cuda || return $?
  python tools/eval/merge_vos_rank_summaries.py \
    --run-dir "${out_root}/full_pred" || return $?
  python tools/eval/run_sav_evaluator.py \
    --evaluator "${SAM2_ROOT}/sav_dataset/sav_evaluator.py" \
    --gt-root "${dataset_root}/Annotations_6fps" \
    --pred-root "${out_root}/full_pred" \
    --out-json "${out_root}/full_eval.json" \
    --num-processes "${NUM_EVAL_PROCESSES}" \
    --do-not-skip-first-and-last-frame
}

full_screen() {
  if [[ ! -f "${SCREEN_DATA}/sav_train_benchmark.txt" ]]; then
    prepare_screen || return $?
  fi
  run_full "${SCREEN_DATA}" "${SCREEN_DATA}/sav_train_benchmark.txt" "${SCREEN_ROOT}"
}

full_val() {
  run_full "${SAV_ROOT}/sav_val" "${SAV_ROOT}/sav_val/sav_val.txt" "${VAL_ROOT}"
}

analyze() {
  local dataset_root="$1"
  local video_list="$2"
  local out_root="$3"
  local run_id="$4"
  local summary="${out_root}/full_pred/summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "missing full baseline summary: ${summary}" >&2
    return 1
  fi
  local full_latency_ms
  full_latency_ms="$(python - "${summary}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["single_stream_model_mean_ms"])
PY
)" || return $?
  python tools/experiments/run_eventsam2_coast_screen.py \
    --image-root "${dataset_root}/JPEGImages_24fps" \
    --gt-root "${dataset_root}/Annotations_6fps" \
    --full-pred-root "${out_root}/full_pred" \
    --video-list-file "${video_list}" \
    --out-dir "${out_root}/coast_screen" \
    --workers "${COAST_WORKERS}" \
    --sentinel-resolution 256 \
    --refresh-intervals 1,2,4,8,16 \
    --safe-horizons 4,8,16 \
    --safe-drop-points 1.0 \
    --full-latency-ms "${full_latency_ms}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-id "${run_id}" \
    --wandb-mode "${WANDB_MODE}"
}

analyze_screen() {
  analyze \
    "${SCREEN_DATA}" \
    "${SCREEN_DATA}/sav_train_benchmark.txt" \
    "${SCREEN_ROOT}" \
    eventsam2-coast-selection-screen32-v2
}

analyze_val() {
  analyze \
    "${SAV_ROOT}/sav_val" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${VAL_ROOT}" \
    eventsam2-coast-sav-val-v2
}

STATUS=0
case "${ACTION}" in
  describe) describe ;;
  audit) audit || STATUS="$?" ;;
  split) split || STATUS="$?" ;;
  prepare-screen) prepare_screen || STATUS="$?" ;;
  full-screen) full_screen || STATUS="$?" ;;
  analyze-screen) analyze_screen || STATUS="$?" ;;
  screen)
    audit || STATUS="$?"
    if [[ "${STATUS}" == "0" ]]; then split || STATUS="$?"; fi
    if [[ "${STATUS}" == "0" ]]; then prepare_screen || STATUS="$?"; fi
    if [[ "${STATUS}" == "0" ]]; then full_screen || STATUS="$?"; fi
    if [[ "${STATUS}" == "0" ]]; then analyze_screen || STATUS="$?"; fi
    ;;
  full-val) full_val || STATUS="$?" ;;
  analyze-val) analyze_val || STATUS="$?" ;;
  val)
    full_val || STATUS="$?"
    if [[ "${STATUS}" == "0" ]]; then analyze_val || STATUS="$?"; fi
    ;;
  *)
    echo "usage: $0 describe|audit|split|prepare-screen|full-screen|analyze-screen|screen|full-val|analyze-val|val" >&2
    STATUS=2
    ;;
esac

echo "EventSAM2 COAST ${ACTION} status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
