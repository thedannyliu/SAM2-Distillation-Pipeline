#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
RECOVERY_ROOT="${EDGETAM_MEMORY_ROOT:-${SAM2D_ROOT}/runs/edgetam_memory_recovery_v2}"
LOG_ROOT="${LOG_ROOT:-/user-volume/edgetam_memory_recovery_logs/${LANE}}"
mkdir -p "${LOG_ROOT}"
FAILED=()

wandb_preflight() {
  [[ "${WANDB_MODE:-online}" != "online" ]] && return 0
  python - <<'PY'
import wandb

viewer = wandb.Api(timeout=30).viewer
username = getattr(viewer, "username", None) or getattr(viewer, "email", None)
if not username:
    raise SystemExit("W&B viewer has no username or email")
print(f"W&B online preflight: PASS | {username}")
PY
}

run_logged() {
  local variant="$1" log status
  log="${LOG_ROOT}/${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "================================================================"
  echo "Starting: ${variant}"
  echo "Log: ${log}"
  echo "================================================================"
  SAM2D_ROOT="${SAM2D_ROOT}" \
  EDGETAM_MEMORY_ROOT="${RECOVERY_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RECOVERY_ROOT}/summary.csv" \
  WANDB_PROJECT=edgetam-memory-recovery-v2 \
  GPUS="${GPUS:-0,1,2,3}" \
  FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS:-0,1,2,3}}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
    scripts/company/49_run_edgetam_memory_ablation.sh run "${variant}" \
      2>&1 | tee -a "${log}"
  status="${PIPESTATUS[0]}"
  echo "${variant} status: ${status}"
  [[ "${status}" -eq 0 ]] || FAILED+=("${variant}:${status}")
}

STATUS=0
wandb_preflight || STATUS="$?"
if [[ "${STATUS}" -eq 0 ]]; then
  case "${LANE}" in
    recovery1)
      run_logged C0_coherent_m0mem_align
      run_logged C3_coherent_m0mem_staged
      ;;
    recovery2)
      run_logged C1_partial_m0mem_align
      run_logged C2_coherent_m0mem_joint2ep
      ;;
    *)
      echo "Usage: $0 {recovery1|recovery2}" >&2
      STATUS=2
      ;;
  esac
fi

if [[ "${STATUS}" -eq 0 ]]; then
  EDGETAM_MEMORY_ROOT="${RECOVERY_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RECOVERY_ROOT}/summary.csv" \
    scripts/company/49_run_edgetam_memory_ablation.sh summarize \
      2>&1 | tee -a "${LOG_ROOT}/summary_$(date +%Y%m%d_%H%M%S).log"
  SUMMARY_STATUS="${PIPESTATUS[0]}"
  [[ "${SUMMARY_STATUS}" -eq 0 ]] || FAILED+=("summary:${SUMMARY_STATUS}")
fi

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  STATUS=1
  echo "Failed jobs: ${FAILED[*]}" >&2
fi
echo "Lane status: ${STATUS}"
echo "Lane: ${LANE}"
echo "Log root: ${LOG_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
