#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

LANE="${1:-}"
LOG_ROOT="${LOG_ROOT:-/user-volume/edgetam_memory_logs/${LANE}}"
mkdir -p "${LOG_ROOT}"
FAILED=()

wandb_preflight() {
  [[ "${WANDB_MODE:-online}" != "online" ]] && return 0
  python - <<'PY'
import wandb

api = wandb.Api(timeout=30)
viewer = api.viewer
username = getattr(viewer, "username", None) or getattr(viewer, "email", None)
if not username:
    raise SystemExit("W&B viewer has no username or email")
print(f"W&B online preflight: PASS | {username}")
PY
}

run_logged() {
  local label="$1" variant="$2" log status
  log="${LOG_ROOT}/${label}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "================================================================"
  echo "Starting: ${label}"
  echo "Log: ${log}"
  echo "================================================================"
  GPUS="${GPUS:-0,1,2,3}" \
  FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS:-0,1,2,3}}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
    scripts/company/49_run_edgetam_memory_ablation.sh run "${variant}" \
      2>&1 | tee -a "${log}"
  status="${PIPESTATUS[0]}"
  echo "${label} status: ${status}"
  [[ "${status}" -eq 0 ]] || FAILED+=("${label}:${status}")
}

STATUS=0
wandb_preflight || STATUS="$?"
if [[ "${STATUS}" -eq 0 ]]; then
  case "${LANE}" in
    memory1)
      run_logged M0_sam2_mem4 M0_sam2_mem4
      run_logged M2a_edgetam_hybrid2_official M2a_edgetam_hybrid2_official
      run_logged R0_edgetam_e2e_t4_task R0_edgetam_e2e_t4_task
      run_logged R3_edgetam_e2e_t8_imgmemkd R3_edgetam_e2e_t8_imgmemkd
      ;;
    memory2)
      run_logged M1_sam2_mem2 M1_sam2_mem2
      run_logged M2b_edgetam_hybrid2_current M2b_edgetam_hybrid2_current
      run_logged R1_edgetam_e2e_t4_imgkd R1_edgetam_e2e_t4_imgkd
      run_logged R2_edgetam_e2e_t4_imgmemkd R2_edgetam_e2e_t4_imgmemkd
      ;;
    repro1)
      run_logged R0_edgetam_e2e_t4_task R0_edgetam_e2e_t4_task
      run_logged R3_edgetam_e2e_t8_imgmemkd R3_edgetam_e2e_t8_imgmemkd
      ;;
    repro2)
      run_logged R1_edgetam_e2e_t4_imgkd R1_edgetam_e2e_t4_imgkd
      run_logged R2_edgetam_e2e_t4_imgmemkd R2_edgetam_e2e_t4_imgmemkd
      ;;
    *)
      echo "Usage: $0 {memory1|memory2|repro1|repro2}" >&2
      STATUS=2
      ;;
  esac
fi

if [[ "${STATUS}" -eq 0 ]]; then
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
