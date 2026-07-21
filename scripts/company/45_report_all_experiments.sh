#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

RUNS_ROOTS="${RUNS_ROOTS:-/danny-dataset/sam2_distill/runs:/group-volume/danny-dataset/sam2_distill/runs:/mnt/data/danny-dataset/sam2_distill/runs}"
REPORT_DIR="${REPORT_DIR:-/user-volume/all_experiment_report_${HOSTNAME}}"
OUT_CSV="${OUT_CSV:-${REPORT_DIR}/all_experiments.csv}"

ARGS=()
IFS=: read -r -a ROOT_ARRAY <<< "${RUNS_ROOTS}"
for root in "${ROOT_ARRAY[@]}"; do
  if [[ -d "${root}" ]]; then
    echo "Scanning runs root: ${root}"
    ARGS+=(--runs-root "${root}")
  else
    echo "Skip missing runs root: ${root}"
  fi
done

if [[ "${#ARGS[@]}" -eq 0 ]]; then
  echo "[ERROR] No run roots are available" >&2
  STATUS=1
else
  mkdir -p "${REPORT_DIR}"
  python tools/experiments/summarize_all_experiments.py \
    "${ARGS[@]}" \
    --out-csv "${OUT_CSV}"
  STATUS="$?"
fi

echo "All-experiment report status: ${STATUS}"
echo "All-experiment CSV: ${OUT_CSV}"
return "${STATUS}" 2>/dev/null || false
