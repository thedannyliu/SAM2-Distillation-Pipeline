#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
RUNS_ROOTS="${RUNS_ROOTS:-/group-volume/danny-dataset/sam2_distill/runs:/mnt/data/danny-dataset/sam2_distill/runs}"
REPORT_DIR="${REPORT_DIR:-/user-volume/stage1_run_progress_${HOSTNAME}}"

[[ -f "${SAV_ROOT}/sav_val/sav_val.txt" ]] || {
  echo "missing full SA-V validation split: ${SAV_ROOT}/sav_val" >&2
  exit 1
}
[[ -f "${SAV_ROOT}/sav_test/sav_test.txt" ]] || {
  echo "missing full SA-V test split: ${SAV_ROOT}/sav_test" >&2
  exit 1
}

args=()
IFS=: read -r -a candidate_roots <<< "${RUNS_ROOTS}"
for root in "${candidate_roots[@]}"; do
  if [[ -d "${root}" ]]; then
    args+=(--runs-root "${root}")
  else
    echo "skip missing run root: ${root}" >&2
  fi
done
if [[ "${#args[@]}" -eq 0 ]]; then
  echo "no run roots found in RUNS_ROOTS=${RUNS_ROOTS}" >&2
  exit 1
fi

mkdir -p "${REPORT_DIR}"
python tools/experiments/audit_stage1_run_progress.py \
  "${args[@]}" \
  --sav-root "${SAV_ROOT}" \
  --out-json "${REPORT_DIR}/stage1_run_progress.json" \
  --out-csv "${REPORT_DIR}/stage1_run_progress.csv"

echo
echo "===== Registered matrix and discovered legacy runs ====="
python - "${REPORT_DIR}/stage1_run_progress.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
columns = ["family", "registered", "queue", "name", "status", "step", "target_steps", "progress_pct", "wandb_run_id"]
widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
print("  ".join(column.ljust(widths[column]) for column in columns))
for row in rows:
    print("  ".join(row[column].ljust(widths[column]) for column in columns))
PY

echo
echo "Detailed CSV: ${REPORT_DIR}/stage1_run_progress.csv"
echo "Detailed JSON: ${REPORT_DIR}/stage1_run_progress.json"
