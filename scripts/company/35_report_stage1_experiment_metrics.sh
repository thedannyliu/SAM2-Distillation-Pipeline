#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
RUNS_ROOTS="${RUNS_ROOTS:-/group-volume/danny-dataset/sam2_distill/runs:/mnt/data/danny-dataset/sam2_distill/runs}"
REPORT_DIR="${REPORT_DIR:-/user-volume/stage1_experiment_report_${HOSTNAME}}"

mkdir -p "${REPORT_DIR}"
SAV_ROOT="${SAV_ROOT}" \
RUNS_ROOTS="${RUNS_ROOTS}" \
REPORT_DIR="${REPORT_DIR}" \
  scripts/company/32_audit_stage1_run_progress.sh

python tools/experiments/summarize_stage1_experiment_metrics.py \
  --progress-json "${REPORT_DIR}/stage1_run_progress.json" \
  --out-dir "${REPORT_DIR}"

echo
echo "===== Incomplete runs ====="
python - "${REPORT_DIR}/incomplete_runs.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
columns = ["family", "queue", "name", "status", "step", "target_steps", "progress_pct", "next_action"]
if not rows:
    print("All discovered runs are complete.")
else:
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))
PY

echo
echo "Human-readable report: ${REPORT_DIR}/experiment_report.md"
echo "Key metrics CSV:       ${REPORT_DIR}/experiment_key_metrics.csv"
echo "Incomplete runs CSV:   ${REPORT_DIR}/incomplete_runs.csv"
