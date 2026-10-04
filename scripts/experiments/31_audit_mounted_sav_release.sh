#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
REPORT="${REPORT:-/user-volume/sav_release_audit_${HOSTNAME}.json}"
NUM_WORKERS="${NUM_WORKERS:-64}"
DECODE_SAMPLES="${DECODE_SAMPLES:-200}"
FULL_DECODE="${FULL_DECODE:-0}"
EXPECTED_TEST_VIDEOS="${EXPECTED_TEST_VIDEOS:-150}"

[[ -d "${SAV_ROOT}" ]] || { echo "missing SA-V mount: ${SAV_ROOT}" >&2; exit 1; }

echo "===== Mounted filesystem ====="
findmnt -T "${SAV_ROOT}" || true
df -hT "${SAV_ROOT}"

echo "===== Dataset size ====="
du -sh \
  "${SAV_ROOT}/sav_train" \
  "${SAV_ROOT}/sav_val" \
  "${SAV_ROOT}/sav_test" \
  "${SAV_ROOT}/JPEGImages"

args=(
  --sav-root "${SAV_ROOT}"
  --report "${REPORT}"
  --workers "${NUM_WORKERS}"
  --decode-samples "${DECODE_SAMPLES}"
  --expected-test-videos "${EXPECTED_TEST_VIDEOS}"
)
if [[ "${FULL_DECODE}" == "1" ]]; then
  args+=(--decode-all)
fi

python tools/data/audit_mounted_sav_release.py "${args[@]}"
echo "SA-V release audit completed; inspect status and warnings above"
echo "report: ${REPORT}"
