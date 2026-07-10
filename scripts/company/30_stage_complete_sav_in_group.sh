#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_SAV="${SOURCE_SAV:-/mnt/data/danny-dataset/SA-V}"
TARGET_SAV="${TARGET_SAV:-/group-volume/danny-dataset/SA-V}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps.parquet}"
REPORT="${REPORT:-${SAM2D_ROOT}/migration_reports/group_sav_completeness.json}"
READY_MARKER="${READY_MARKER:-${SAM2D_ROOT}/migration_reports/dataset_complete.ready}"
NUM_WORKERS="${NUM_WORKERS:-64}"

usage() {
  cat <<'EOF'
Usage:
  scripts/company/30_stage_complete_sav_in_group.sh preflight
  scripts/company/30_stage_complete_sav_in_group.sh sync-raw
  scripts/company/30_stage_complete_sav_in_group.sh materialize-val
  scripts/company/30_stage_complete_sav_in_group.sh audit
  scripts/company/30_stage_complete_sav_in_group.sh all

This stages a complete raw SA-V train/val/test copy in group-volume and verifies
the existing 807,248-frame Stage 1 cache. It does not transfer to data lake and
does not delete any source files.
EOF
}

check_source() {
  for split in sav_train sav_val sav_test; do
    [[ -d "${SOURCE_SAV}/${split}" ]] || {
      echo "missing source split: ${SOURCE_SAV}/${split}" >&2
      exit 1
    }
  done
}

preflight() {
  check_source
  mkdir -p "${TARGET_SAV}" "$(dirname "${REPORT}")"
  echo "===== Filesystems ====="
  df -hT "${SOURCE_SAV}" "${TARGET_SAV}"
  echo "===== Source sizes ====="
  du -sh "${SOURCE_SAV}/sav_train" "${SOURCE_SAV}/sav_val" "${SOURCE_SAV}/sav_test"
  echo "===== Target sizes before sync ====="
  du -sh "${TARGET_SAV}/sav_train" "${TARGET_SAV}/sav_val" "${TARGET_SAV}/sav_test" 2>/dev/null || true
  echo "===== Critical source sentinels ====="
  test -f "${SOURCE_SAV}/sav_val/JPEGImages_24fps/sav_000262/00060.jpg"
  test -f "${SOURCE_SAV}/sav_test/Annotations_6fps/sav_013624/000/00000.png"
  ls -lh \
    "${SOURCE_SAV}/sav_val/JPEGImages_24fps/sav_000262/00060.jpg" \
    "${SOURCE_SAV}/sav_test/Annotations_6fps/sav_013624/000/00000.png"
  echo "preflight: PASS"
}

sync_raw() {
  preflight
  mkdir -p "${TARGET_SAV}/sav_train" "${TARGET_SAV}/sav_val" "${TARGET_SAV}/sav_test"
  for split in sav_train sav_val sav_test; do
    echo "===== Sync ${split} ====="
    rsync -aH --partial --info=progress2 \
      "${SOURCE_SAV}/${split}/" "${TARGET_SAV}/${split}/"
  done
}

materialize_val() {
  [[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }
  check_source
  cp -p "${MANIFEST}" "${MANIFEST}.before_group_dataset_gate"
  DATA_ROOT=/group-volume/danny-dataset \
  SAV_ROOT="${TARGET_SAV}" \
  CACHE_NAME=stage1_vbal16_6fps \
  MANIFEST="${MANIFEST}" \
  REUSE_TRAIN_MANIFEST="${MANIFEST}" \
  TRAIN_FRAMES_PER_VIDEO=16 \
  VAL_FRAMES_PER_VIDEO=8 \
  TEST_FRAMES_PER_VIDEO=0 \
  NUM_WORKERS="${NUM_WORKERS}" \
    scripts/company/18_prepare_sav_stage1_frame_cache.sh
  touch "${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps.done"
}

audit() {
  check_source
  python tools/data/audit_sav_dataset_copy.py \
    --source-root "${SOURCE_SAV}" \
    --target-root "${TARGET_SAV}" \
    --manifest "${MANIFEST}" \
    --report "${REPORT}" \
    --ready-marker "${READY_MARKER}" \
    --workers "${NUM_WORKERS}"
  echo "ready marker: ${READY_MARKER}"
}

case "${1:-}" in
  preflight) preflight ;;
  sync-raw) sync_raw ;;
  materialize-val) materialize_val ;;
  audit) audit ;;
  all) sync_raw; materialize_val; audit ;;
  -h|--help|"") usage ;;
  *) usage; exit 2 ;;
esac
