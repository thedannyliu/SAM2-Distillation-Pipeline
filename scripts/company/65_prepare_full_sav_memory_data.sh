#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
RAW_ROOT="${RAW_ROOT:-${SAV_ROOT}/sav_train}"
CACHE_ROOT="${CACHE_ROOT:-${SAV_ROOT}/sav_train_6fps_full}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
COHORT_ROOT="${COHORT_ROOT:-${SAM2D_ROOT}/cohorts/sav_train_6fps_full}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-32}"
NUM_WORKERS="${NUM_WORKERS:-64}"
AUDIT_SAMPLE_VIDEOS="${AUDIT_SAMPLE_VIDEOS:-1000}"
RAW_AUDIT_JSON_SAMPLES="${RAW_AUDIT_JSON_SAMPLES:-500}"
RAW_AUDIT_VIDEO_SAMPLES="${RAW_AUDIT_VIDEO_SAMPLES:-200}"

describe() {
  echo "Full SA-V train preparation for SAM2 memory/task training"
  echo "Source: s3://sdp-ril/danny-dataset/SA-V/sav_train/"
  echo "Raw destination: ${RAW_ROOT}"
  echo "Frame cache: ${CACHE_ROOT}/JPEGImages"
  echo "Manifest: ${MANIFEST}"
  echo "Policy: retain every 6fps annotation-aligned train frame"
  echo "Expected raw release: 50,453 MP4; 50,452 manual JSON; 48,306 auto JSON"
}

sync_raw() {
  python tools/data/sync_sav_runtime_from_s3.py \
    --bucket sdp-ril \
    --source-root danny-dataset/SA-V \
    --out-root "${SAV_ROOT}" \
    --components sav_train \
    --workers "${DOWNLOAD_WORKERS}" \
    --file-retries 8 \
    --reserve-gib 100
}

audit_source_inventory() {
  python tools/data/sync_sav_runtime_from_s3.py \
    --bucket sdp-ril \
    --source-root danny-dataset/SA-V \
    --out-root "${SAV_ROOT}" \
    --components sav_train \
    --workers "${DOWNLOAD_WORKERS}" \
    --audit-only
}

audit_raw() {
  python - "${RAW_ROOT}" "${RAW_AUDIT_JSON_SAMPLES}" "${RAW_AUDIT_VIDEO_SAMPLES}" <<'PY'
import json
import random
import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1])
json_sample_count = int(sys.argv[2])
video_sample_count = int(sys.argv[3])
mp4 = sorted(root.rglob("*.mp4"))
manual = sorted(root.rglob("*_manual.json"))
auto = sorted(root.rglob("*_auto.json"))
counts = {"mp4": len(mp4), "manual_json": len(manual), "auto_json": len(auto)}
expected = {"mp4": 50453, "manual_json": 50452, "auto_json": 48306}
failures = [
    f"{name}: got {counts[name]}, expected {value}"
    for name, value in expected.items()
    if counts[name] != value
]
zero_size = [str(path) for path in mp4 + manual + auto if path.stat().st_size == 0]
if zero_size:
    failures.append(f"zero-size files: {zero_size[:10]}")

mp4_ids = [path.stem for path in mp4]
manual_ids = [path.name.removesuffix("_manual.json") for path in manual]
auto_ids = [path.name.removesuffix("_auto.json") for path in auto]
duplicate_ids = {
    "mp4": len(mp4_ids) - len(set(mp4_ids)),
    "manual_json": len(manual_ids) - len(set(manual_ids)),
    "auto_json": len(auto_ids) - len(set(auto_ids)),
}
if any(duplicate_ids.values()):
    failures.append(f"duplicate IDs: {duplicate_ids}")
orphan_manual = sorted(set(manual_ids) - set(mp4_ids))
orphan_auto = sorted(set(auto_ids) - set(mp4_ids))
if orphan_manual or orphan_auto:
    failures.append(
        f"annotations without MP4: manual={orphan_manual[:10]}, auto={orphan_auto[:10]}"
    )

rng = random.Random(310107256)
json_sample = rng.sample(manual, min(json_sample_count, len(manual)))
json_errors = []
for path in json_sample:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        masklet = payload.get("masklet")
        if not isinstance(masklet, list) or not masklet:
            raise ValueError("missing or empty masklet list")
    except Exception as exc:
        json_errors.append(f"{path}: {exc}")
if json_errors:
    failures.append(f"sampled JSON errors: {json_errors[:10]}")

video_sample = rng.sample(mp4, min(video_sample_count, len(mp4)))
video_errors = []
for path in video_sample:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("VideoCapture could not open file")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise ValueError(f"invalid frame count {frame_count}")
        for frame_index in sorted({0, min(160, frame_count - 1)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"could not decode frame {frame_index}")
    except Exception as exc:
        video_errors.append(f"{path}: {exc}")
    finally:
        capture.release()
if video_errors:
    failures.append(f"sampled MP4 errors: {video_errors[:10]}")

summary = {
    "root": str(root),
    "counts": counts,
    "expected": expected,
    "duplicate_ids": duplicate_ids,
    "mp4_without_manual_json": len(set(mp4_ids) - set(manual_ids)),
    "mp4_without_auto_json": len(set(mp4_ids) - set(auto_ids)),
    "manual_json_sampled": len(json_sample),
    "mp4_sampled": len(video_sample),
}
print(json.dumps(summary, indent=2))
if failures:
    print(json.dumps({"status": "fail", "failures": failures}, indent=2))
    raise SystemExit(1)
print(json.dumps({"status": "pass"}, indent=2))
PY
}

prepare_frames() {
  mkdir -p "$(dirname "${MANIFEST}")" "${CACHE_ROOT}"
  df -h "${CACHE_ROOT}"
  DATA_ROOT="/group-volume/danny-dataset" \
  SAV_ROOT="${SAV_ROOT}" \
  TRAIN_ROOT="${RAW_ROOT}" \
  OUT_ROOT="${CACHE_ROOT}" \
  MANIFEST="${MANIFEST}" \
  TRAIN_FRAMES_PER_VIDEO=1000000 \
  VAL_FRAMES_PER_VIDEO=0 \
  TEST_FRAMES_PER_VIDEO=0 \
  MAX_TRAIN_VIDEOS=0 \
  NUM_WORKERS="${NUM_WORKERS}" \
  JPEG_QUALITY=90 \
    scripts/company/18_prepare_sav_stage1_frame_cache.sh
}

audit_ready() {
  python - "${MANIFEST}" "${CACHE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

manifest = Path(sys.argv[1])
cache_root = Path(sys.argv[2]) / "JPEGImages"
frame = pd.read_parquet(manifest)
train = frame[frame["split"] == "train"].copy()
counts = train.groupby("video_id").size()
cached_images = sum(1 for _ in cache_root.rglob("*.jpg"))
summary = {
    "manifest": str(manifest),
    "train_rows": len(train),
    "train_videos": int(train["video_id"].nunique()),
    "cached_images": cached_images,
    "frames_per_video_min": int(counts.min()),
    "frames_per_video_median": float(counts.median()),
    "frames_per_video_max": int(counts.max()),
    "duplicate_sample_ids": int(train["sample_id"].duplicated().sum()),
    "off_6fps_cadence": int((train["frame_idx_24fps"] % 4 != 0).sum()),
}
print(json.dumps(summary, indent=2))
failures = []
if summary["train_videos"] != 50453:
    failures.append("manifest does not contain exactly 50,453 train videos")
if summary["train_rows"] < 807248:
    failures.append("full manifest has fewer rows than the previous 16-frame cache")
if summary["cached_images"] != summary["train_rows"]:
    failures.append("cached JPEG count differs from manifest row count")
if summary["frames_per_video_min"] < 16:
    failures.append("at least one train video has fewer than 16 decoded frames")
if summary["duplicate_sample_ids"]:
    failures.append("manifest contains duplicate sample IDs")
if summary["off_6fps_cadence"]:
    failures.append("manifest contains frames outside the 6fps annotation cadence")
if failures:
    print(json.dumps({"status": "fail", "failures": failures}, indent=2))
    raise SystemExit(1)
print(json.dumps({"status": "pass"}, indent=2))
PY

  python tools/train/audit_sam2_task_inputs.py \
    --manifest "${MANIFEST}" \
    --stage1-checkpoint "${SOURCE_STAGE1_CHECKPOINT}" \
    --sav-root "${SAV_ROOT}" \
    --sample-videos "${AUDIT_SAMPLE_VIDEOS}" \
    --compact
}

build_cohort() {
  local min_objects="$1"
  python tools/data/select_sav_dense_training_videos.py \
    --manifest "${MANIFEST}" \
    --sav-root "${SAV_ROOT}" \
    --output-video-ids "${COHORT_ROOT}/dense${min_objects}_unique.txt" \
    --out-csv "${COHORT_ROOT}/dense${min_objects}_index.csv" \
    --out-summary "${COHORT_ROOT}/dense${min_objects}_summary.json" \
    --min-objects "${min_objects}" \
    --min-dense-frames 4 \
    --target-samples 0 \
    --seed 310107256
}

build_cohorts() {
  mkdir -p "${COHORT_ROOT}"
  build_cohort 2 || return $?
  build_cohort 4 || return $?
  build_cohort 8
}

STATUS=0
case "${ACTION}" in
  describe)
    describe
    ;;
  sync)
    sync_raw || STATUS="$?"
    ;;
  source-audit)
    audit_source_inventory && audit_raw || STATUS="$?"
    ;;
  source-repair)
    sync_raw && audit_raw || STATUS="$?"
    ;;
  prepare)
    audit_raw && prepare_frames || STATUS="$?"
    ;;
  audit)
    audit_raw && audit_ready || STATUS="$?"
    ;;
  cohorts)
    build_cohorts || STATUS="$?"
    ;;
  all)
    sync_raw &&
      audit_raw &&
      prepare_frames &&
      audit_ready &&
      build_cohorts || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|sync|source-audit|source-repair|prepare|audit|cohorts|all}" >&2
    STATUS=2
    ;;
esac

echo "Full SA-V memory-data status: ${STATUS}"
echo "Manifest: ${MANIFEST}"
echo "Cohorts: ${COHORT_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
