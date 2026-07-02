#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_ROOT="${SMOKE_ROOT:-${ROOT}/runs/edgetam_smoke}"
SMOKE_DATA_ROOT="${SMOKE_DATA_ROOT:-${ROOT}/data/edgetam_smoke}"
SA1B_SOURCE="${SA1B_SOURCE:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd/data/SA-1B-1P}"
SAV_SOURCE="${SAV_SOURCE:-/storage/project/r-agarg35-0/eliu354/projects/efficientsam3-benchmark/data/sa-v/sav_val_fixed10}"
COCO_SOURCE="${COCO_SOURCE:-/storage/project/r-agarg35-0/eliu354/projects/efficientsam3-benchmark/data/coco}"
SAV_EVALUATOR="${SAV_EVALUATOR:-}"
EDGETAM_ROOT="${EDGETAM_ROOT:-${ROOT}/third_party/EdgeTAM}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${ROOT}/checkpoints/edgetam.pt}"
EDGETAM_CFG="${EDGETAM_CFG:-configs/edgetam.yaml}"
EDGETAM_IMAGE="${EDGETAM_IMAGE:-${SMOKE_DATA_ROOT}/coco_smoke/images/val2017/000000000139.jpg}"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-${ROOT}/third_party/sam2}"
MAX_ITEMS="${MAX_ITEMS:-500}"
MAX_FRAMES="${MAX_FRAMES:-500}"

usage() {
  cat <<'EOF'
Usage:
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh prepare-data
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh validate-data
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh probe-tinyvit
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh write-config
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh validate-train-config
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh stage1-feature-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh sav-eval-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh vos-style-data
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh vos-style-eval-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh davis-data
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh davis-eval-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh video-mask-train-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-distill-loss-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-resume-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-cache-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-teacher-cache-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-forward-cache-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-trainer-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-forward-cache-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-progressive-full-trainer-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh progressive-video-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-benchmark
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-vos-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-vos-track-later-smoke
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-sav-eval
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh all-cpu

This script prepares <=500-item PACE smoke subsets only. GPU train/eval
commands should be submitted through Slurm using embers QOS.

Defaults:
  SMOKE_DATA_ROOT=${ROOT}/data/edgetam_smoke
  SMOKE_ROOT=${ROOT}/runs/edgetam_smoke
EOF
}

prepare_data() {
  python "${ROOT}/tools/data/make_edgetam_smoke_subset.py" sa1b \
    --source-root "${SA1B_SOURCE}" \
    --out-root "${SMOKE_DATA_ROOT}/sa1b_smoke" \
    --max-items "${MAX_ITEMS}" \
    --train-count 400 \
    --val-count 100

  python "${ROOT}/tools/data/make_edgetam_smoke_subset.py" sav-val \
    --source-root "${SAV_SOURCE}" \
    --out-root "${SMOKE_DATA_ROOT}/sav_val_smoke" \
    --max-frames "${MAX_FRAMES}"

  python "${ROOT}/tools/data/make_edgetam_smoke_subset.py" coco \
    --source-root "${COCO_SOURCE}" \
    --out-root "${SMOKE_DATA_ROOT}/coco_smoke" \
    --max-items "${MAX_ITEMS}"
}

validate_data() {
  python "${ROOT}/tools/data/validate_edgetam_smoke_subset.py" \
    --image-manifest "${SMOKE_DATA_ROOT}/sa1b_smoke/manifests/sa1b_smoke_manifest.jsonl" \
    --image-manifest "${SMOKE_DATA_ROOT}/coco_smoke/manifests/coco_smoke_manifest.jsonl" \
    --sav-val-root "${SMOKE_DATA_ROOT}/sav_val_smoke" \
    --max-items "${MAX_ITEMS}" \
    --max-frames "${MAX_FRAMES}"
}

probe_tinyvit() {
  python "${ROOT}/tools/edgetam/probe_tinyvit_backbone.py" \
    --model-name tiny_vit_21m_512.dist_in22k_ft_in1k
}

write_config() {
  python "${ROOT}/tools/edgetam/write_tinyvit_edgetam_config.py" \
    --out "${SMOKE_ROOT}/configs/edgetam_tinyvit21m.yaml"
}

validate_train_config() {
  local model_args=()
  if [[ "${VALIDATE_TRAIN_CONFIG_INSTANTIATE_MODEL:-0}" == "1" ]]; then
    model_args=(--instantiate-model)
  fi
  python "${ROOT}/tools/edgetam/validate_training_config.py" \
    --config "${ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml" \
    --sam2-training-root "${SAM2_TRAINING_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-json "${SMOKE_ROOT}/validate_train_config/summary.json" \
    --instantiate-loss \
    "${model_args[@]}"
}

stage1_feature_smoke() {
  python "${ROOT}/tools/train/smoke_stage1_features.py" \
    --manifest "${SMOKE_DATA_ROOT}/sa1b_smoke/manifests/sa1b_smoke_manifest.jsonl" \
    --out-dir "${SMOKE_ROOT}/stage1_feature_smoke" \
    --max-items "${STAGE1_SMOKE_MAX_ITEMS:-16}" \
    --steps "${STAGE1_SMOKE_STEPS:-2}" \
    --batch-size "${STAGE1_SMOKE_BATCH_SIZE:-1}" \
    --image-size "${STAGE1_SMOKE_IMAGE_SIZE:-512}"
}

sav_eval_smoke() {
  local evaluator_args=()
  if [[ -n "${SAV_EVALUATOR}" ]]; then
    evaluator_args=(--evaluator "${SAV_EVALUATOR}" --num-processes "${SAV_EVAL_NUM_PROCESSES:-2}")
  fi

  python "${ROOT}/tools/eval/sav_identity_smoke.py" \
    --sav-root "${SMOKE_DATA_ROOT}/sav_val_smoke" \
    --filelist "${SMOKE_DATA_ROOT}/sav_val_smoke/sav_val.txt" \
    --out-dir "${SMOKE_ROOT}/sav_identity_pred" \
    --max-frames "${MAX_FRAMES}" \
    "${evaluator_args[@]}"
}

vos_style_data() {
  python "${ROOT}/tools/data/make_vos_smoke_subset.py" sav-to-davis-style \
    --sav-root "${SMOKE_DATA_ROOT}/sav_val_smoke" \
    --out-root "${SMOKE_DATA_ROOT}/vos_style_from_sav_smoke" \
    --max-frames "${MAX_FRAMES}"
}

vos_style_eval_smoke() {
  python "${ROOT}/tools/eval/vos_identity_smoke.py" \
    --annotation-root "${SMOKE_DATA_ROOT}/vos_style_from_sav_smoke/Annotations" \
    --out-dir "${SMOKE_ROOT}/vos_style_identity_pred" \
    --max-frames "${MAX_FRAMES}"
}

davis_data() {
  python "${ROOT}/tools/data/extract_davis_zip_smoke_subset.py" \
    --zip "${DAVIS_ZIP:-${SMOKE_DATA_ROOT}/_downloads/DAVIS-2017-trainval-480p.zip}" \
    --out-root "${SMOKE_DATA_ROOT}/davis2017_smoke" \
    --max-frames "${DAVIS_MAX_FRAMES:-${MAX_FRAMES}}"
}

davis_eval_smoke() {
  python "${ROOT}/tools/eval/vos_identity_smoke.py" \
    --annotation-root "${SMOKE_DATA_ROOT}/davis2017_smoke/Annotations" \
    --out-dir "${SMOKE_ROOT}/davis2017_identity_pred" \
    --max-frames "${DAVIS_MAX_FRAMES:-${MAX_FRAMES}}"
}

video_mask_train_smoke() {
  python "${ROOT}/tools/train/smoke_video_masks.py" \
    --manifest "${SMOKE_DATA_ROOT}/vos_style_from_sav_smoke/manifests/vos_smoke_manifest.jsonl" \
    --out-dir "${SMOKE_ROOT}/video_mask_train_smoke" \
    --max-clips "${VIDEO_SMOKE_MAX_CLIPS:-8}" \
    --steps "${VIDEO_SMOKE_STEPS:-2}" \
    --batch-size "${VIDEO_SMOKE_BATCH_SIZE:-1}" \
    --clip-frames "${VIDEO_SMOKE_CLIP_FRAMES:-4}" \
    --image-size "${VIDEO_SMOKE_IMAGE_SIZE:-128}" \
    --phase-name "${VIDEO_SMOKE_PHASE_NAME:-video_mask_train_smoke}" \
    ${VIDEO_SMOKE_RESUME:+--resume}
}

edgetam_distill_loss_smoke() {
  python "${ROOT}/tools/train/smoke_edgetam_distill_loss.py" \
    --sam2-training-root "${SAM2_TRAINING_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-json "${SMOKE_ROOT}/edgetam_distill_loss_smoke/summary.json" \
    --frames "${EDGETAM_DISTILL_SMOKE_FRAMES:-2}" \
    --objects "${EDGETAM_DISTILL_SMOKE_OBJECTS:-1}" \
    --masks "${EDGETAM_DISTILL_SMOKE_MASKS:-3}"
}

edgetam_full_trainer_smoke() {
  local activation_checkpoint_args=()
  local teacher_cache_args=()
  local freeze_args=()
  local lambda_args=()
  if [[ "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT:-0}" == "1" ]]; then
    activation_checkpoint_args=(--image-encoder-activation-checkpoint)
  fi
  if [[ -n "${EDGETAM_TRAINER_SMOKE_TEACHER_CACHE:-}" ]]; then
    teacher_cache_args=(--teacher-feature-cache "${EDGETAM_TRAINER_SMOKE_TEACHER_CACHE}")
  fi
  if [[ "${EDGETAM_TRAINER_SMOKE_FREEZE_IMAGE_ENCODER:-0}" == "1" ]]; then
    freeze_args=(--freeze-image-encoder)
  fi
  if [[ -n "${EDGETAM_TRAINER_SMOKE_LAMBDA_IMG:-}" ]]; then
    lambda_args+=(--lambda-img "${EDGETAM_TRAINER_SMOKE_LAMBDA_IMG}")
  fi
  if [[ -n "${EDGETAM_TRAINER_SMOKE_LAMBDA_MEM:-}" ]]; then
    lambda_args+=(--lambda-mem "${EDGETAM_TRAINER_SMOKE_LAMBDA_MEM}")
  fi
  python "${ROOT}/tools/train/run_edgetam_trainer_smoke.py" \
    --config "${ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml" \
    --sam2-training-root "${SAM2_TRAINING_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-dir "${EDGETAM_TRAINER_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_full_trainer_smoke}" \
    --max-epochs "${EDGETAM_TRAINER_SMOKE_EPOCHS:-1}" \
    --num-workers "${EDGETAM_TRAINER_SMOKE_WORKERS:-0}" \
    --num-frames "${EDGETAM_TRAINER_SMOKE_FRAMES:-8}" \
    --max-num-objects "${EDGETAM_TRAINER_SMOKE_OBJECTS:-1}" \
    --image-encoder-forward-batch-size "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH:-0}" \
    "${activation_checkpoint_args[@]}" \
    "${freeze_args[@]}" \
    "${lambda_args[@]}" \
    "${teacher_cache_args[@]}"
}

edgetam_full_trainer_resume_smoke() {
  local out_dir="${EDGETAM_TRAINER_RESUME_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_full_trainer_resume_smoke}"
  rm -rf "${out_dir}"
  EDGETAM_TRAINER_SMOKE_OUT_DIR="${out_dir}" \
  EDGETAM_TRAINER_SMOKE_EPOCHS=1 \
  edgetam_full_trainer_smoke
  EDGETAM_TRAINER_SMOKE_OUT_DIR="${out_dir}" \
  EDGETAM_TRAINER_SMOKE_EPOCHS=2 \
  edgetam_full_trainer_smoke
}

edgetam_full_trainer_cache_smoke() {
  local out_dir="${EDGETAM_TRAINER_CACHE_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_full_trainer_cache_smoke}"
  local frames="${EDGETAM_TRAINER_SMOKE_FRAMES:-2}"
  local cache="${EDGETAM_TRAINER_CACHE_SMOKE_CACHE:-${out_dir}/teacher_cache.pt}"
  python "${ROOT}/tools/train/make_teacher_feature_cache_smoke.py" \
    --out "${cache}" \
    --frames "${frames}"
  EDGETAM_TRAINER_SMOKE_OUT_DIR="${out_dir}" \
  EDGETAM_TRAINER_SMOKE_TEACHER_CACHE="${cache}" \
  edgetam_full_trainer_smoke
}

edgetam_teacher_cache_smoke() {
  local out_dir="${EDGETAM_TEACHER_CACHE_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_teacher_cache_smoke}"
  local frames="${EDGETAM_TRAINER_SMOKE_FRAMES:-2}"
  local cache="${EDGETAM_TEACHER_CACHE_SMOKE_CACHE:-${out_dir}/teacher_forward_cache.pt}"
  local activation_checkpoint_args=()
  if [[ "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT:-0}" == "1" ]]; then
    activation_checkpoint_args=(--image-encoder-activation-checkpoint)
  fi
  python "${ROOT}/tools/train/cache_edgetam_teacher_features.py" \
    --config "${ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml" \
    --sam2-training-root "${SAM2_TRAINING_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out "${cache}" \
    --work-dir "${out_dir}" \
    --num-workers 0 \
    --num-frames "${frames}" \
    --max-num-objects "${EDGETAM_TRAINER_SMOKE_OBJECTS:-1}" \
    --image-encoder-forward-batch-size "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH:-0}" \
    "${activation_checkpoint_args[@]}"
}

edgetam_full_trainer_forward_cache_smoke() {
  local cache_dir="${EDGETAM_TEACHER_CACHE_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_teacher_cache_smoke}"
  local trainer_dir="${EDGETAM_TRAINER_FORWARD_CACHE_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_full_trainer_forward_cache_smoke}"
  local cache="${EDGETAM_TEACHER_CACHE_SMOKE_CACHE:-${cache_dir}/teacher_forward_cache.pt}"
  EDGETAM_TEACHER_CACHE_SMOKE_OUT_DIR="${cache_dir}" \
  EDGETAM_TEACHER_CACHE_SMOKE_CACHE="${cache}" \
  edgetam_teacher_cache_smoke
  EDGETAM_TRAINER_SMOKE_OUT_DIR="${trainer_dir}" \
  EDGETAM_TRAINER_SMOKE_TEACHER_CACHE="${cache}" \
  edgetam_full_trainer_smoke
}

edgetam_image_trainer_smoke() {
  local activation_checkpoint_args=()
  local teacher_cache_args=()
  if [[ "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT:-0}" == "1" ]]; then
    activation_checkpoint_args=(--image-encoder-activation-checkpoint)
  fi
  if [[ -n "${EDGETAM_IMAGE_TRAINER_SMOKE_TEACHER_CACHE:-}" ]]; then
    teacher_cache_args=(--teacher-feature-cache "${EDGETAM_IMAGE_TRAINER_SMOKE_TEACHER_CACHE}")
  fi
  python "${ROOT}/tools/train/run_edgetam_trainer_smoke.py" \
    --config "${ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml" \
    --sam2-training-root "${SAM2_TRAINING_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out-dir "${EDGETAM_IMAGE_TRAINER_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_image_trainer_smoke}" \
    --max-epochs 1 \
    --num-workers 0 \
    --dataset-mode sa1b-image \
    --sa1b-image-root "${EDGETAM_SA1B_IMAGE_ROOT:-${SMOKE_DATA_ROOT}/sa1b_smoke/images/train}" \
    --sa1b-ann-root "${EDGETAM_SA1B_ANN_ROOT:-${SMOKE_DATA_ROOT}/sa1b_smoke/annotations/train}" \
    --sa1b-max-items "${EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS:-2}" \
    --num-frames 1 \
    --max-num-objects "${EDGETAM_IMAGE_TRAINER_SMOKE_OBJECTS:-4}" \
    --image-encoder-forward-batch-size "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH:-0}" \
    "${activation_checkpoint_args[@]}" \
    "${teacher_cache_args[@]}"
}

edgetam_image_forward_cache_smoke() {
  local cache_dir="${EDGETAM_IMAGE_FORWARD_CACHE_SMOKE_OUT_DIR:-${SMOKE_ROOT}/edgetam_image_forward_cache_smoke}"
  local trainer_dir="${EDGETAM_IMAGE_FORWARD_CACHE_TRAINER_OUT_DIR:-${SMOKE_ROOT}/edgetam_image_forward_cache_trainer_smoke}"
  local cache="${EDGETAM_IMAGE_FORWARD_CACHE_SMOKE_CACHE:-${cache_dir}/teacher_forward_cache.pt}"
  local activation_checkpoint_args=()
  if [[ "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT:-0}" == "1" ]]; then
    activation_checkpoint_args=(--image-encoder-activation-checkpoint)
  fi
  python "${ROOT}/tools/train/cache_edgetam_teacher_features.py" \
    --config "${ROOT}/configs/edgetam/tinyvit_video_distill_smoke.yaml" \
    --sam2-training-root "${SAM2_TRAINING_ROOT}" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --out "${cache}" \
    --work-dir "${cache_dir}" \
    --num-workers 0 \
    --dataset-mode sa1b-image \
    --sa1b-image-root "${EDGETAM_SA1B_IMAGE_ROOT:-${SMOKE_DATA_ROOT}/sa1b_smoke/images/train}" \
    --sa1b-ann-root "${EDGETAM_SA1B_ANN_ROOT:-${SMOKE_DATA_ROOT}/sa1b_smoke/annotations/train}" \
    --sa1b-max-items "${EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS:-1}" \
    --num-frames 1 \
    --max-num-objects "${EDGETAM_IMAGE_TRAINER_SMOKE_OBJECTS:-4}" \
    --image-encoder-forward-batch-size "${EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH:-0}" \
    "${activation_checkpoint_args[@]}"
  EDGETAM_IMAGE_TRAINER_SMOKE_OUT_DIR="${trainer_dir}" \
  EDGETAM_IMAGE_TRAINER_SMOKE_TEACHER_CACHE="${cache}" \
  EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS="${EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS:-1}" \
  edgetam_image_trainer_smoke
}

edgetam_progressive_full_trainer_smoke() {
  local frames_list="${EDGETAM_PROGRESSIVE_FULL_FRAMES:-2 4 8}"
  local idx=0
  for frames in ${frames_list}; do
    idx=$((idx + 1))
    local out_dir="${SMOKE_ROOT}/edgetam_progressive_full_trainer_smoke/phase_${idx}_${frames}f"
    if [[ "${idx}" == "1" ]]; then
      EDGETAM_TRAINER_SMOKE_OUT_DIR="${out_dir}" \
      EDGETAM_TRAINER_SMOKE_FRAMES="${frames}" \
      EDGETAM_TRAINER_SMOKE_OBJECTS="${EDGETAM_PROGRESSIVE_FULL_OBJECTS:-1}" \
      EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH="${EDGETAM_PROGRESSIVE_FULL_IMAGE_ENCODER_BATCH:-1}" \
      EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT="${EDGETAM_PROGRESSIVE_FULL_IMAGE_ENCODER_CKPT:-1}" \
      EDGETAM_TRAINER_SMOKE_FREEZE_IMAGE_ENCODER=0 \
      EDGETAM_TRAINER_SMOKE_LAMBDA_IMG="${EDGETAM_PROGRESSIVE_FULL_LAMBDA_IMG_PHASE1:-0.5}" \
      EDGETAM_TRAINER_SMOKE_LAMBDA_MEM="${EDGETAM_PROGRESSIVE_FULL_LAMBDA_MEM_PHASE1:-0.25}" \
      edgetam_full_trainer_smoke
    else
      EDGETAM_TRAINER_SMOKE_OUT_DIR="${out_dir}" \
      EDGETAM_TRAINER_SMOKE_FRAMES="${frames}" \
      EDGETAM_TRAINER_SMOKE_OBJECTS="${EDGETAM_PROGRESSIVE_FULL_OBJECTS:-1}" \
      EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH="${EDGETAM_PROGRESSIVE_FULL_IMAGE_ENCODER_BATCH:-1}" \
      EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT="${EDGETAM_PROGRESSIVE_FULL_IMAGE_ENCODER_CKPT:-1}" \
      EDGETAM_TRAINER_SMOKE_FREEZE_IMAGE_ENCODER=1 \
      EDGETAM_TRAINER_SMOKE_LAMBDA_IMG=0 \
      EDGETAM_TRAINER_SMOKE_LAMBDA_MEM=0 \
      edgetam_full_trainer_smoke
    fi
  done
}

progressive_video_smoke() {
  for frames in 8 16 32; do
    VIDEO_SMOKE_CLIP_FRAMES="${frames}" \
    VIDEO_SMOKE_IMAGE_SIZE="${PROGRESSIVE_SMOKE_IMAGE_SIZE:-64}" \
    VIDEO_SMOKE_MAX_CLIPS="${PROGRESSIVE_SMOKE_MAX_CLIPS:-2}" \
    VIDEO_SMOKE_STEPS="${PROGRESSIVE_SMOKE_STEPS:-1}" \
    VIDEO_SMOKE_PHASE_NAME="progressive_${frames}f" \
    python "${ROOT}/tools/train/smoke_video_masks.py" \
      --manifest "${SMOKE_DATA_ROOT}/vos_style_from_sav_smoke/manifests/vos_smoke_manifest.jsonl" \
      --out-dir "${SMOKE_ROOT}/progressive_video_smoke/${frames}f" \
      --max-clips "${PROGRESSIVE_SMOKE_MAX_CLIPS:-2}" \
      --steps "${PROGRESSIVE_SMOKE_STEPS:-1}" \
      --batch-size "${VIDEO_SMOKE_BATCH_SIZE:-1}" \
      --clip-frames "${frames}" \
      --image-size "${PROGRESSIVE_SMOKE_IMAGE_SIZE:-64}" \
      --phase-name "progressive_${frames}f" \
      --freeze-image-encoder \
      --teacher-disabled \
      --distill-disabled
  done
}

edgetam_vos_smoke() {
  local out_dir="${EDGETAM_VOS_OUT_DIR:-${SMOKE_ROOT}/edgetam_vos_pred}"
  local extra_args=()
  if [[ "${EDGETAM_VOS_TRACK_LATER:-0}" == "1" ]]; then
    extra_args+=(--track-object-appearing-later-in-video)
  fi
  python "${ROOT}/tools/eval/run_edgetam_vos_smoke.py" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --sam2-cfg "${EDGETAM_CFG}" \
    --checkpoint "${EDGETAM_CHECKPOINT}" \
    --sav-root "${SMOKE_DATA_ROOT}/sav_val_smoke" \
    --video-list-file "${SMOKE_DATA_ROOT}/sav_val_smoke/sav_val.txt" \
    --out-dir "${out_dir}" \
    "${extra_args[@]}"
}

edgetam_vos_track_later_smoke() {
  EDGETAM_VOS_TRACK_LATER=1 \
  EDGETAM_VOS_OUT_DIR="${EDGETAM_VOS_OUT_DIR:-${SMOKE_ROOT}/edgetam_vos_track_later_pred}" \
  edgetam_vos_smoke
}

edgetam_image_smoke() {
  python "${ROOT}/tools/eval/run_edgetam_image_smoke.py" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --sam2-cfg "${EDGETAM_CFG}" \
    --checkpoint "${EDGETAM_CHECKPOINT}" \
    --image "${EDGETAM_IMAGE}" \
    --out-dir "${SMOKE_ROOT}/edgetam_image_smoke"
}

edgetam_image_benchmark() {
  python "${ROOT}/tools/benchmark/benchmark_edgetam_image_predictor.py" \
    --edgetam-root "${EDGETAM_ROOT}" \
    --sam2-cfg "${EDGETAM_CFG}" \
    --checkpoint "${EDGETAM_CHECKPOINT}" \
    --manifest "${SMOKE_DATA_ROOT}/coco_smoke/manifests/coco_smoke_manifest.jsonl" \
    --out-dir "${SMOKE_ROOT}/edgetam_image_benchmark" \
    --limit "${EDGETAM_BENCH_LIMIT:-4}" \
    --warmup "${EDGETAM_BENCH_WARMUP:-1}" \
    --iters "${EDGETAM_BENCH_ITERS:-4}"
}

edgetam_sav_eval() {
  if [[ -z "${SAV_EVALUATOR}" ]]; then
    echo "SAV_EVALUATOR must point to sam2/sav_dataset/sav_evaluator.py" >&2
    exit 2
  fi
  local pred_root="${EDGETAM_VOS_OUT_DIR:-${SMOKE_ROOT}/edgetam_vos_pred}"
  python "${ROOT}/tools/eval/run_sav_evaluator.py" \
    --evaluator "${SAV_EVALUATOR}" \
    --gt-root "${SMOKE_DATA_ROOT}/sav_val_smoke/Annotations_6fps" \
    --pred-root "${pred_root}" \
    --out-json "${pred_root}/eval_summary.json" \
    --num-processes "${SAV_EVAL_NUM_PROCESSES:-2}" \
    --strict
}

case "${1:-}" in
  prepare-data)
    prepare_data
    ;;
  validate-data)
    validate_data
    ;;
  probe-tinyvit)
    probe_tinyvit
    ;;
  write-config)
    write_config
    ;;
  validate-train-config)
    validate_train_config
    ;;
  stage1-feature-smoke)
    stage1_feature_smoke
    ;;
  sav-eval-smoke)
    sav_eval_smoke
    ;;
  vos-style-data)
    vos_style_data
    ;;
  vos-style-eval-smoke)
    vos_style_eval_smoke
    ;;
  davis-data)
    davis_data
    ;;
  davis-eval-smoke)
    davis_eval_smoke
    ;;
  video-mask-train-smoke)
    video_mask_train_smoke
    ;;
  edgetam-distill-loss-smoke)
    edgetam_distill_loss_smoke
    ;;
  edgetam-full-trainer-smoke)
    edgetam_full_trainer_smoke
    ;;
  edgetam-full-trainer-resume-smoke)
    edgetam_full_trainer_resume_smoke
    ;;
  edgetam-full-trainer-cache-smoke)
    edgetam_full_trainer_cache_smoke
    ;;
  edgetam-teacher-cache-smoke)
    edgetam_teacher_cache_smoke
    ;;
  edgetam-full-trainer-forward-cache-smoke)
    edgetam_full_trainer_forward_cache_smoke
    ;;
  edgetam-image-trainer-smoke)
    edgetam_image_trainer_smoke
    ;;
  edgetam-image-forward-cache-smoke)
    edgetam_image_forward_cache_smoke
    ;;
  edgetam-progressive-full-trainer-smoke)
    edgetam_progressive_full_trainer_smoke
    ;;
  progressive-video-smoke)
    progressive_video_smoke
    ;;
  edgetam-image-smoke)
    edgetam_image_smoke
    ;;
  edgetam-image-benchmark)
    edgetam_image_benchmark
    ;;
  edgetam-vos-smoke)
    edgetam_vos_smoke
    ;;
  edgetam-vos-track-later-smoke)
    edgetam_vos_track_later_smoke
    ;;
  edgetam-sav-eval)
    edgetam_sav_eval
    ;;
  all-cpu)
    prepare_data
    validate_data
    probe_tinyvit
    write_config
    ;;
  *)
    usage
    exit 2
    ;;
esac
