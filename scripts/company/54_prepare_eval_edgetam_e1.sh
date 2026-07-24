#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-all}"
case "${ACTION}" in
  prepare|eval|all) ;;
  *)
    echo "Usage: $0 {prepare|eval|all}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
BEHAVIOR_ROOT="${EDGETAM_BEHAVIOR_ROOT:-${SAM2D_ROOT}/runs/edgetam_tinyvit21_behavior_v4}"
NAME=E1_a02_official_nonimage
RUN_DIR="${BEHAVIOR_ROOT}/${NAME}/main"
SAM2_TRAINING_ROOT="${SAM2_TRAINING_ROOT:-/user-volume/repo/facebookresearch-sam2}"
EDGETAM_ROOT="${EDGETAM_ROOT:-/user-volume/repo/EdgeTAM}"
EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/edgetam/edgetam.pt}"
A02_CHECKPOINT="${A02_CHECKPOINT:-${SAM2D_ROOT}/runs/sam2_mask_finetune_ablation_v2/A02_e2e_t4_official_prompt/main/checkpoints/checkpoint.pt}"
TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT:-${SAM2D_ROOT}/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors}"
SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT:-${SAM2D_ROOT}/runs/sav_stage1_ablation_v2/4gpu_adapter_teacher/tv21_proj_sam21l_msehr_l1_025/checkpoints/best.pt}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_group_runtime.parquet}"
CONFIG="${CONFIG:-configs/sam2_task/tv21_sav_progressive.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-edgetam-tinyvit21-behavior-v4}"
WANDB_MODE="${WANDB_MODE:-online}"
SKIP_DONE="${EDGETAM_MEMORY_SKIP_DONE:-1}"

if [[ -z "${SAV_ROOT:-}" ]]; then
  for candidate in \
    /group-volume/danny-dataset/SA-V \
    /mnt/data/danny-dataset/SA-V \
    /danny-dataset/SA-V; do
    if [[ -f "${candidate}/sav_val/sav_val.txt" && \
          -f "${candidate}/sav_test/sav_test.txt" ]]; then
      SAV_ROOT="${candidate}"
      break
    fi
  done
fi
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"

require_path() {
  [[ -e "$1" ]] || {
    echo "[ERROR] Missing required path: $1" >&2
    return 1
  }
}

validate_inputs() {
  local path
  for path in \
    "${A02_CHECKPOINT}" \
    "${EDGETAM_CHECKPOINT}" \
    "${EDGETAM_ROOT}/sam2/modeling/perceiver.py" \
    "${SAM2_TRAINING_ROOT}/training/model/sam2.py" \
    "${TINYVIT_CHECKPOINT}" \
    "${SOURCE_STAGE1_CHECKPOINT}" \
    "${MANIFEST}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAV_ROOT}/sav_test/sav_test.txt" \
    "${CONFIG}"; do
    require_path "${path}" || return 1
  done
}

prepare_checkpoint() {
  mkdir -p "${RUN_DIR}/checkpoints"
  exec 7>"${BEHAVIOR_ROOT}/.e1_prepare.lock" || return 1
  flock 7 || return 1
  if [[ ! -f "${RUN_DIR}/checkpoints/last.pt" ]]; then
    python tools/train/export_edgetam_tinyvit_transplant.py \
      --tinyvit-task-checkpoint "${A02_CHECKPOINT}" \
      --official-edgetam-checkpoint "${EDGETAM_CHECKPOINT}" \
      --output "${RUN_DIR}/checkpoints/last.pt" \
      --summary "${RUN_DIR}/initialization_summary.json" \
      --name "${NAME}" || return 1
  fi
  ln -sfn last.pt "${RUN_DIR}/checkpoints/best.pt"
  ln -sfn last.pt "${RUN_DIR}/checkpoints/checkpoint.pt"

  TASK_EDGETAM_MEMORY_ABLATION=1 \
  TASK_MEMORY_TOPOLOGY=edgetam_hybrid2 \
  TASK_MEMORY_LAYOUT=official \
  TASK_MEMORY_INITIALIZER=current_full \
  TASK_MEMORY_LAYERS=2 \
  TASK_NUM_GLOBAL_LATENTS=256 \
  TASK_NUM_2D_LATENTS=256 \
  TASK_TRAINABLE_MODE=memory_perceiver_full \
  TASK_FREEZE_BATCHNORM=true \
  TASK_EPOCHS=1 \
  TASK_NUM_FRAMES=4 \
  TASK_NUM_WORKERS=8 \
  TASK_ENCODER_LR=0 \
  TASK_ENCODER_LR_END=0 \
  TASK_HEAD_LR=1.0e-7 \
  TASK_HEAD_LR_END=1.0e-8 \
  TASK_MEMORY_LR=3.0e-7 \
  TASK_MEMORY_LR_END=3.0e-8 \
  TASK_MEMORY_AUX_LR=1.0e-7 \
  TASK_MEMORY_AUX_LR_END=1.0e-8 \
  TASK_PERCEIVER_LR=1.0e-6 \
  TASK_PERCEIVER_LR_END=1.0e-7 \
  PREVIOUS_TASK_CHECKPOINT="${RUN_DIR}/checkpoints/last.pt" \
  TASK_RUN_DIR="${RUN_DIR}" \
  TASK_MANIFEST="${MANIFEST}" \
  SAV_ROOT="${SAV_ROOT}" \
  SOURCE_STAGE1_CHECKPOINT="${SOURCE_STAGE1_CHECKPOINT}" \
  EDGETAM_CHECKPOINT="${EDGETAM_CHECKPOINT}" \
  TINYVIT_CHECKPOINT="${TINYVIT_CHECKPOINT}" \
  TINYVIT_MODEL_NAME=tiny_vit_21m_512.dist_in22k_ft_in1k \
    python - "${CONFIG}" "${RUN_DIR}/resolved_config.yaml" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

from tools.train.run_sam2_task_training import (
    apply_edgetam_memory_overrides,
)

config = OmegaConf.load(Path(sys.argv[1]))
apply_edgetam_memory_overrides(config)
OmegaConf.save(config, Path(sys.argv[2]), resolve=True)
print(f"Resolved E1 config: {sys.argv[2]}")
PY
  status=$?
  flock -u 7
  return "${status}"
}

ensure_wandb_run() {
  [[ "${WANDB_MODE}" == "online" ]] || return 0
  WANDB_PROJECT="${WANDB_PROJECT}" RUN_DIR="${RUN_DIR}" NAME="${NAME}" \
    python - <<'PY'
import json
import os
from pathlib import Path

import wandb

run_dir = Path(os.environ["RUN_DIR"])
run_file = run_dir / "wandb/wandb_run.json"
run_file.parent.mkdir(parents=True, exist_ok=True)
run_id = None
if run_file.is_file():
    run_id = json.loads(run_file.read_text(encoding="utf-8"))["run_id"]
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    name=os.environ["NAME"],
    id=run_id,
    resume="must" if run_id else None,
    dir=str(run_file.parent),
    config={
        "experiment": os.environ["NAME"],
        "training": False,
        "image_source": "A02 TinyViT-21M",
        "non_image_source": "official EdgeTAM",
        "selection_split": "sav_val",
    },
)
run_file.write_text(
    json.dumps(
        {
            "run_id": run.id,
            "url": run.url,
            "entity": run.entity,
            "project": os.environ["WANDB_PROJECT"],
            "name": os.environ["NAME"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
run.finish()
print(f"W&B E1 run: {run.url}", flush=True)
PY
}

evaluate_split() {
  local split="$1"
  MODEL_FAMILY=edgetam \
  STAGE1_CHECKPOINT="${RUN_DIR}/checkpoints/last.pt" \
  EDGETAM_ROOT="${EDGETAM_ROOT}" \
  EDGETAM_CONFIG="${RUN_DIR}/resolved_config.yaml" \
  SAM2_ROOT="${SAM2_TRAINING_ROOT}" \
  EXPERIMENT="${NAME}" \
  RUN_DIR="${RUN_DIR}" \
  AGGREGATE_CSV="${BEHAVIOR_ROOT}/all_metrics.csv" \
  SAV_ROOT="${SAV_ROOT}" \
  SAV_SPLIT="${split}" \
  MAX_VIDEOS=0 \
  EVAL_GPUS="${FULL_EVAL_GPUS}" \
  SKIP_DONE="${SKIP_DONE}" \
  CLEAN_PREDICTIONS=1 \
    scripts/company/25_benchmark_stage1_sav_test.sh
}

evaluate_all() {
  exec 9>"${BEHAVIOR_ROOT}/${NAME}/.pipeline.lock" || return 1
  flock 9 || return 1
  ensure_wandb_run || return 1
  evaluate_split sav_val || return 1
  evaluate_split sav_test || return 1
  if [[ "${WANDB_MODE}" == "online" ]]; then
    env -u WANDB_RUN_ID python tools/train/log_task_eval_to_wandb.py \
      --run-file "${RUN_DIR}/wandb/wandb_run.json" \
      --metrics "sav_val=${RUN_DIR}/sav_val_box_benchmark/metrics.csv" \
      --metrics "sav_test=${RUN_DIR}/sav_test_box_benchmark/metrics.csv" || return 1
  fi
  python - "${RUN_DIR}/training_status.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "complete",
            "training": False,
            "note": "Strict official non-image plus A02 TinyViT transplant",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  touch "${BEHAVIOR_ROOT}/${NAME}/.pipeline_complete"
  python tools/train/summarize_mask_finetune_ablations.py record \
    --variant-dir "${BEHAVIOR_ROOT}/${NAME}" \
    --stage-dir "${RUN_DIR}" \
    --central-csv "${BEHAVIOR_ROOT}/summary.csv"
  flock -u 9
}

validate_inputs || return 1 2>/dev/null || exit 1
prepare_checkpoint || return 1 2>/dev/null || exit 1
if [[ "${ACTION}" == "prepare" ]]; then
  echo "E1 prepare status: 0"
  echo "Checkpoint: ${RUN_DIR}/checkpoints/last.pt"
  return 0 2>/dev/null || exit 0
fi
evaluate_all || return 1 2>/dev/null || exit 1
echo "E1 evaluation status: 0"
echo "Val: ${RUN_DIR}/sav_val_box_benchmark/metrics.csv"
echo "Test: ${RUN_DIR}/sav_test_box_benchmark/metrics.csv"
return 0 2>/dev/null || exit 0
