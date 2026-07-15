#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LANE="${1:-}"
GPUS="${GPUS:-0,1,2,3}"
FULL_EVAL_GPUS="${FULL_EVAL_GPUS:-${GPUS}}"
DATA_ROOT="${DATA_ROOT:-/group-volume/danny-dataset}"
SAM2D_ROOT="${SAM2D_ROOT:-${DATA_ROOT}/sam2_distill}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs}"
DRY_RUN="${DRY_RUN:-0}"
SAV_ROOT="${SAV_ROOT:-/mnt/data/danny-dataset/SA-V}"
FULL_EVAL="${FULL_EVAL:-1}"

[[ -s "${MANIFEST}" ]] || { echo "missing mounted manifest: ${MANIFEST}" >&2; exit 1; }
if [[ "${FULL_EVAL}" == "1" && "${DRY_RUN}" != "1" ]]; then
  for split in sav_val sav_test; do
    for path in \
      "${SAV_ROOT}/${split}/JPEGImages_24fps" \
      "${SAV_ROOT}/${split}/Annotations_6fps" \
      "${SAV_ROOT}/${split}/${split}.txt"; do
      [[ -e "${path}" ]] || { echo "missing full-eval input: ${path}" >&2; exit 1; }
    done
  done
fi

checkpoint_complete() {
  local run_dir="$1" target="$2"
  local last="${run_dir}/checkpoints/last.pt" best="${run_dir}/checkpoints/best.pt"
  [[ -f "${last}" && -f "${best}" ]] || return 1
  python - "${last}" "${best}" "${target}" <<'PY'
import sys
from pathlib import Path
import torch
last, best, target = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if not last.is_file() or not best.is_file():
    raise SystemExit(1)
try:
    checkpoint = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
except TypeError:
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("step", 0)) >= target else 1)
PY
}

sam2_queue() {
  case "$1" in
    tv21_proj_sam21l_msehr|tv21_proj_sam21l_msehr_cos025|tv21_adapter_sam21l_msehr) echo 8gpu_tv21_main ;;
    tv11_proj_sam21l_msehr|tv5_proj_sam21l_msehr|tv11_proj_sam21l_msehr_cos025) echo 4gpu_size_scaling ;;
    tv5_proj_sam21l_msehr_cos025|tv21_proj_sam21l_image_only|tv21_proj_sam21l_hr025) echo 4gpu_loss_ablation ;;
    tv21_proj_sam21l_msehr_l1_025|tv21_adapter_sam21l_msehr_cos025|tv21_proj_sam21bplus_msehr) echo 4gpu_adapter_teacher ;;
    tv11_adapter_sam21l_msehr|tv5_adapter_sam21l_msehr|tv21_proj_sam21l_msehr_cos1) echo 4gpu_extra_adapter_cos ;;
    *) echo "unknown SAM2 recovery experiment: $1" >&2; exit 2 ;;
  esac
}

run_full_eval() {
  local family="$1" name="$2" run_dir="$3" skip_done="$4"
  local split sam2_config sam2_checkpoint
  [[ "${FULL_EVAL}" == "1" ]] || return
  sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml"
  sam2_checkpoint="${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt"
  if [[ "${name}" == *sam21bplus* ]]; then
    sam2_config="configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_checkpoint="${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_base_plus.pt"
  fi
  for split in sav_val sav_test; do
    echo "===== full ${split} ${family} ${name} on GPUs ${FULL_EVAL_GPUS} ====="
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "MODEL_FAMILY=${family} SAV_SPLIT=${split} RUN_DIR=${run_dir}"
      continue
    fi
    MODEL_FAMILY="${family}" \
    EXPERIMENT="${name}" \
    RUN_DIR="${run_dir}" \
    SAV_ROOT="${SAV_ROOT}" \
    SAV_SPLIT="${split}" \
    CHECKPOINT_ROOT="${SAM2D_ROOT}/checkpoints" \
    SAM2L_CONFIG="${sam2_config}" \
    SAM2L_CHECKPOINT="${sam2_checkpoint}" \
    SAM31_CHECKPOINT="${DATA_ROOT}/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt" \
    SKIP_DONE="${skip_done}" \
    MAX_VIDEOS=0 \
    MAX_IMAGE_OBJECTS=0 \
    DEVICE=cuda \
    EVAL_GPUS="${FULL_EVAL_GPUS}" \
      scripts/company/25_benchmark_stage1_sav_test.sh
  done
}

run_sam2() {
  local name="$1" target="$2" queue run_dir eval_skip_done=1
  queue="$(sam2_queue "${name}")"
  run_dir="${RUN_ROOT}/sav_stage1_ablation_v2/${queue}/${name}"
  if checkpoint_complete "${run_dir}" "${target}"; then
    echo "skip training-complete SAM2 run: ${name}"
  else
    echo "===== recover SAM2 ${name} on ${GPUS} ====="
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "EXPERIMENT=${name} RUN_DIR=${run_dir} GPUS=${GPUS} MANIFEST=${MANIFEST}"
    else
      DATA_ROOT="${DATA_ROOT}" \
      SAM2D_ROOT="${SAM2D_ROOT}" \
      MANIFEST="${MANIFEST}" \
      EXPERIMENT="${name}" \
      GPUS="${GPUS}" \
      RUN_ROOT="${RUN_ROOT}/sav_stage1_ablation_v2/${queue}" \
      RUN_DIR="${run_dir}" \
      WANDB_NAME="${queue}_${name}" \
      RESUME=auto \
      RUN_TEST_AFTER_TRAIN=0 \
      SAVE_STEP_CHECKPOINTS=0 \
      PRINT_EVERY=300 \
      BATCH_SIZE= \
      MAX_STEPS= \
      SAVE_EVERY= \
      EVAL_EVERY= \
        scripts/company/19_run_sav_stage1_ablation.sh
      eval_skip_done=0
      checkpoint_complete "${run_dir}" "${target}" || {
        echo "training did not reach target or best.pt is missing: ${run_dir}" >&2
        exit 1
      }
    fi
  fi
  run_full_eval sam2 "${name}" "${run_dir}" "${eval_skip_done}"
}

sam31_config() {
  case "$1" in
    n1_cos000_adapter_ft_w2k) echo "node1_cosine residual_dwconv 2000 0.0 0.0 k0othn97" ;;
    n1_cos025_adapter_ft_w2k) echo "node1_cosine residual_dwconv 2000 0.25 0.0 p3iow86e" ;;
    n1_cos100_adapter_ft_w2k) echo "node1_cosine residual_dwconv 2000 1.0 0.0 -" ;;
    n2_projection_cos025_ft_w2k) echo "node2_interface projection 2000 0.25 0.0 7ll2x8qv" ;;
    n2_adapter_cos025_frozen) echo "node2_interface residual_dwconv 999999999 0.25 0.0 -" ;;
    n2_adapter_cos025_ft_w0) echo "node2_interface residual_dwconv 0 0.25 0.0 -" ;;
    n3_cos150_adapter_ft_w2k) echo "node3_relations residual_dwconv 2000 1.5 0.0 -" ;;
    n3_relation010_adapter_ft_w2k) echo "node3_relations residual_dwconv 2000 0.0 0.1 -" ;;
    n3_cos025_relation010_adapter_ft_w2k) echo "node3_relations residual_dwconv 2000 0.25 0.1 -" ;;
    *) echo "unknown SAM3.1 recovery experiment: $1" >&2; exit 2 ;;
  esac
}

run_sam31() {
  local name="$1" queue adapter warmup cosine relation run_id run_dir
  local eval_skip_done=1
  read -r queue adapter warmup cosine relation run_id < <(sam31_config "${name}")
  run_dir="${RUN_ROOT}/sam31_stage1_ablation_v1/${queue}/${name}"
  if checkpoint_complete "${run_dir}" 252265; then
    echo "skip training-complete SAM3.1 run: ${name}"
  else
    echo "===== recover SAM3.1 ${name} on ${GPUS} ====="
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "RUN_DIR=${run_dir} GPUS=${GPUS} MANIFEST=${MANIFEST} WANDB_RUN_ID=${run_id}"
    else
      if [[ "${run_id}" == "-" ]]; then
        run_id=""
      fi
      DATA_ROOT="${DATA_ROOT}" \
      MANIFEST="${MANIFEST}" \
      GPUS="${GPUS}" \
      BATCH_SIZE=4 \
      NUM_WORKERS=16 \
      MAX_TRAIN_ITEMS=0 \
      MAX_STEPS= \
      EPOCHS=5 \
      LR=1e-4 \
      MIN_LR=1e-6 \
      WEIGHT_DECAY=0.05 \
      PROJECTION_WARMUP_STEPS="${warmup}" \
      LR_WARMUP_STEPS=2000 \
      LAMBDA_MSE=1.0 \
      LAMBDA_COS="${cosine}" \
      LAMBDA_RELATION="${relation}" \
      ADAPTER_MODE="${adapter}" \
      WANDB_PROJECT=sam31-distill-stage1-ablation-v1 \
      WANDB_NAME="${name}" \
      WANDB_RUN_ID="${run_id}" \
      RUN_DIR="${run_dir}" \
      RESUME=auto \
      NO_WANDB=0 \
      PRINT_EVERY=300 \
        scripts/company/26_run_sam31_stage1_tv21.sh train
      eval_skip_done=0
      checkpoint_complete "${run_dir}" 252265 || {
        echo "training did not reach target or best.pt is missing: ${run_dir}" >&2
        exit 1
      }
    fi
  fi
  run_full_eval sam31 "${name}" "${run_dir}" "${eval_skip_done}"
}

gpu_count="$(python - "${GPUS}" <<'PY'
import sys
print(len([value for value in sys.argv[1].split(",") if value.strip()]))
PY
)"
if [[ "${LANE}" == "8gpu_primary" && "${gpu_count}" -ne 8 ]]; then
  echo "8gpu_primary requires exactly 8 GPUs; got GPUS=${GPUS}" >&2
  exit 2
fi
if [[ "${LANE}" != "8gpu_primary" && "${gpu_count}" -ne 4 ]]; then
  echo "${LANE} requires exactly 4 GPUs; got GPUS=${GPUS}" >&2
  exit 2
fi

case "${LANE}" in
  8gpu_primary)
    run_sam2 tv21_proj_sam21l_msehr 126135
    run_sam2 tv21_proj_sam21l_msehr_cos025 126135
    run_sam2 tv21_adapter_sam21l_msehr 126135
    ;;
  lane1)
    run_sam2 tv11_proj_sam21l_msehr 126135
    run_sam2 tv5_proj_sam21l_msehr 63070
    run_sam2 tv11_proj_sam21l_msehr_cos025 126135
    run_sam31 n1_cos100_adapter_ft_w2k
    ;;
  lane2)
    run_sam2 tv5_proj_sam21l_msehr_cos025 63070
    run_sam2 tv21_proj_sam21l_image_only 252265
    run_sam2 tv21_proj_sam21l_hr025 252265
    run_sam31 n1_cos025_adapter_ft_w2k
    ;;
  lane3)
    run_sam2 tv21_proj_sam21l_msehr_l1_025 252265
    run_sam2 tv5_adapter_sam21l_msehr 63070
    run_sam2 tv21_proj_sam21l_msehr_cos1 252265
    run_sam31 n3_cos150_adapter_ft_w2k
    ;;
  lane4)
    run_sam31 n1_cos000_adapter_ft_w2k
    run_sam31 n2_projection_cos025_ft_w2k
    run_sam31 n2_adapter_cos025_frozen
    run_sam31 n2_adapter_cos025_ft_w0
    run_sam31 n3_relation010_adapter_ft_w2k
    ;;
  lane5)
    run_sam2 tv11_adapter_sam21l_msehr 126135
    run_sam2 tv21_adapter_sam21l_msehr_cos025 252265
    run_sam2 tv21_proj_sam21bplus_msehr 252265
    run_sam31 n3_cos025_relation010_adapter_ft_w2k
    ;;
  *)
    echo "Usage: $0 {8gpu_primary|lane1|lane2|lane3|lane4|lane5}" >&2
    exit 2
    ;;
esac
