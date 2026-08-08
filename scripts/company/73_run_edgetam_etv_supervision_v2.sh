#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1

ACTION="${1:-describe}"
REQUESTED_VARIANT="${2:-}"
RUNNER="scripts/company/49_run_edgetam_memory_ablation.sh"
GPUS="${GPUS:-0,1,2,3}"
SAM2D_ROOT="${SAM2D_ROOT:-/group-volume/danny-dataset/sam2_distill}"
SAV_ROOT="${SAV_ROOT:-/group-volume/danny-dataset/SA-V}"
MANIFEST="${MANIFEST:-${SAM2D_ROOT}/manifests/sav_train_6fps_full.parquet}"
RUN_ROOT="${RUN_ROOT:-${SAM2D_ROOT}/runs/edgetam_etv_supervision_v2}"
WANDB_PROJECT="${WANDB_PROJECT:-sam2-edgetam-etv-supervision-v2}"
GATE_VIDEOS="${GATE_VIDEOS:-32}"
VARIANTS=(
  ETD0_task_only_500
  ETD1_hiera_l_logits_500
  ETD2_m0_logits_500
  ETD3_m0_memlogits_500
)
SELECTED_VARIANTS=("${VARIANTS[@]}")
if [[ -n "${REQUESTED_VARIANT}" ]]; then
  SELECTED_VARIANTS=("${REQUESTED_VARIANT}")
fi

validate_selection() {
  local requested candidate
  for requested in "${SELECTED_VARIANTS[@]}"; do
    for candidate in "${VARIANTS[@]}"; do
      [[ "${requested}" == "${candidate}" ]] && continue 2
    done
    echo "[ERROR] Unknown ETV supervision variant: ${requested}" >&2
    return 2
  done
}

runtime_preflight() {
  python - <<'PY'
import importlib

modules = (
    "hydra",
    "omegaconf",
    "iopath",
    "fvcore",
    "tensordict",
    "tensorboard",
    "pandas",
    "pyarrow",
    "skimage",
    "timm",
    "wandb",
)
missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing.append(error.name)
if missing:
    raise SystemExit("missing Python modules: " + ", ".join(sorted(set(missing))))

import hydra
import omegaconf

print(
    "Python runtime: PASS "
    f"hydra-core={hydra.__version__} "
    f"omegaconf={omegaconf.__version__}"
)
PY
  local status="$?"
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] Install the EdgeTAM runtime in this container:" >&2
    echo "  python -m pip install --user -r requirements-edgetam.txt" >&2
  fi
  return "${status}"
}

run_variant() {
  local variant="$1"
  echo "===== ${variant}: 500 updates + ${GATE_VIDEOS}-video gate ====="
  SAM2D_ROOT="${SAM2D_ROOT}" \
  SAV_ROOT="${SAV_ROOT}" \
  MANIFEST="${MANIFEST}" \
  GPUS="${GPUS}" \
  FULL_EVAL_GPUS="${GPUS}" \
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/summary.csv" \
  EDGETAM_EVAL_MODE=gate \
  EDGETAM_GATE_MAX_VIDEOS="${GATE_VIDEOS}" \
  EDGETAM_GATE_ENFORCE=0 \
  EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE:-1}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
  LOG_EVERY="${LOG_EVERY:-10}" \
    "${RUNNER}" run "${variant}"
}

smoke_variant() {
  local variant="$1"
  echo "===== ${variant}: 1-update interface smoke ====="
  SAM2D_ROOT="${SAM2D_ROOT}" \
  SAV_ROOT="${SAV_ROOT}" \
  MANIFEST="${MANIFEST}" \
  GPUS="${GPUS}" \
  EDGETAM_MEMORY_ROOT="${RUN_ROOT}/smoke" \
  EDGETAM_MEMORY_SUMMARY_CSV="${RUN_ROOT}/smoke/summary.csv" \
  EDGETAM_EVAL_MODE=none \
  EDGETAM_MEMORY_SKIP_DONE="${SKIP_DONE:-1}" \
  TASK_MAX_VIDEOS_OVERRIDE=24 \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE=disabled \
  TASK_NUM_WORKERS="${TASK_NUM_WORKERS:-8}" \
  LOG_EVERY=1 \
    "${RUNNER}" run "${variant}"
}

describe() {
  validate_selection || return $?
  echo "EdgeTAM ETV supervision v2 diagnostic"
  echo "GPUs: ${GPUS}"
  echo "Data: ${MANIFEST}"
  echo "Train: 12,000 usable videos, T4, global batch 24, exactly 500 updates"
  echo "Eval: fixed ${GATE_VIDEOS}-video SA-V val gate; no test"
  echo "Run root: ${RUN_ROOT}"
  echo "W&B: ${WANDB_PROJECT}"
  local variant
  for variant in "${SELECTED_VARIANTS[@]}"; do
    EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
      "${RUNNER}" describe "${variant}" || return $?
  done
}

audit() {
  local path status=0
  validate_selection || return $?
  runtime_preflight || status="$?"
  for path in \
    "${MANIFEST}" \
    "${SAV_ROOT}/sav_val/sav_val.txt" \
    "${SAM2D_ROOT}/runs/edgetam_tv21_sam21l_v1/formal/ETV1_image_reanchor_1ep/main/checkpoints/last.pt" \
    "${SAM2D_ROOT}/runs/edgetam_memory_ablation_v1/M0_sam2_mem4/main/checkpoints/last.pt" \
    "${SAM2D_ROOT}/runs/edgetam_memory_ablation_v1/M0_sam2_mem4/main/resolved_config.yaml" \
    "${SAM2D_ROOT}/checkpoints/sam2.1/sam2.1_hiera_large.pt" \
    "${SAM2D_ROOT}/checkpoints/edgetam/edgetam.pt"; do
    if [[ ! -e "${path}" ]]; then
      echo "[ERROR] Missing required path: ${path}" >&2
      status=1
    fi
  done
  if [[ "${status}" -eq 0 ]]; then
    SAM2D_ROOT="${SAM2D_ROOT}" \
    SAV_ROOT="${SAV_ROOT}" \
    MANIFEST="${MANIFEST}" \
    EDGETAM_MEMORY_ROOT="${RUN_ROOT}" \
      "${RUNNER}" describe "${SELECTED_VARIANTS[0]}" || status="$?"
  fi
  echo "ETV supervision audit status: ${status}"
  return "${status}"
}

run_all() {
  validate_selection || return $?
  audit || return $?
  local variant
  for variant in "${SELECTED_VARIANTS[@]}"; do
    smoke_variant "${variant}" || return $?
  done
  for variant in "${SELECTED_VARIANTS[@]}"; do
    run_variant "${variant}" || return $?
  done
}

status() {
  local variant run_dir checkpoint metrics diagnostics
  validate_selection || return $?
  for variant in "${SELECTED_VARIANTS[@]}"; do
    run_dir="${RUN_ROOT}/${variant}/main"
    checkpoint="${run_dir}/checkpoints/last.pt"
    metrics="${run_dir}/sav_val_gate${GATE_VIDEOS}_box_benchmark/metrics.csv"
    diagnostics="${run_dir}/gradient_diagnostics.json"
    echo "===== ${variant} ====="
    if [[ -f "${checkpoint}" ]]; then
      python - "${checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
steps = checkpoint.get("steps", {})
if isinstance(steps, dict):
    steps = steps.get("train", steps)
print(f"epoch={checkpoint.get('epoch', '-')} train_steps={steps}")
PY
    else
      echo "checkpoint: pending"
    fi
    if [[ -f "${diagnostics}" ]]; then
      python - "${diagnostics}" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
def number(value, spec):
    return format(value, spec) if value is not None else "pending"

print(
    "grad "
    f"status={d['status']} "
    f"mean={number(d.get('pre_clip_norm_mean'), '.4f')} "
    f"max={number(d.get('pre_clip_norm_max'), '.4f')} "
    f"clip_fraction={number(d.get('clip_fraction'), '.2%')} "
    f"nonfinite={d['nonfinite_steps']}"
)
PY
    else
      echo "gradient diagnostics: pending"
    fi
    if [[ -f "${metrics}" ]]; then
      column -s, -t < "${metrics}"
    else
      echo "mini-val: pending"
    fi
  done
  if [[ -f "${RUN_ROOT}/summary.csv" ]]; then
    echo "===== Central summary ====="
    column -s, -t < "${RUN_ROOT}/summary.csv"
  fi
}

compact_results() {
  validate_selection || return $?
  python - "${RUN_ROOT}" "/user-volume/log/edgetam_etv_supervision_v2" \
    "${GATE_VIDEOS}" "${SELECTED_VARIANTS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import torch

run_root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
gate_videos = int(sys.argv[3])
variants = sys.argv[4:]


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_steps(path):
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    steps = checkpoint.get("steps", {})
    if isinstance(steps, dict):
        steps = steps.get("train", steps)
    return int(steps)


def gate_metrics(path):
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row.get("mode") == "image":
            result["miou"] = row.get("mIoU")
            result["ap"] = row.get("AP")
        elif row.get("mode") == "video_tracking":
            result["jf"] = row.get("J&F")
    return result


rows = []
for variant in variants:
    stage = run_root / variant / "main"
    checkpoint = stage / "checkpoints" / "last.pt"
    gradients = read_json(stage / "gradient_diagnostics.json")
    gate = read_json(stage / "gate_status.json")
    metrics = gate_metrics(
        stage / f"sav_val_gate{gate_videos}_box_benchmark" / "metrics.csv"
    )
    log_path = log_root / variant / "run.log"
    affine_skips = None
    if log_path.is_file():
        affine_skips = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).count("Skip RandomAffine for zero-area mask")
    missing = []
    if not checkpoint.is_file():
        missing.append("ckpt")
    if not gradients:
        missing.append("grad")
    if not metrics.get("jf"):
        missing.append("val")
    rows.append(
        {
            "variant": variant.split("_", 1)[0],
            "steps": checkpoint_steps(checkpoint),
            "grad_mean": gradients.get("pre_clip_norm_mean"),
            "grad_max": gradients.get("pre_clip_norm_max"),
            "clip_fraction": gradients.get("clip_fraction"),
            "nonfinite": gradients.get("nonfinite_steps"),
            "miou": metrics.get("miou"),
            "ap": metrics.get("ap"),
            "jf": metrics.get("jf"),
            "gate": gate.get("status", "pending"),
            "affine_skips": affine_skips,
            "missing": ",".join(missing) or "-",
        }
    )

baseline = next(
    (float(row["jf"]) for row in rows if row["variant"] == "ETD0" and row["jf"]),
    None,
)


def number(value, digits=3):
    return "-" if value is None or value == "" else f"{float(value):.{digits}f}"


header = (
    f"{'variant':<7} {'steps':>5} {'grad_mean':>10} {'grad_max':>9} "
    f"{'clip%':>7} {'nan':>3} {'mIoU':>7} {'AP':>7} {'J&F':>6} "
    f"{'dJ&F':>6} {'gate':>7} {'affine':>7} {'skip%':>7} {'missing'}"
)
print(header)
for row in rows:
    jf = float(row["jf"]) if row["jf"] not in (None, "") else None
    delta = jf - baseline if jf is not None and baseline is not None else None
    skips = row["affine_skips"]
    skip_rate = 100 * skips / 12000 if skips is not None else None
    print(
        f"{row['variant']:<7} "
        f"{str(row['steps'] if row['steps'] is not None else '-'):>5} "
        f"{number(row['grad_mean'], 4):>10} "
        f"{number(row['grad_max'], 4):>9} "
        f"{number(100 * row['clip_fraction'] if row['clip_fraction'] is not None else None, 2):>7} "
        f"{str(row['nonfinite'] if row['nonfinite'] is not None else '-'):>3} "
        f"{number(row['miou'], 4):>7} {number(row['ap'], 4):>7} "
        f"{number(jf, 1):>6} {number(delta, 1):>6} "
        f"{row['gate']:>7} "
        f"{str(skips if skips is not None else '-'):>7} "
        f"{number(skip_rate, 2):>7} {row['missing']}"
    )

print(f"mini_val_videos={gate_videos} affine_skip_denominator=12000")
print(f"run_root={run_root}")
PY
}

STATUS=0
case "${ACTION}" in
  describe)
    describe || STATUS="$?"
    ;;
  audit)
    audit || STATUS="$?"
    ;;
  run)
    run_all || STATUS="$?"
    ;;
  status)
    status || STATUS="$?"
    ;;
  compact|summarize)
    compact_results || STATUS="$?"
    ;;
  *)
    echo "Usage: $0 {describe|audit|run|status|compact|summarize} [VARIANT]" >&2
    STATUS=2
    ;;
esac

echo "ETV supervision v2 status: ${STATUS}"
echo "Run root: ${RUN_ROOT}"
return "${STATUS}" 2>/dev/null || exit "${STATUS}"
