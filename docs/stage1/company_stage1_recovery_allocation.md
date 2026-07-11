# Company Stage 1 Recovery Allocation

This allocation follows the run-progress audit from 2026-07-11. It resumes or
starts the 17 incomplete registered runs while preserving each experiment's
original world size, batch size, run directory, checkpoint, and W&B run ID.
The 7 training-complete runs are not retrained.

## Prepare the mounted manifest

Run once before starting any node:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

SAV_ROOT=/mnt/data/danny-dataset/SA-V \
NUM_WORKERS=64 \
scripts/company/33_prepare_mounted_sav_stage1_manifest.sh
```

This preserves the corrected manifest row order and exact 807,248-frame train
selection while rebasing image paths to the mounted release. It does not scan
all 812,972 JPEGs.

## Node allocation

All commands use the shared mounted manifest:

```text
/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps_mounted_v1401.parquet
```

### 8 H100 node 1

Runs the two incomplete experiments from the original 8-GPU queue:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline
WANDB_MODE=online GPUS=0,1,2,3,4,5,6,7 \
scripts/company/34_run_stage1_recovery_lane.sh 8gpu_primary
```

### 8 H100 node 2

Runs two independent 4-GPU lanes concurrently:

```bash
set -euo pipefail
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline
mkdir -p /user-volume/stage1_recovery_logs

WANDB_MODE=online GPUS=0,1,2,3 \
scripts/company/34_run_stage1_recovery_lane.sh lane1 \
  2>&1 | tee /user-volume/stage1_recovery_logs/lane1.log &
pid1=$!

WANDB_MODE=online GPUS=4,5,6,7 \
scripts/company/34_run_stage1_recovery_lane.sh lane2 \
  2>&1 | tee /user-volume/stage1_recovery_logs/lane2.log &
pid2=$!

wait "${pid1}"
wait "${pid2}"
```

### 4 H100 node 1

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline
WANDB_MODE=online GPUS=0,1,2,3 \
scripts/company/34_run_stage1_recovery_lane.sh lane3
```

### 4 H100 node 2

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline
WANDB_MODE=online GPUS=0,1,2,3 \
scripts/company/34_run_stage1_recovery_lane.sh lane4
```

### 4 H100 node 3

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline
WANDB_MODE=online GPUS=0,1,2,3 \
scripts/company/34_run_stage1_recovery_lane.sh lane5
```

The lane launcher reads `last.pt` before every run. A run that has already
reached its registered target and has `best.pt` is skipped. Resume uses the
same run directory and W&B ID. Only `last.pt` and `best.pt` are retained.

## Evaluation boundary

These recovery lanes finish training and validation-based `best.pt` selection.
They intentionally disable the old test-only hook. Full downstream box-prompt
image and memory-VOS evaluation on both complete `sav_val` and `sav_test` is a
separate gate. SAM2 evaluation can reuse the existing benchmark components;
SAM3.1 first needs a verified student-trunk splice into the official downstream
model. Do not label a run complete until the progress audit reports full val
and test coverage.
