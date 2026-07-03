# Company SA-V Formal Image-Encoder Run

This run trains the TinyViT image encoder path on a SA-V shard range with
non-image SAM2 components frozen.

Default formal range:

```text
sav_000 ... sav_018
```

Default schedule:

```text
warmup:   3 epochs, train image_encoder.neck only
finetune: 15 epochs, train full image_encoder only
```

Default runtime:

```text
BATCH_SIZE=1
IMAGE_ENCODER_BATCH=8
IMAGE_ENCODER_CKPT=0
NUM_FRAMES=8
RESOLUTION=1024
```

## Prepare Data

Pull the latest code:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline
```

Move frame directories into one canonical combined root and leave shard-local
symlinks so there is no duplicate JPEG storage:

```bash
START_SHARD=0 END_SHARD=18 \
COMBINED_ROOT=/group-volume/danny-dataset/SA-V/sav000_018_formal \
MOVE_FRAMES_TO_COMBINED=1 \
scripts/company/10_run_sav_range_formal_image_encoder.sh prepare
```

If any shard still needs frame extraction:

```bash
START_SHARD=0 END_SHARD=18 \
COMBINED_ROOT=/group-volume/danny-dataset/SA-V/sav000_018_formal \
EXTRACT_MISSING_FRAMES=1 \
MOVE_FRAMES_TO_COMBINED=1 \
scripts/company/10_run_sav_range_formal_image_encoder.sh prepare
```

Verify:

```bash
cat /group-volume/danny-dataset/SA-V/sav000_018_formal/prepare_summary.json
find /group-volume/danny-dataset/SA-V/sav000_018_formal/JPEGImages_24fps -mindepth 1 -maxdepth 1 -type d | wc -l
find /group-volume/danny-dataset/SA-V/sav_000/JPEGImages_24fps -mindepth 1 -maxdepth 1 -type l | head
```

## W&B And TensorBoard

TensorBoard is written by the SAM2 trainer under:

```text
<RUN_DIR>/tensorboard
```

W&B companion logging records preflight metadata, phase runtime, throughput,
checkpoint path, TensorBoard path, and final summaries.

Default W&B projects are separated by GPU setup:

```text
sam2-distill-edgetam-formal-1gpu
sam2-distill-edgetam-formal-4gpu
```

Set W&B before running:

```bash
wandb login
export WANDB_PROJECT=sam2-distill-edgetam-formal-4gpu
export WANDB_NAME=sav000_018_4gpu_b1_ieb8_ckpt0_w3_f15
```

To resume the same W&B run after interruption:

```bash
export WANDB_RUN_ID=<id from wandb_run.json>
```

Disable W&B if needed:

```bash
NO_WANDB=1 scripts/company/10_run_sav_range_formal_image_encoder.sh 4gpu
```

## Run

One H100:

```bash
START_SHARD=0 END_SHARD=18 \
COMBINED_ROOT=/group-volume/danny-dataset/SA-V/sav000_018_formal \
RUN_NAME=sav000_018_1gpu_b1_ieb8_ckpt0_w3_f15 \
WANDB_PROJECT=sam2-distill-edgetam-formal-1gpu \
WANDB_NAME=sav000_018_1gpu_b1_ieb8_ckpt0_w3_f15 \
BATCH_SIZE=1 \
IMAGE_ENCODER_BATCH=8 \
IMAGE_ENCODER_CKPT=0 \
WARMUP_EPOCHS=3 \
FINETUNE_EPOCHS=15 \
CHECKPOINT_SAVE_FREQ=1 \
NUM_WORKERS=8 \
scripts/company/10_run_sav_range_formal_image_encoder.sh 1gpu
```

Four H100s:

```bash
START_SHARD=0 END_SHARD=18 \
COMBINED_ROOT=/group-volume/danny-dataset/SA-V/sav000_018_formal \
RUN_NAME=sav000_018_4gpu_b1_ieb8_ckpt0_w3_f15 \
WANDB_PROJECT=sam2-distill-edgetam-formal-4gpu \
WANDB_NAME=sav000_018_4gpu_b1_ieb8_ckpt0_w3_f15 \
BATCH_SIZE=1 \
IMAGE_ENCODER_BATCH=8 \
IMAGE_ENCODER_CKPT=0 \
WARMUP_EPOCHS=3 \
FINETUNE_EPOCHS=15 \
CHECKPOINT_SAVE_FREQ=1 \
NUM_WORKERS=8 \
scripts/company/10_run_sav_range_formal_image_encoder.sh 4gpu
```

Outputs:

```text
/group-volume/danny-dataset/sam2_distill/runs/sav000_018_formal_image_encoder/<RUN_NAME>/
  checkpoints/checkpoint.pt
  tensorboard/
  config_resolved.yaml
  preflight.json
  run_metadata.json
  runtime_warmup.json
  runtime_finetune.json
  summary_warmup.json
  summary_finetune.json
  formal_summary.json
  wandb_run.json
```
