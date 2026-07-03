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

W&B logging records preflight metadata, phase runtime, throughput, checkpoint
path, TensorBoard path, final summaries, and synced TensorBoard scalars from
rank 0. The synced training curves include the SAM2 trainer scalars such as:

```text
Losses/train_all_loss
Losses/train_all_loss_mask
Losses/train_all_loss_dice
Losses/train_all_loss_iou
Losses/train_all_loss_class
Trainer/epoch
Trainer/steps_train
Trainer/where
```

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
MOVE_FRAMES_TO_COMBINED=1 \
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
MOVE_FRAMES_TO_COMBINED=1 \
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

## TinyViT 11M And 5M 4-GPU Runs

The 11M and 5M open pretrained timm checkpoints are 224px ImageNet-22k
distillation checkpoints fine-tuned on ImageNet-1k. The run still trains with
`RESOLUTION=1024`; only the classifier-pretraining checkpoint family differs.

Download both checkpoints:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
python -m pip install -U huggingface_hub

python - <<'PY'
from pathlib import Path
from shutil import copy2

from huggingface_hub import hf_hub_download

out = Path("/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit")
out.mkdir(parents=True, exist_ok=True)

models = {
    "timm/tiny_vit_11m_224.dist_in22k_ft_in1k": "tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors",
    "timm/tiny_vit_5m_224.dist_in22k_ft_in1k": "tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors",
}
for repo_id, filename in models.items():
    src = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    dst = out / filename
    copy2(src, dst)
    print(dst)
PY
```

Generate matching EdgeTAM configs:

```bash
mkdir -p /group-volume/danny-dataset/sam2_distill/configs/edgetam

python tools/edgetam/write_tinyvit_edgetam_config.py \
  --model-name tiny_vit_11m_224.dist_in22k_ft_in1k \
  --out /group-volume/danny-dataset/sam2_distill/configs/edgetam/tinyvit11m_video_distill.yaml

python tools/edgetam/write_tinyvit_edgetam_config.py \
  --model-name tiny_vit_5m_224.dist_in22k_ft_in1k \
  --out /group-volume/danny-dataset/sam2_distill/configs/edgetam/tinyvit5m_video_distill.yaml
```

Run TinyViT-11M on four H100s in the same W&B project as the 21M 4-GPU run:

```bash
START_SHARD=0 END_SHARD=18 \
COMBINED_ROOT=/group-volume/danny-dataset/SA-V/sav000_018_formal \
MOVE_FRAMES_TO_COMBINED=1 \
CONFIG=/group-volume/danny-dataset/sam2_distill/configs/edgetam/tinyvit11m_video_distill.yaml \
TINYVIT_CKPT=/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors \
RUN_NAME=sav000_018_4gpu_tinyvit11m_b1_ieb8_ckpt0_w3_f15 \
WANDB_PROJECT=sam2-distill-edgetam-formal-4gpu \
WANDB_NAME=sav000_018_4gpu_tinyvit11m_b1_ieb8_ckpt0_w3_f15 \
BATCH_SIZE=1 \
IMAGE_ENCODER_BATCH=8 \
IMAGE_ENCODER_CKPT=0 \
WARMUP_EPOCHS=3 \
FINETUNE_EPOCHS=15 \
CHECKPOINT_SAVE_FREQ=1 \
NUM_WORKERS=8 \
scripts/company/10_run_sav_range_formal_image_encoder.sh 4gpu
```

Run TinyViT-5M on four H100s:

```bash
START_SHARD=0 END_SHARD=18 \
COMBINED_ROOT=/group-volume/danny-dataset/SA-V/sav000_018_formal \
MOVE_FRAMES_TO_COMBINED=1 \
CONFIG=/group-volume/danny-dataset/sam2_distill/configs/edgetam/tinyvit5m_video_distill.yaml \
TINYVIT_CKPT=/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors \
RUN_NAME=sav000_018_4gpu_tinyvit5m_b1_ieb8_ckpt0_w3_f15 \
WANDB_PROJECT=sam2-distill-edgetam-formal-4gpu \
WANDB_NAME=sav000_018_4gpu_tinyvit5m_b1_ieb8_ckpt0_w3_f15 \
BATCH_SIZE=1 \
IMAGE_ENCODER_BATCH=8 \
IMAGE_ENCODER_CKPT=0 \
WARMUP_EPOCHS=3 \
FINETUNE_EPOCHS=15 \
CHECKPOINT_SAVE_FREQ=1 \
NUM_WORKERS=8 \
scripts/company/10_run_sav_range_formal_image_encoder.sh 4gpu
```
